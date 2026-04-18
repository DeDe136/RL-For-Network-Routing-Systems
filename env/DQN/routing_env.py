"""
env/routing_env.py

NetworkRoutingEnv — môi trường Gymnasium định tuyến mạng 8 node.

Vòng lặp một episode:
  reset() → randomize_links + step_background → sinh demand (src,dst,vol)
  step(action) lặp lại:
    1. Kiểm tra link hợp lệ (has_link). Nếu không → phạt -0.5, giữ vị trí.
    2. Lưu link vừa chọn (cho shaping reward).
    3. Di chuyển cur → action, tăng hops.
    4. send_traffic(path, volume, is_first_hop) — MDP model.
    5. reduce_load(path) — leaky bucket + decay ngoài path.
       Nếu reduce_load trả về is_drop=True → _dropped = True.
    6. Tính reward từ trạng thái link.
    7. Kiểm tra điều kiện kết thúc.

Điều kiện kết thúc (if/elif — chỉ một True mỗi bước):
  dropped             → truncated, phạt nặng
  current_node == dst → terminated, thưởng
  hops >= max_hops-1  → truncated, phạt (cho agent 1 hop cuối rồi kết thúc)

Observation (dict):
  current_node : int
  dst_node     : int
  link_states  : (26, 5) — delay_norm, bw_norm, queue_size_cur_norm,
                            utilization, queue_util

obs_to_flat() → shape (132,) = 2 + 26×5 — input cho DQN.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym

from network.DQN.topology import NetworkTopology
from network.DQN.traffic_generator import TrafficGenerator
from network.DQN.metrics import NetworkMetrics
from env.DQN.spaces import make_observation_space, make_action_space, NUM_NODES, obs_to_flat
from env.DQN.reward import compute_final_reward, compute_shaping_reward


class NetworkRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mean_traffic_mbps: float = 10.0,
        max_hops:          int   = 8,
        bg_intensity:      float = 0.3,
        render_mode:       Optional[str] = None,
        seed:              int   = 42,
    ):
        super().__init__()
        self.max_hops     = max_hops
        self.bg_intensity = bg_intensity
        self.render_mode  = render_mode

        rng          = np.random.default_rng(seed)
        self.topo    = NetworkTopology(rng=rng)
        self.traffic = TrafficGenerator(mean_mbps=mean_traffic_mbps, seed=seed)
        self.metrics = NetworkMetrics(self.topo)

        self.observation_space = make_observation_space(self.topo.num_edges())
        self.action_space      = make_action_space()

        # Episode state
        self._src:          int   = 0
        self._dst:          int   = 7
        self._volume:       float = 10.0
        self._current_node: int   = 0
        self._path:         List[int] = []
        self._hops:         int   = 0
        self._total_delay:  float = 0.0
        self._dropped:      bool  = False

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        rng = np.random.default_rng(seed) if seed is not None else None
        self.topo.reset(rng=rng)
        if seed is not None:
            self.traffic.reset(seed=seed)

        # Background traffic để tạo trạng thái khởi đầu đa dạng
        self.topo.step_background(intensity=self.bg_intensity)

        self._src, self._dst, self._volume = self.traffic.generate()
        self._current_node = self._src
        self._path         = [self._src]
        self._hops         = 0
        self._total_delay  = 0.0
        self._dropped      = False

        return self._obs(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert 0 <= action < NUM_NODES, f"Invalid action {action}"

        terminated = False
        truncated  = False

        # ── Kiểm tra link hợp lệ ─────────────────────────────────────
        # Agent được training chỉ chọn valid neighbors (action masking),
        # nhưng giữ guard này để routing_env không crash nếu action sai.
        if not self.topo.has_link(self._current_node, action):
            reward = -0.5
            info   = self._info()
            if self.render_mode == "human":
                self._render_step(action, reward, "invalid link")
            return self._obs(), reward, terminated, truncated, info

        # Lưu link trước khi di chuyển (dùng cho shaping reward sau)
        link = self.topo.link(self._current_node, action)

        # ── Di chuyển đến next_hop ────────────────────────────────────
        self._hops         += 1
        self._current_node  = action
        self._path.append(action)

        # ── send_traffic theo MDP model ───────────────────────────────
        is_first_hop = (self._hops == 1)
        result = self.topo.send_traffic(
            path         = self._path,
            volume_mbps  = self._volume,
            is_first_hop = is_first_hop,
        )
        self._total_delay += result["total_delay"]
        if result["dropped"]:
            self._dropped = True

        # ── reduce_load (leaky bucket) ────────────────────────────────
        # reduce_load trả về is_drop=True nếu phát hiện drop trong quá
        # trình xử lý queue (queue_used_cur vượt queue_size_cur).
        is_drop = self.topo.reduce_load(self._path)
        if is_drop:
            self._dropped = True

        # ── Tính metrics sau cập nhật ─────────────────────────────────
        avg_util, avg_queue, avg_bw, avg_qsize = self._path_avg_metrics()

        # ── Điều kiện kết thúc (if/elif: chỉ 1 True mỗi bước) ────────
        if self._dropped:
            truncated = True
            reward = compute_final_reward(
                path_found     = False,
                total_delay    = self._total_delay,
                dropped        = True,
                hops           = self._hops,
                utilization    = avg_util,
                avg_queue_util = avg_queue,
                avg_bandwidth  = avg_bw,
                avg_queue_size = avg_qsize,
            )

        elif self._current_node == self._dst:
            terminated = True
            reward = compute_final_reward(
                path_found     = True,
                total_delay    = self._total_delay,
                dropped        = False,
                hops           = self._hops,
                utilization    = avg_util,
                avg_queue_util = avg_queue,
                avg_bandwidth  = avg_bw,
                avg_queue_size = avg_qsize,
            )

        elif self._hops >= self.max_hops - 1:
            # Cho agent 1 hop nữa rồi kết thúc, tránh bị cắt quá sớm
            truncated = True
            reward = compute_final_reward(
                path_found     = False,
                total_delay    = self._total_delay,
                dropped        = False,
                hops           = self._hops,
                utilization    = avg_util,
                avg_queue_util = avg_queue,
                avg_bandwidth  = avg_bw,
                avg_queue_size = avg_qsize,
            )

        else:
            # Shaping reward: dựa trên link vừa đi qua (đọc sau reduce_load)
            reward = compute_shaping_reward(
                current_link_delay       = link.delay,
                current_link_utilization = link.utilization,
                current_link_queue_util  = link.queue_util,
                current_link_bandwidth   = link.bandwidth,
                current_link_queue_size  = link.queue_size_cur,
            )

        info = self._info()
        if self.render_mode == "human":
            if terminated:
                status = "success"
            elif truncated:
                status = "fail (drop)" if self._dropped else "fail (max_hops)"
            else:
                status = f"hop {self._hops}"
            self._render_step(action, reward, status)

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"  demand={self._src}→{self._dst} "
                f"({self._volume:.1f}Mbps) | "
                f"path={self._path} | delay={self._total_delay:.2f}ms | "
                f"drop={self._dropped}"
            )

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def flat_obs(self) -> np.ndarray:
        """Flat vector (132,) — input cho DQN network."""
        return obs_to_flat(self._obs())

    def _path_avg_metrics(self) -> Tuple[float, float, float, float]:
        """
        (avg_utilization, avg_queue_util, avg_bandwidth, avg_queue_size_cur)
        trên path hiện tại.
        """
        if len(self._path) < 2:
            return 0.0, 0.0, 0.0, 0.0
        links = [
            self.topo.link(self._path[i], self._path[i + 1])
            for i in range(len(self._path) - 1)
        ]
        return (
            float(np.mean([lk.utilization     for lk in links])),
            float(np.mean([lk.queue_util      for lk in links])),
            float(np.mean([lk.bandwidth       for lk in links])),
            float(np.mean([lk.queue_size_cur  for lk in links])),
        )

    def _obs(self) -> Dict:
        return {
            "current_node": self._current_node,
            "dst_node":     self._dst,
            "link_states":  self.topo.link_state_vector(),
        }

    def _info(self) -> Dict:
        """
        Info đầy đủ bao gồm chi tiết từng link trên path đã đi.
        Dùng bởi EpisodeLogger để ghi CSV.
        """
        info = {
            "src":          self._src,
            "dst":          self._dst,
            "current_node": self._current_node,
            "path":         list(self._path),
            "hops":         self._hops,
            "total_delay":  self._total_delay,
            "dropped":      self._dropped,
            **self.topo.summary(),
        }
        # Per-link details
        if len(self._path) >= 2:
            link_details = []
            for i in range(len(self._path) - 1):
                u, v = self._path[i], self._path[i + 1]
                lk = self.topo.link(u, v)
                link_details.append({
                    "src_node":       u,
                    "dst_node":       v,
                    "delay":          lk.delay,
                    "bandwidth":      lk.bandwidth,
                    "load":           lk.load,
                    "utilization":    lk.utilization,
                    "queue_size_cur": lk.queue_size_cur,
                    "queue_used_cur": lk.queue_used_cur,
                    "queue_util":     lk.queue_util,
                })
            info["link_details"] = link_details
        else:
            info["link_details"] = []
        return info

    def _render_step(self, action: int, reward: float, status: str):
        print(
            f"  [{self._src}→{self._dst}] "
            f"cur={self._path[-2] if len(self._path) > 1 else self._src}"
            f"→{action} | r={reward:+.3f} | {status}"
        )