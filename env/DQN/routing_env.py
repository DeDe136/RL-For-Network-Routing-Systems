"""
env/DQN/routing_env.py

NetworkRoutingEnv — môi trường Gymnasium định tuyến mạng 8 node.

Vòng lặp một episode:
  reset() → randomize_links + step_background → sinh demand (src,dst,vol)
  step(action) lặp lại:
    1. Di chuyển cur → action.
    2. send_traffic([...path...], volume, is_first_hop) — MDP model.
    3. reduce_load(path) — leaky bucket trên path + decay ngoài path.
    4. Tính reward từ trạng thái link vừa đi qua.
    5. Kiểm tra terminated / truncated.

Điều kiện kết thúc (if/elif — chỉ một True mỗi bước):
  dropped             → truncated, phạt nặng
  current_node == dst → terminated, thưởng
  hops >= max_hops    → truncated, phạt

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

        # Lưu link vừa đi qua (để dùng cho shaping reward)
        link = self.topo.link(self._current_node, action)

        # Di chuyển đến next_hop
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
        is_drop = self.topo.reduce_load(self._path)
        if is_drop:
            self._dropped = True

        # ── Tính reward ───────────────────────────────────────────────
        # Đọc sau reduce_load để reward phản ánh trạng thái sau cập nhật
        avg_util, avg_queue = self._path_avg_metrics()

        # Điều kiện kết thúc — if/elif đảm bảo chỉ 1 True mỗi bước
        if self._dropped:
            truncated = True
            reward = compute_final_reward(
                path_found     = False,
                total_delay    = self._total_delay,
                dropped        = True,
                hops           = self._hops,
                utilization    = avg_util,
                avg_queue_util = avg_queue,
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
            )

        elif self._hops >= self.max_hops:
            truncated = True
            reward = compute_final_reward(
                path_found     = False,
                total_delay    = self._total_delay,
                dropped        = False,
                hops           = self._hops,
                utilization    = avg_util,
                avg_queue_util = avg_queue,
            )

        else:
            # Shaping reward: dùng trạng thái link vừa đi qua
            reward = compute_shaping_reward(
                current_link_delay       = link.delay,
                current_link_utilization = link.utilization,
                current_link_queue_util  = link.queue_util,
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

    def _path_avg_metrics(self) -> Tuple[float, float]:
        """(avg_utilization, avg_queue_util) trên path hiện tại."""
        if len(self._path) < 2:
            return 0.0, 0.0
        links = [
            self.topo.link(self._path[i], self._path[i + 1])
            for i in range(len(self._path) - 1)
        ]
        return (
            float(np.mean([lk.utilization for lk in links])),
            float(np.mean([lk.queue_util  for lk in links])),
        )

    def _obs(self) -> Dict:
        return {
            "current_node": self._current_node,
            "dst_node":     self._dst,
            "link_states":  self.topo.link_state_vector(),
        }

    def _info(self) -> Dict:
        return {
            "src":          self._src,
            "dst":          self._dst,
            "current_node": self._current_node,
            "path":         list(self._path),
            "hops":         self._hops,
            "total_delay":  self._total_delay,
            "dropped":      self._dropped,
            **self.topo.summary(),
        }

    def _render_step(self, action: int, reward: float, status: str):
        print(
            f"  [{self._src}→{self._dst}] "
            f"cur={self._path[-2] if len(self._path) > 1 else self._src}"
            f"→{action} | r={reward:+.3f} | {status}"
        )