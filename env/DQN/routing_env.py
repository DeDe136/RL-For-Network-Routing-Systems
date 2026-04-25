"""
env/DQN/routing_env.py

NetworkRoutingEnv — môi trường Gymnasium định tuyến mạng 8 node.

Vòng lặp một episode:
  reset() → randomize_links + step_background → sinh demand (src,dst,vol)
  step(action) lặp lại:
    1. Kiểm tra link hợp lệ (has_link). Nếu không → phạt -0.5, giữ vị trí.
    2. Kiểm tra is_loop (node đã thăm chưa) TRƯỚC khi di chuyển.
    3. Lưu link vừa chọn (cho shaping reward).
    4. Di chuyển cur → action, tăng hops, cập nhật visited_mask.
    5. Nếu is_loop → tăng self._loop_count.
    6. send_traffic(path, volume, is_first_hop) — MDP model.
       dropped_data tích lũy nếu có overflow, KHÔNG truncated.
    7. reduce_load(path) — leaky bucket + decay ngoài path.
       dropped_data có thể tăng thêm, KHÔNG truncated.
    8. Tính reward:
       - intermediate: compute_shaping_reward(..., is_loop=is_loop)
       - terminal:     compute_final_reward(..., loop_count=self._loop_count)
    7. Kiểm tra điều kiện kết thúc.

Điều kiện kết thúc — chỉ 2 trường hợp:
  current_node == dst → terminated (đến đích)
  hops >= max_hops-1  → truncated  (hết bước)

  Không còn truncated vì drop — drop được phạt liên tục qua reward.

Observation (dict):
  current_node : int
  dst_node     : int
  visited_mask : np.ndarray (NUM_NODES,) — 1.0 nếu node đã đi qua
  link_states  : (26, 6) — delay_norm, bw_norm, queue_size_cur_norm,
                            utilization, queue_util, drop_norm

obs_to_flat() → shape (180,) = 8 + 8 + 8 + 26×6 — input cho DQN.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym

from network.DQN.topology import NetworkTopology
from network.DQN.traffic_generator import TrafficGenerator
from network.DQN.metrics import NetworkMetrics
from env.DQN.spaces import make_observation_space, make_action_space, NUM_NODES, obs_to_flat
from env.DQN.reward import compute_final_reward, compute_shaping_reward, compute_traffic_penalty


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
        self._src:               int   = 0
        self._dst:               int   = 7
        self._volume:            float = 10.0
        self._current_node:      int   = 0
        self._path:              List[int] = []
        self._hops:              int   = 0
        self._total_delay:       float = 0.0
        self._total_dropped_data:float = 0.0   # tích lũy data bị drop (packets)
        self._visited_mask:      np.ndarray = np.zeros(NUM_NODES, dtype=np.float32)
        self._loop_count:        int   = 0   # số lần đi vào node đã thăm

        # Accumulator per-link cho info — tích lũy load và queue_used_cur
        # sau mỗi send_traffic + reduce_load, KHÔNG ảnh hưởng reward.
        self._accum_load:           Dict[Tuple[int,int], float] = {}
        self._accum_queue_used_cur: Dict[Tuple[int,int], float] = {}

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
        self._current_node       = self._src
        self._path               = [self._src]
        self._hops               = 0
        self._total_delay        = 0.0
        self._total_dropped_data = 0.0
        self._loop_count         = 0
        self._accum_load           = {}
        self._accum_queue_used_cur = {}

        self._visited_mask = np.zeros(NUM_NODES, dtype=np.float32)
        self._visited_mask[self._src] = 1.0

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

        # ── Kiểm tra loop TRƯỚC khi di chuyển ────────────────────────
        # visited_mask[action] == 1.0 → node này đã đi qua rồi
        is_loop = (self._visited_mask[action] > 0.0)

        # Lưu link trước khi di chuyển (dùng cho shaping reward)
        link = self.topo.link(self._current_node, action)

        # ── Di chuyển đến next_hop ────────────────────────────────────
        self._hops         += 1
        self._current_node  = action
        self._path.append(action)

        # Cập nhật visited_mask và loop_count
        if is_loop:
            self._loop_count += 1
        self._visited_mask[action] = 1.0

        # ── send_traffic theo MDP model ───────────────────────────────
        # Khi có drop: dropped_data tích lũy trên link, queue_used_cur
        # bị clamp, nhưng KHÔNG truncated — episode tiếp tục.
        is_first_hop = (self._hops == 1)
        result = self.topo.send_traffic(
            path         = self._path,
            volume_mbps  = self._volume,
            is_first_hop = is_first_hop,
        )
        self._total_delay        += result["total_delay"]
        self._total_dropped_data += result["dropped_data"]

        # ── Phạt nhẹ load & queue_used_cur ngay sau send_traffic ─────
        # Lấy thông số vật lý của link cuối để normalize
        _last_attr = self.topo.link(self._path[-2], self._path[-1])
        _traffic_penalty = compute_traffic_penalty(
            load           = result["load"],
            queue_used_cur = result["queue_used_cur"],
            bandwidth      = _last_attr.bandwidth,
            queue_size_cur = _last_attr.queue_size_cur,
        )

        # ── Tích lũy load & queue_used_cur của link cuối sau send_traffic ─
        _last_link = (self._path[-2], self._path[-1])
        self._accum_load[_last_link] = (
            self._accum_load.get(_last_link, 0.0) + result["load"]
        )
        self._accum_queue_used_cur[_last_link] = (
            self._accum_queue_used_cur.get(_last_link, 0.0)
            + result["queue_used_cur"]
        )

        # ── reduce_load (leaky bucket) ────────────────────────────────
        # Cũng có thể gây thêm drop khi forward load → trả về float
        drop_from_reduce          = self.topo.reduce_load(self._path)
        self._total_dropped_data += drop_from_reduce

        # ── Tích lũy load & queue_used_cur toàn path sau reduce_load ──
        # Reward đã tính xong — đọc topo ở đây không ảnh hưởng reward.
        for _i in range(len(self._path) - 1):
            _lk = (self._path[_i], self._path[_i + 1])
            _attr = self.topo.link(_lk[0], _lk[1])
            self._accum_load[_lk] = (
                self._accum_load.get(_lk, 0.0) + _attr.load
            )
            self._accum_queue_used_cur[_lk] = (
                self._accum_queue_used_cur.get(_lk, 0.0)
                + _attr.queue_used_cur
            )

        # ── Metrics sau cập nhật ─────────────────────────────────────
        avg_util, avg_queue, avg_bw, avg_qsize = self._path_avg_metrics()

        # ── Điều kiện kết thúc — chỉ 2 trường hợp ────────────────────
        if self._current_node == self._dst:
            terminated = True
            reward = compute_final_reward(
                path_found         = True,
                total_delay        = self._total_delay,
                hops               = self._hops,
                utilization        = avg_util,
                avg_queue_util     = avg_queue,
                avg_bandwidth      = avg_bw,
                avg_queue_size     = avg_qsize,
                total_dropped_data = self._total_dropped_data,
                loop_count         = self._loop_count,
            ) + _traffic_penalty

        elif self._hops >= self.max_hops - 1:
            truncated = True
            reward = compute_final_reward(
                path_found         = False,
                total_delay        = self._total_delay,
                hops               = self._hops,
                utilization        = avg_util,
                avg_queue_util     = avg_queue,
                avg_bandwidth      = avg_bw,
                avg_queue_size     = avg_qsize,
                total_dropped_data = self._total_dropped_data,
                loop_count         = self._loop_count,
            ) + _traffic_penalty

        else:
            # Shaping reward: dùng trạng thái link vừa đi qua
            reward = compute_shaping_reward(
                current_link_delay       = link.delay,
                current_link_utilization = link.utilization,
                current_link_queue_util  = link.queue_util,
                current_link_bandwidth   = link.bandwidth,
                current_link_queue_size  = link.queue_size_cur,
                current_link_dropped     = link.dropped_data,
                is_loop                  = is_loop,
            ) + _traffic_penalty
        
        info = self._info()
        if self.render_mode == "human":
            status = "success" if terminated else \
                     "fail (max_hops)" if truncated else \
                     f"hop {self._hops}"
            self._render_step(action, reward, status)

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"  demand={self._src}→{self._dst} "
                f"({self._volume:.1f}Mbps) | "
                f"path={self._path} | delay={self._total_delay:.2f}ms | "
                f"loops={self._loop_count} | "
                f"dropped_data={self._total_dropped_data:.2f}"
            )

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def flat_obs(self) -> np.ndarray:
        """Flat vector (180,) — input cho DQN network."""
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
            float(np.mean([lk.utilization    for lk in links])),
            float(np.mean([lk.queue_util     for lk in links])),
            float(np.mean([lk.bandwidth      for lk in links])),
            float(np.mean([lk.queue_size_cur for lk in links])),
        )

    def _obs(self) -> Dict:
        return {
            "current_node": self._current_node,
            "dst_node":     self._dst,
            "visited_mask": self._visited_mask.copy(),
            "link_states":  self.topo.link_state_vector(),
        }

    def _info(self) -> Dict:
        """
        Info đầy đủ bao gồm chi tiết từng link trên path đã đi.
        Dùng bởi EpisodeLogger để ghi CSV.
        """
        info = {
            "src":               self._src,
            "dst":               self._dst,
            "current_node":      self._current_node,
            "path":              list(self._path),
            "hops":              self._hops,
            "total_delay":       self._total_delay,
            "total_dropped_data":self._total_dropped_data,
            # backward-compat: dropped=True nếu có bất kỳ data bị drop
            "dropped":           self._total_dropped_data > 0,
            **self.topo.summary(),
        }
        # Per-link details
        if len(self._path) >= 2:
            link_details = []
            for i in range(len(self._path) - 1):
                u, v = self._path[i], self._path[i + 1]
                lk   = self.topo.link(u, v)
                link_details.append({
                    "src_node":       u,
                    "dst_node":       v,
                    "delay":          lk.delay,
                    "bandwidth":      lk.bandwidth,
                    # load & queue_used_cur: tổng tích lũy qua send_traffic
                    # + reduce_load trong episode, không phải giá trị tức thời
                    "load":           self._accum_load.get((u, v), lk.load),
                    "utilization":    self._accum_load.get((u, v), lk.load) / lk.bandwidth,
                    "queue_size_cur": lk.queue_size_cur,
                    "queue_used_cur": self._accum_queue_used_cur.get(
                                           (u, v), lk.queue_used_cur),
                    "queue_util":     self._accum_queue_used_cur.get(
                                           (u, v), lk.queue_used_cur) / lk.queue_size_cur,
                    "dropped_data":   lk.dropped_data,
                })
            info["link_details"] = link_details
        else:
            info["link_details"] = []
        return info

    def _render_step(self, action: int, reward: float, status: str):
        print(
            f"  [{self._src}→{self._dst}] "
            f"cur={self._path[-2] if len(self._path) > 1 else self._src}"
            f"→{action} | r={reward:+.3f} | "
            f"loops={self._loop_count} | drop={self._total_dropped_data:.1f} | {status}"
        )