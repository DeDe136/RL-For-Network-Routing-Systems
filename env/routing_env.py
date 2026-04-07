"""
env/routing_env.py

NetworkRoutingEnv — môi trường Gymnasium định tuyến mạng 8 node.

Vòng lặp một episode:
  reset()  →  sinh demand (src, dst, volume)
  step(action) được gọi lặp lại:
    - action = next_hop node mà agent chọn
    - env dịch chuyển "con trỏ" từ current_node → next_hop
    - Khi current_node == dst và không drop → terminated=True
    - Khi drop hoặc vượt max_hops → truncated=True
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym

from network.topology import NetworkTopology
from network.traffic_generator import TrafficGenerator
from network.metrics import NetworkMetrics
from env.spaces import make_observation_space, make_action_space, NUM_NODES
from env.reward import compute_final_reward, compute_shaping_reward


class NetworkRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mean_traffic_mbps: float = 10.0,
        max_hops: int = 8,
        render_mode: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.max_hops    = max_hops
        self.render_mode = render_mode

        # Network components
        self.topo    = NetworkTopology()
        self.traffic = TrafficGenerator(mean_mbps=mean_traffic_mbps, seed=seed)
        self.metrics = NetworkMetrics(self.topo)

        # Gym spaces
        self.observation_space = make_observation_space(self.topo.num_edges())
        self.action_space      = make_action_space()

        # Episode state
        self._src: int = 0
        self._dst: int = 7
        self._volume: float = 10.0
        self._current_node: int = 0
        self._path: List[int] = []
        self._hops: int = 0
        self._total_delay: float = 0.0
        self._dropped: bool = False

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        self.topo.reset()
        if seed is not None:
            self.traffic.reset(seed=seed)

        self._src, self._dst, self._volume = self.traffic.generate()
        self._current_node = self._src
        self._path = [self._src]
        self._hops = 0
        self._total_delay = 0.0
        self._dropped = False

        return self._obs(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert 0 <= action < NUM_NODES, f"Invalid action {action}"

        terminated = False
        truncated  = False

        # Di chuyển đến next_hop
        # (action luôn hợp lệ vì agent chỉ chọn trong valid neighbors)
        link = self.topo.link(self._current_node, action)
        self._total_delay += link.delay
        self._hops += 1
        self._current_node = action
        self._path.append(action)

        # Gửi traffic trên đoạn link vừa đi qua
        result = self.topo.send_traffic(
            [self._path[-2], self._path[-1]], self._volume
        )
        if result["dropped"]:
            self._dropped = True

        # ── Điều kiện kết thúc: if/elif đảm bảo chỉ một trong
        #    terminated/truncated được True tại mỗi bước ──────────────
        if self._dropped:
            # Drop → truncated, phạt nặng.
            # Ưu tiên cao nhất: drop thắng kể cả khi current_node == dst.
            truncated = True
            avg_util, avg_queue = self._path_avg_metrics()
            reward = compute_final_reward(
                path_found=False,
                total_delay=self._total_delay,
                dropped=True,
                hops=self._hops,
                utilization=avg_util,
                avg_queue_util=avg_queue,
            )

        elif self._current_node == self._dst:
            # Đến đích thành công, không có drop
            terminated = True
            avg_util, avg_queue = self._path_avg_metrics()
            reward = compute_final_reward(
                path_found=True,
                total_delay=self._total_delay,
                dropped=False,
                hops=self._hops,
                utilization=avg_util,
                avg_queue_util=avg_queue,
            )

        elif self._hops >= self.max_hops:
            # Hết hop mà chưa đến đích
            truncated = True
            avg_util, avg_queue = self._path_avg_metrics()
            reward = compute_final_reward(
                path_found=False,
                total_delay=self._total_delay,
                dropped=False,
                hops=self._hops,
                utilization=avg_util,
                avg_queue_util=avg_queue,
            )

        else:
            # Step trung gian: shaping reward dựa trên link vừa đi qua
            reward = compute_shaping_reward(
                current_link_delay=link.delay,
                current_link_utilization=link.utilization,
                current_link_queue_util=link.queue_util,
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
                f"  demand={self._src}→{self._dst} ({self._volume:.1f}Mbps) | "
                f"path={self._path} | delay={self._total_delay:.1f}ms | "
                f"drop={self._dropped}"
            )

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _path_avg_metrics(self) -> Tuple[float, float]:
        """Trả về (avg_utilization, avg_queue_util) trên path hiện tại."""
        if len(self._path) < 2:
            return 0.0, 0.0
        links = [
            self.topo.link(self._path[i], self._path[i + 1])
            for i in range(len(self._path) - 1)
        ]
        avg_util  = float(np.mean([lk.utilization  for lk in links]))
        avg_queue = float(np.mean([lk.queue_util   for lk in links]))
        return avg_util, avg_queue

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
            f"cur={self._path[-2] if len(self._path) > 1 else self._src} "
            f"→ next={action} | r={reward:+.3f} | {status}"
        )