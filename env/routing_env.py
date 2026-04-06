"""
env/routing_env.py

NetworkRoutingEnv — môi trường Gymnasium định tuyến mạng 8 node.

Vòng lặp một episode:
  reset()  →  sinh demand (src, dst, volume)
  step(action) được gọi lặp lại:
    - action = next_hop node mà agent chọn
    - env dịch chuyển "con trỏ" từ current_node → next_hop
    - Khi current_node == dst  →  terminated=True, tính reward đầy đủ
    - Nếu vượt max_hops hoặc chọn node không kết nối → truncated/penalty
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym

from network.topology import NetworkTopology
from network.traffic_generator import TrafficGenerator
from network.metrics import NetworkMetrics
from env.spaces import make_observation_space, make_action_space, NUM_NODES
from env.reward import compute_reward


class NetworkRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mean_traffic_mbps: float = 10.0,
        max_hops: int = 7,
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
        path_found = False
        reward     = 0.0

        # Kiểm tra link hợp lệ
        # if not self.topo.has_link(self._current_node, action):
        #     # Chọn node không có link → phạt nhẹ, giữ nguyên vị trí
        #     reward = -0.5
        #     info = self._info()
        #     if self.render_mode == "human":
        #         self._render_step(action, reward, "invalid link")
        #     return self._obs(), reward, terminated, truncated, info

        # Di chuyển đến next_hop
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
            truncated = True

        # Kiểm tra điều kiện kết thúc
        if self._current_node == self._dst:
            # Đến đích
            path_found = True
            terminated = True

        elif self._hops >= self.max_hops:
            # Vượt giới hạn hop
            truncated = True
        
        # utilization trung bình các link trên path
        avg_util = np.mean([
            self.topo.link(self._path[i], self._path[i+1]).utilization
            for i in range(len(self._path) - 1)
        ])
        # Tính reward thu được khi thực hiện action
        reward = compute_reward(
            path_found,
            total_delay=self._total_delay,
            dropped=self._dropped,
            hops=self._hops,
            utilization=float(avg_util),
        )

        info = self._info()
        if self.render_mode == "human":
            self._render_step(action, reward, "ok" if not self._dropped else "drop")

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
            f"cur={self._path[-2] if len(self._path)>1 else self._src} "
            f"→ next={action} | r={reward:+.3f} | {status}"
        )