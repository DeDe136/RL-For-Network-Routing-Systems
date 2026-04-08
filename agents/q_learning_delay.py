"""
agents/q_learning.py

Q-Learning thuần (tabular) cho bài toán định tuyến mạng 8 node.

State = (current_node, dst_node)  →  8 × 8 = 64 states
Action = next_hop node            →  8 actions

Q-table shape: (8, 8, 8)
  Q[current_node][dst_node][next_hop] = expected return

Update rule (Bellman):
  Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') − Q(s,a)]
"""

import numpy as np
import os
from typing import Any, Dict, Tuple

from agents.base_agent import BaseAgent

NUM_NODES = 8


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning.

    State encoding: tuple (current_node, dst_node)
    Action        : next_hop node (0..7)
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        super().__init__(
            n_states  = NUM_NODES * NUM_NODES,
            n_actions = NUM_NODES,
            config    = cfg,
        )
        self.alpha   = cfg.get("alpha",   0.1)    # learning rate
        self.gamma   = cfg.get("gamma",   0.99)   # discount
        self.epsilon = cfg.get("epsilon", 1.0)    # exploration
        self.eps_min = cfg.get("eps_min", 0.1)
        self.eps_decay = cfg.get("eps_decay", 0.99999585)

        # Q-table: shape (current_node, dst_node, next_hop)
        self.Q = np.zeros((NUM_NODES, NUM_NODES, NUM_NODES), dtype=np.float64)

        # Neighbors mask — agent chỉ được chọn node có kết nối
        # Được set từ bên ngoài sau khi topo khởi tạo
        self._neighbor_mask: np.ndarray = np.ones(
            (NUM_NODES, NUM_NODES), dtype=bool
        )

    def set_neighbor_mask(self, adjacency: np.ndarray):
        """
        adjacency: (8,8) float — adj_matrix của NetworkTopology.
        Chỉ cho phép action là neighbor thực sự.
        """
        self._neighbor_mask = adjacency.astype(bool)

    def _valid_actions(self, current_node: int) -> np.ndarray:
        """Trả về các action hợp lệ (có link) từ current_node."""
        return np.where(self._neighbor_mask[current_node])[0]

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

    def select_action(self, state: Tuple[int, int]) -> int:
        """
        ε-greedy: với xác suất ε chọn ngẫu nhiên trong valid actions,
        còn lại chọn argmax Q.
        """
        current_node, dst_node = state
        valid = self._valid_actions(current_node)

        if not self.training:
            # Greedy hoàn toàn khi eval
            q_vals = self.Q[current_node, dst_node, :]
            max_q = max(q_vals[a] for a in valid)
            best_actions = [a for a in valid if q_vals[a] == max_q]
            return int(np.random.choice(best_actions))

        # ε-greedy
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid))

        q_vals = self.Q[current_node, dst_node, :]
        max_q = max(q_vals[a] for a in valid)
        best_actions = [a for a in valid if q_vals[a] == max_q]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state:      Tuple[int, int],
        action:     int,
        reward:     float,
        next_state: Tuple[int, int],
        done:       bool,
    ) -> Dict[str, float]:
        """
        Bellman update.
        """
        cur, dst = state
        nxt, _   = next_state

        # Target
        if done:
            target = reward
        else:
            valid_next = self._valid_actions(nxt)
            if len(valid_next) == 0:
                target = reward
            else:
                best_next = np.max(self.Q[nxt, dst, valid_next])
                target = reward + self.gamma * best_next

        # Q update
        td_error = target - self.Q[cur, dst, action]
        self.Q[cur, dst, action] += self.alpha * td_error

        # Decay ε
        if self.training:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        return {"td_error": abs(td_error), "epsilon": self.epsilon}

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, self.Q)
        print(f"Q-table saved → {path}")

    def load(self, path: str):
        self.Q = np.load(path)
        print(f"Q-table loaded ← {path}")

    # ------------------------------------------------------------------ #
    #  Debug helpers                                                       #
    # ------------------------------------------------------------------ #

    def best_path(self, src: int, dst: int, max_hops: int = 8) -> list:
        """
        Greedy rollout từ src đến dst theo Q-table hiện tại.
        Dùng để kiểm tra policy đã học.
        """
        path = [src]
        cur  = src
        visited = {src}
        for _ in range(max_hops):
            if cur == dst:
                break
            valid = [a for a in self._valid_actions(cur) if a not in visited]
            if not valid:
                break
            max_q = max(self.Q[cur, dst, a] for a in valid)
            best_actions = [a for a in valid if self.Q[cur, dst, a] == max_q]
            nxt = np.random.choice(best_actions)
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        return path

    def q_table_summary(self) -> str:
        """In thống kê Q-table để debug."""
        nonzero = np.count_nonzero(self.Q)
        total   = self.Q.size
        return (
            f"Q-table: {NUM_NODES}×{NUM_NODES}×{NUM_NODES}={total} entries | "
            f"nonzero={nonzero} ({100*nonzero/total:.1f}%) | "
            f"min={self.Q.min():.3f} max={self.Q.max():.3f} "
            f"mean={self.Q.mean():.3f}"
        )