"""
env/DQN/spaces.py

Định nghĩa observation space và action space cho môi trường định tuyến.

Observation (flat vector, dễ dùng với Q-table):
  - current_node       : int (0-7)
  - dst_node           : int (0-7)
  - link_states        : (num_edges × 5) — delay_norm, bw_norm, queue_size_norm, util, queue_util

Action:
  - Chọn next-hop node (0-7)  →  Discrete(8)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

NUM_NODES = 8


def make_observation_space(num_edges: int) -> spaces.Dict:
    """Tạo không gian quan sát dạng dictionary."""
    return spaces.Dict({
        "current_node": spaces.Discrete(NUM_NODES),
        "dst_node":     spaces.Discrete(NUM_NODES),
        "link_states":  spaces.Box(
            low=0.0, high=2.0,           # các giá trị có thể >1 (queue_util có thể =1, delay_norm tối đa 1)
            shape=(num_edges, 5),        # 5 đặc trưng
            dtype=np.float32,
        ),
    })


def make_action_space() -> spaces.Discrete:
    """Hành động: chọn nút tiếp theo (0..7)."""
    return spaces.Discrete(NUM_NODES)


def obs_to_flat(obs: dict) -> np.ndarray:
    """
    Chuyển observation dictionary thành vector 1D cho mạng nơ‑ron.
    Shape: (2 + num_edges * 5,)
    """
    return np.concatenate([
        np.array([obs["current_node"] / (NUM_NODES - 1),
                  obs["dst_node"]     / (NUM_NODES - 1)], dtype=np.float32),
        obs["link_states"].flatten(),
    ])