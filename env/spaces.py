"""
env/spaces.py

Định nghĩa observation space và action space.

Observation (flat vector, dễ dùng với Q-table):
  - current_node       : int (0-7)
  - dst_node           : int (0-7)
  - link_states        : (num_edges × 4) — delay_norm, bw_norm, util, queue_util

Action:
  - Chọn next-hop node (0-7)  →  Discrete(8)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

NUM_NODES = 8


def make_observation_space(num_edges: int) -> spaces.Dict:
    return spaces.Dict({
        # node hiện tại và đích
        "current_node": spaces.Discrete(NUM_NODES),
        "dst_node":     spaces.Discrete(NUM_NODES),
        # trạng thái toàn bộ link
        "link_states":  spaces.Box(
            low=0.0, high=2.0,           # queuing delay có thể > 1
            shape=(num_edges, 4),
            dtype=np.float32,
        ),
    })


def make_action_space() -> spaces.Discrete:
    """Next-hop node (0 .. NUM_NODES-1)."""
    return spaces.Discrete(NUM_NODES)


def obs_to_flat(obs: dict) -> np.ndarray:
    """
    Flatten observation thành 1D vector cho Q-learning.
    Shape: (2 + num_edges * 4,)
    """
    return np.concatenate([
        np.array([obs["current_node"] / (NUM_NODES - 1),
                  obs["dst_node"]     / (NUM_NODES - 1)], dtype=np.float32),
        obs["link_states"].flatten(),
    ])