"""
env/DQN/spaces.py

Observation space và action space cho môi trường định tuyến.

Observation:
  - current_node : int (0-7)
  - dst_node     : int (0-7)
  - link_states  : (num_edges, 6) — delay_norm, bw_norm, queue_size_cur_norm,
                                    utilization, queue_util, drop_norm

Action: Discrete(8) — chọn next-hop node

obs_to_flat() → shape (158,) = 2 + 26×6 — input cho DQN.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

NUM_NODES = 8


def make_observation_space(num_edges: int) -> spaces.Dict:
    return spaces.Dict({
        "current_node": spaces.Discrete(NUM_NODES),
        "dst_node":     spaces.Discrete(NUM_NODES),
        "link_states":  spaces.Box(
            low=0.0, high=2.0,
            shape=(num_edges, 6),   # 6 features: thêm drop_norm
            dtype=np.float32,
        ),
    })


def make_action_space() -> spaces.Discrete:
    return spaces.Discrete(NUM_NODES)


def obs_to_flat(obs: dict) -> np.ndarray:
    """
    Flatten observation thành 1D vector.
    Shape: (2 + num_edges × 6,)  →  158 với 26 edges.
    """
    return np.concatenate([
        np.array([
            obs["current_node"] / (NUM_NODES - 1),
            obs["dst_node"]     / (NUM_NODES - 1),
        ], dtype=np.float32),
        obs["link_states"].flatten(),
    ])