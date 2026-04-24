"""
env/DQN/spaces.py

Observation space và action space cho môi trường định tuyến.

Observation:
  - current_node : int (0-7)
  - dst_node     : int (0-7)
  - visited_mask : List[int] — các node đã đi qua trong episode
  - link_states  : (num_edges, 6) — delay_norm, bw_norm, queue_size_cur_norm,
                                    utilization, queue_util, drop_norm

Action: Discrete(8) — chọn next-hop node

obs_to_flat() → shape (180,) = 8 + 8 + 8 + 26×6
  - 8  : one-hot current_node   (node hiện tại)
  - 8  : one-hot dst_node       (node đích)
  - 8  : visited_mask           (1.0 nếu đã đi qua, 0.0 nếu chưa)
  - 156: link_states flattened  (26 edges × 6 features)

Lý do đổi từ scalar norm → one-hot + visited_mask:
  - Scalar norm (node/7) tạo ra quan hệ thứ tự giả: node 6 "gần" node 7
    hơn node 0, nhưng topology không phản ánh điều đó.
  - One-hot: mỗi node là một chiều độc lập, model không bị nhầm lẫn
    giữa node index và khoảng cách trong topology.
  - visited_mask: agent biết node nào đã đi qua → tránh vòng lặp
    chủ động, kết hợp với phạt loop trong reward.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

NUM_NODES = 8


def make_observation_space(num_edges: int) -> spaces.Dict:
    return spaces.Dict({
        "current_node": spaces.Discrete(NUM_NODES),
        "dst_node":     spaces.Discrete(NUM_NODES),
        "visited_mask": spaces.Box(
            low=0.0, high=1.0,
            shape=(NUM_NODES,),
            dtype=np.float32,
        ),
        "link_states":  spaces.Box(
            low=0.0, high=2.0,
            shape=(num_edges, 6),
            dtype=np.float32,
        ),
    })


def make_action_space() -> spaces.Discrete:
    return spaces.Discrete(NUM_NODES)


def obs_to_flat(obs: dict) -> np.ndarray:
    """
    Flatten observation thành 1D vector.

    Shape: (8 + 8 + 8 + num_edges × 6,) → 180 với 26 edges.

    Layout:
      [0:8]   — one-hot current_node
      [8:16]  — one-hot dst_node
      [16:24] — visited_mask (1.0 = đã đi qua)
      [24:]   — link_states flattened
    """
    current_one_hot = np.zeros(NUM_NODES, dtype=np.float32)
    current_one_hot[obs["current_node"]] = 1.0

    dst_one_hot = np.zeros(NUM_NODES, dtype=np.float32)
    dst_one_hot[obs["dst_node"]] = 1.0

    # visited_mask: backward-compat nếu obs cũ không có key này
    visited = obs.get("visited_mask", np.zeros(NUM_NODES, dtype=np.float32))
    visited = np.asarray(visited, dtype=np.float32)

    return np.concatenate([
        current_one_hot,              # 8
        dst_one_hot,                  # 8
        visited,                      # 8
        obs["link_states"].flatten(), # 26×6 = 156
    ])