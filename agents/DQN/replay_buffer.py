"""
agents/DQN/replay_buffer.py

Bộ nhớ trải nghiệm (Experience Replay Buffer) cho DQN.
"""

from collections import deque
import random
import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Bộ đệm vòng lưu trữ các transition và cho phép lấy mẫu ngẫu nhiên.

    Tham số:
        capacity (int): Số lượng transition tối đa có thể lưu trữ.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Thêm transition vào buffer theo cơ chế FIFO:
        - Nếu buffer đã đầy (len == capacity) → bỏ phần tử đầu tiên (cũ nhất).
        - Sau đó push phần tử mới vào cuối.

        Tham số:
            state: Vector trạng thái đã làm phẳng.
            action: Hành động đã thực hiện.
            reward: Phần thưởng nhận được.
            next_state: Vector trạng thái kế tiếp.
            done: Cờ kết thúc episode.
        """

        # Trước khi push: nếu đầy thì bỏ phần tử đầu tiên
        if len(self.buffer) == self.capacity:
            self.buffer.popleft()
        
        # Push transition mới vào cuối
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                np.ndarray]:
        """
        Lấy ngẫu nhiên một batch transition.

        Tham số:
            batch_size (int): Số lượng transition cần lấy.

        Trả về:
            Tuple các mảng numpy: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Số lượng transition hiện có trong bộ đệm."""
        return len(self.buffer)
    
    @property
    def is_full(self) -> bool:
        return len(self.buffer) == self.capacity

    @property
    def fill_ratio(self) -> float:
        return len(self.buffer) / self.capacity