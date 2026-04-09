"""
agents/base_agent.py

Lớp cơ sở trừu tượng cho tất cả các agent (Q‑learning, DQN, ...).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng cho các agent học tăng cường.

    Attributes:
        n_states (int): Kích thước không gian trạng thái (chỉ mang tính tượng trưng với DQN).
        n_actions (int): Số lượng hành động có thể chọn.
        config (Dict): Từ điển chứa các tham số cấu hình.
        training (bool): Cờ báo agent đang ở chế độ huấn luyện hay đánh giá.
    """

    def __init__(self, n_states: int, n_actions: int, config: Dict[str, Any]):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.training = True

    @abstractmethod
    def select_action(self, state: Any) -> int:
        """Chọn một hành động dựa trên trạng thái hiện tại."""
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        Thực hiện một bước học (ví dụ: lấy mẫu từ replay buffer, cập nhật mạng Q).
        Trả về dictionary chứa các chỉ số huấn luyện (ví dụ: loss).
        """
        ...

    def train_mode(self):
        """Chuyển agent sang chế độ huấn luyện (có exploration)."""
        self.training = True

    def eval_mode(self):
        """Chuyển agent sang chế độ đánh giá (không exploration)."""
        self.training = False

    def save(self, path: str):
        """Lưu tham số mô hình."""
        pass

    def load(self, path: str):
        """Nạp tham số mô hình."""
        pass