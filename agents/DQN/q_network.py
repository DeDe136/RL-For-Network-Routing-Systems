"""
agents/DQN/q_network.py

Định nghĩa mạng nơ‑ron ước lượng Q(s, a) cho DQN.
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """
    Mạng nơ‑ron truyền thẳng (MLP) để xấp xỉ hàm Q.

    Tham số:
        input_dim (int): Chiều của vector trạng thái đã được làm phẳng.
        output_dim (int): Số lượng hành động (số nút mạng).
        hidden_dims (List[int]): Danh sách kích thước các lớp ẩn.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        # Xây dựng các lớp ẩn với hàm kích hoạt ReLU
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        # Lớp đầu ra (không có hàm kích hoạt, trả về giá trị Q thô)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến.

        Tham số:
            x (torch.Tensor): Batch vector trạng thái, shape (batch_size, input_dim).

        Trả về:
            torch.Tensor: Giá trị Q cho từng hành động, shape (batch_size, output_dim).
        """
        return self.net(x)