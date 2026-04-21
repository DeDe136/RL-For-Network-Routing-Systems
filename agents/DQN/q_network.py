"""
agents/DQN/q_network.py

Mạng nơ-ron ước lượng Q(s, a) cho DQN.

Hai kiến trúc:
  QNetwork       : MLP tiêu chuẩn — Q(s,a) trực tiếp từ đầu ra cuối.
  DuelingQNetwork: Dueling DQN — tách stream V(s) và A(s,a), kết hợp:
                     Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
                   Lợi thế: V(s) học được dù action không được thực hiện,
                   giúp ổn định hơn khi action space lớn hoặc nhiều
                   action có Q tương đương nhau.
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """
    MLP tiêu chuẩn xấp xỉ Q(s, a).

    Input : flat state vector (batch_size, input_dim)
    Output: Q-value cho mỗi action (batch_size, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Network — tách biệt Value stream và Advantage stream.

    Kiến trúc:
      Shared backbone → [hidden layers với ReLU]
                       ├─ Value stream    → V(s): scalar
                       └─ Advantage stream → A(s,a): vector (output_dim,)

    Kết hợp theo Wang et al. (2016):
      Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

    Lý do dùng mean thay vì max:
      - Ổn định hơn về mặt tối ưu hoá (gradient không bị chi phối bởi 1 action).
      - Đảm bảo tính nhận dạng được (identifiability) của V và A.

    Tại sao Dueling tốt hơn DQN thường?
      - V(s) được cập nhật cho MỌI transition, dù action nào được chọn.
      - A(s,a) chỉ cập nhật cho action cụ thể được thực hiện.
      - Đặc biệt hữu ích khi nhiều action có Q tương đương (như routing:
        cả 2 hop đều tốt → V(s) vẫn học, A(s,a) phân biệt được).

    Input : flat state vector (batch_size, input_dim)
    Output: Q-values (batch_size, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Shared backbone
        backbone_layers = []
        prev = input_dim
        for h in hidden_dims[:-1]:          # tất cả lớp trừ lớp cuối
            backbone_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*backbone_layers)

        # Kích thước lớp trước khi tách stream
        last_hidden = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        # Nếu hidden_dims chỉ có 1 phần tử thì backbone rỗng
        # → prev vẫn là input_dim, last_hidden = hidden_dims[0]
        if len(hidden_dims) == 0:
            last_hidden = input_dim
        elif len(hidden_dims) == 1:
            # backbone rỗng, stream tự build từ input
            last_hidden = hidden_dims[0]
            prev = input_dim

        # Value stream: V(s) — scalar
        self.value_stream = nn.Sequential(
            nn.Linear(prev, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1),
        )

        # Advantage stream: A(s,a) — vector
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tính Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)].

        Args:
            x: (batch_size, input_dim)

        Returns:
            Q-values: (batch_size, output_dim)
        """
        shared   = self.backbone(x)              # (B, prev_hidden)
        value    = self.value_stream(shared)     # (B, 1)
        advantage = self.advantage_stream(shared) # (B, output_dim)

        # Q = V + A - mean(A)  — Wang et al. 2016
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q