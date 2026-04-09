"""
network/DQN/link.py

Dataclass mô tả trạng thái một link — được dùng bởi NetworkTopology.
File này giữ logic tính toán độc lập để dễ test.
"""

from dataclasses import dataclass


@dataclass
class Link:
    src: int
    dst: int
    delay: float        # ms
    bandwidth: float    # Mbps
    queue_size: int     # packets
    load: float = 0.0
    queue_used: int = 0

    @property
    def utilization(self) -> float:
        return min(1.0, self.load / self.bandwidth)

    @property
    def queue_util(self) -> float:
        return min(1.0, self.queue_used / self.queue_size)

    @property
    def is_congested(self) -> bool:
        return self.utilization > 0.8

    @property
    def effective_delay(self) -> float:
        """Delay thực tế = propagation + queuing delay (M/D/1 approximation)."""
        rho = self.utilization
        queuing = (rho / (1 - rho + 1e-6)) * (1.0 / self.bandwidth) * 1000 if rho < 1 else 100.0
        return self.delay + queuing

    def reset(self):
        self.load = 0.0
        self.queue_used = 0