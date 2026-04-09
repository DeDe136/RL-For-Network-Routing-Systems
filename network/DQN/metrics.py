"""
network/DQN/metrics.py

Tính throughput, latency, drop rate cho một path.
"""

from typing import List, Dict
import numpy as np
from network.Q_Learning.topology import NetworkTopology


class NetworkMetrics:

    def __init__(self, topo: NetworkTopology):
        self.topo = topo

    def path_delay(self, path: List[int]) -> float:
        """Tổng propagation delay trên path (ms)."""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.topo.link(path[i], path[i+1]).delay
        return total

    def path_effective_delay(self, path: List[int]) -> float:
        """Delay thực tế bao gồm queuing (ms)."""
        from network.Q_Learning.link import Link as LinkDC
        total = 0.0
        for i in range(len(path) - 1):
            attr = self.topo.link(path[i], path[i+1])
            rho = attr.utilization
            queuing = (rho / (1 - rho + 1e-6)) * 10.0 if rho < 1 else 100.0
            total += attr.delay + queuing
        return total

    def path_drop_prob(self, path: List[int]) -> float:
        """Xác suất drop ít nhất 1 gói trên path."""
        survive = 1.0
        for i in range(len(path) - 1):
            attr = self.topo.link(path[i], path[i+1])
            survive *= (1.0 - attr.drop_prob)
        return 1.0 - survive

    def path_throughput(self, path: List[int], volume_mbps: float) -> float:
        """Throughput thực tế = volume × (1 - drop_prob)."""
        return volume_mbps * (1.0 - self.path_drop_prob(path))

    def global_summary(self) -> Dict:
        return self.topo.summary()