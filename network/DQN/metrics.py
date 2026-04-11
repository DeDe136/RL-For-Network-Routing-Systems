"""
network/DQN/metrics.py

Tính các chỉ số mạng dựa trên trạng thái link hiện tại.
"""

from typing import List, Dict
import numpy as np
from network.DQN.topology import NetworkTopology


class NetworkMetrics:

    def __init__(self, topo: NetworkTopology):
        self.topo = topo

    def path_delay(self, path: List[int]) -> float:
        """Tổng propagation delay (ms) — chỉ delay vật lý, không tính queuing."""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.topo.link(path[i], path[i + 1]).delay
        return total

    def path_utilization(self, path: List[int]) -> float:
        """Utilization trung bình trên path."""
        if len(path) < 2:
            return 0.0
        utils = [
            self.topo.link(path[i], path[i + 1]).utilization
            for i in range(len(path) - 1)
        ]
        return float(np.mean(utils))

    def path_queue_util(self, path: List[int]) -> float:
        """Queue utilization trung bình trên path."""
        if len(path) < 2:
            return 0.0
        qutils = [
            self.topo.link(path[i], path[i + 1]).queue_util
            for i in range(len(path) - 1)
        ]
        return float(np.mean(qutils))

    def path_drop_prob(self, path: List[int]) -> float:
        """
        Xác suất bị drop trên path.
        Drop xảy ra khi queue_used_cur > queue_size_cur.
        Ở đây dùng queue_util như proxy: nếu queue_util = 1.0 → drop.
        """
        for i in range(len(path) - 1):
            attr = self.topo.link(path[i], path[i + 1])
            if attr.queue_util >= 1.0:
                return 1.0
        return 0.0

    def global_summary(self) -> Dict:
        return self.topo.summary()