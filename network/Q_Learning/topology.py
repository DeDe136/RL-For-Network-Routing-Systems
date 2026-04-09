"""
network/Q_Learning/ql_topology.py

Topology mạng 8 node cố định.
Mỗi link có 3 thông số:
  - delay      : ms          (độ trễ truyền dẫn)
  - bandwidth  : Mbps        (thông lượng tối đa)
  - queue_size : packets     (kích thước hàng đợi tại node đích)

Utilization được cập nhật khi traffic đi qua link.
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class LinkAttr:
    delay: float        # ms
    bandwidth: float    # Mbps
    queue_size: int     # packets
    queue_used: int = 0 # packets đang trong hàng đợi
    load: float = 0.0   # Mbps hiện tại đang dùng

    @property
    def utilization(self) -> float:
        """Tỉ lệ sử dụng băng thông [0, 1]."""
        return min(1.0, self.load / self.bandwidth)

    @property
    def queue_util(self) -> float:
        """Tỉ lệ đầy hàng đợi [0, 1]."""
        return min(1.0, self.queue_used / self.queue_size)

    @property
    def is_congested(self) -> bool:
        return self.utilization > 0.8

    @property
    def drop_prob(self) -> float:
        """Xác suất drop gói khi hàng đợi gần đầy (tail-drop model)."""
        if self.queue_util < 0.8:
            return 0.0
        return min(1.0, (self.queue_util - 0.8) / 0.2)

    def reset(self):
        self.queue_used = 0
        self.load = 0.0


class NetworkTopology:
    """
    Đồ thị mạng 8 node dạng mesh bán đầy đủ.

    Node layout (để dễ hình dung):

        0 --- 1 --- 3 --- 5
        |  \\  |     |  /  |
        2 --- 4 --- 6 --- 7

    Tất cả link đều hai chiều.
    """

    NUM_NODES = 8

    # (src, dst, delay_ms, bandwidth_mbps, queue_packets)
    EDGES: List[Tuple] = [
        (0, 1, 2,  100, 50),
        (0, 2, 3,  100, 50),
        (1, 2, 1,   50, 30),
        (1, 3, 4,  100, 50),
        (1, 4, 5,   50, 30),
        (2, 4, 2,  100, 50),
        (3, 5, 3,  100, 50),
        (3, 4, 2,   50, 30),  # cross-link
        (4, 5, 3,  100, 50),
        (4, 6, 2,  100, 50),
        (5, 7, 4,  100, 50),
        (6, 7, 2,   50, 30),
        (2, 6, 6,  100, 50),  # long-haul
    ]

    def __init__(self):
        self.graph = nx.DiGraph()
        self._link_attrs: Dict[Tuple[int,int], LinkAttr] = {}
        self._build()

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def _build(self):
        self.graph.add_nodes_from(range(self.NUM_NODES))
        for src, dst, delay, bw, q in self.EDGES:
            for u, v in [(src, dst), (dst, src)]:
                attr = LinkAttr(delay=delay, bandwidth=bw, queue_size=q)
                self._link_attrs[(u, v)] = attr
                self.graph.add_edge(u, v,
                    delay=delay, bandwidth=bw, queue_size=q)

    # ------------------------------------------------------------------ #
    #  State access                                                        #
    # ------------------------------------------------------------------ #

    def link(self, u: int, v: int) -> LinkAttr:
        return self._link_attrs[(u, v)]

    def neighbors(self, node: int) -> List[int]:
        return list(self.graph.successors(node))

    def has_link(self, u: int, v: int) -> bool:
        return (u, v) in self._link_attrs

    def num_edges(self) -> int:
        return len(self._link_attrs)

    @property
    def adj_matrix(self) -> np.ndarray:
        """Ma trận kề nhị phân (8x8)."""
        return nx.to_numpy_array(self.graph, weight=None).astype(np.float32)

    def link_state_vector(self) -> np.ndarray:
        """
        Vector trạng thái toàn bộ link, shape (num_edges, 4):
        [delay_norm, bw_norm, utilization, queue_util]
        """
        rows = []
        for (u, v), attr in sorted(self._link_attrs.items()):
            rows.append([
                attr.delay / 10.0,            # normalize: max ~10ms
                attr.bandwidth / 100.0,        # normalize: max 100Mbps
                attr.utilization,
                attr.queue_util,
            ])
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Update                                                              #
    # ------------------------------------------------------------------ #

    def send_traffic(self, path: List[int], volume_mbps: float) -> Dict:
        """
        Gửi traffic qua path, cập nhật load và queue.
        Trả về dict metrics: delay, drops, success.
        """
        if len(path) < 2:
            return {"total_delay": 0.0, "dropped": False, "hops": 0}

        total_delay = 0.0
        dropped = False

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            attr = self._link_attrs[(u, v)]

            total_delay += attr.delay

            # Cập nhật load (đơn giản: cộng dồn, reset mỗi episode)
            attr.load = min(attr.bandwidth, attr.load + volume_mbps)

            # Mô hình queue: mỗi Mbps ~ 1 packet đơn giản hoá
            pkts = int(volume_mbps)
            if attr.queue_used + pkts <= attr.queue_size:
                attr.queue_used += pkts
            else:
                dropped = True   # tail-drop

        return {
            "total_delay": total_delay,
            "dropped": dropped,
            "hops": len(path) - 1,
        }

    def reset(self):
        """Reset trạng thái link về 0 (đầu episode)."""
        for attr in self._link_attrs.values():
            attr.reset()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def shortest_path(self, src: int, dst: int) -> List[int]:
        """Dijkstra theo delay (baseline so sánh với RL)."""
        try:
            return nx.shortest_path(self.graph, src, dst, weight="delay")
        except nx.NetworkXNoPath:
            return []

    def summary(self) -> Dict:
        """Thống kê nhanh trạng thái mạng."""
        utils = [a.utilization for a in self._link_attrs.values()]
        queues = [a.queue_util for a in self._link_attrs.values()]
        return {
            "avg_utilization": float(np.mean(utils)),
            "max_utilization": float(np.max(utils)),
            "avg_queue_util":  float(np.mean(queues)),
            "congested_links": sum(1 for a in self._link_attrs.values() if a.is_congested),
        }