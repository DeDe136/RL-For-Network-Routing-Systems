"""
network/DQN/topology.py

Topology mạng 8 node cố định với tham số link thay đổi động.

Mỗi link có 5 thông số được expose trong link_state_vector():
  - delay_norm      : delay chuẩn hóa (ms / 10)
  - bw_norm         : bandwidth chuẩn hóa (Mbps / 200)
  - queue_size_norm : kích thước queue chuẩn hóa (packets / 100)
  - utilization     : tỉ lệ băng thông đang dùng [0,1]
  - queue_util      : tỉ lệ queue đang dùng [0,1]

Hai thông số cuối thay đổi liên tục theo traffic — DQN phải học
policy phụ thuộc vào chúng thay vì chỉ nhớ cặp (src, dst).

randomize_links(): sinh lại delay/bw/queue_size ngẫu nhiên mỗi episode.
step_background(): thêm load ngẫu nhiên để mô phỏng traffic nền.
decay_load()     : giảm dần load (leaky bucket), tránh tích lũy vô hạn.
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class LinkAttr:
    delay: float        # ms
    bandwidth: float    # Mbps
    queue_size: int     # packets
    queue_used: int = 0
    load: float = 0.0

    @property
    def utilization(self) -> float:
        return min(1.0, self.load / self.bandwidth) if self.bandwidth > 0 else 0.0

    @property
    def queue_util(self) -> float:
        return min(1.0, self.queue_used / self.queue_size) if self.queue_size > 0 else 0.0

    @property
    def is_congested(self) -> bool:
        return self.utilization > 0.8

    @property
    def drop_prob(self) -> float:
        if self.queue_util < 0.8:
            return 0.0
        return min(1.0, (self.queue_util - 0.8) / 0.2)

    def reset(self):
        self.queue_used = 0
        self.load = 0.0


class NetworkTopology:
    """
    Đồ thị mạng 8 node.

        0 --- 1 --- 3 --- 5
        |  \\  |     |  /  |
        2 --- 4 --- 6 --- 7
    """

    NUM_NODES = 8

    # Danh sách cạnh vô hướng (mỗi cặp được tạo 2 chiều)
    EDGE_LIST: List[Tuple[int, int]] = [
        (0, 1), (0, 2), (1, 2), (1, 3), (1, 4),
        (2, 4), (3, 5), (3, 4), (4, 5), (4, 6),
        (5, 7), (6, 7), (2, 6),
    ]

    # Khoảng ngẫu nhiên cho các tham số link
    DELAY_RANGE     = (1.0,  10.0)   # ms
    BANDWIDTH_RANGE = (20.0, 200.0)  # Mbps
    QUEUE_SIZE_RANGE = (20,  100)    # packets

    def __init__(self, seed: int = None,
                 rng: Optional[np.random.Generator] = None):
        self.graph = nx.DiGraph()
        self._link_attrs: Dict[Tuple[int, int], LinkAttr] = {}
        # Ưu tiên rng nếu truyền vào, ngược lại dùng seed
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self._build_structure()
        self.randomize_links()

    # ------------------------------------------------------------------ #
    #  Xây dựng cấu trúc                                                  #
    # ------------------------------------------------------------------ #

    def _build_structure(self):
        """Tạo graph với node và edge, chưa gán tham số link."""
        self.graph.add_nodes_from(range(self.NUM_NODES))
        for src, dst in self.EDGE_LIST:
            for u, v in [(src, dst), (dst, src)]:
                self.graph.add_edge(u, v)

    def randomize_links(self):
        """
        Sinh ngẫu nhiên delay, bandwidth, queue_size cho mọi link.
        Gọi mỗi episode để tạo topology mới — cùng cấu trúc kết nối
        nhưng tham số link khác nhau buộc DQN phải đọc link_states.
        """
        self._link_attrs.clear()
        for u, v in self.EDGE_LIST:
            delay  = float(self._rng.uniform(*self.DELAY_RANGE))
            bw     = float(self._rng.uniform(*self.BANDWIDTH_RANGE))
            qsize  = int(self._rng.integers(*self.QUEUE_SIZE_RANGE))
            # Hai chiều chia sẻ cùng tham số vật lý
            for a, b in [(u, v), (v, u)]:
                self._link_attrs[(a, b)] = LinkAttr(
                    delay=delay, bandwidth=bw, queue_size=qsize
                )

    # ------------------------------------------------------------------ #
    #  Truy xuất trạng thái                                               #
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
        return nx.to_numpy_array(self.graph, weight=None).astype(np.float32)

    def link_state_vector(self) -> np.ndarray:
        """
        Shape (num_edges, 5):
        [delay_norm, bw_norm, queue_size_norm, utilization, queue_util]

        Cột 0-2: tham số vật lý, thay đổi mỗi episode (randomize_links).
        Cột 3-4: trạng thái lưu lượng, thay đổi mỗi step.
        """
        rows = []
        for (u, v), attr in sorted(self._link_attrs.items()):
            rows.append([
                attr.delay      / 10.0,   # delay_norm     (max ~10ms)
                attr.bandwidth  / 200.0,  # bw_norm        (max 200Mbps)
                attr.queue_size / 100.0,  # queue_size_norm (max 100 pkts)
                attr.utilization,
                attr.queue_util,
            ])
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Cập nhật traffic                                                   #
    # ------------------------------------------------------------------ #

    def send_traffic(self, path: List[int], volume_mbps: float) -> Dict:
        """Gửi traffic của agent qua path, cập nhật load và queue."""
        if len(path) < 2:
            return {"total_delay": 0.0, "dropped": False, "hops": 0}
        total_delay = 0.0
        dropped = False
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            attr = self._link_attrs[(u, v)]
            total_delay += attr.delay
            attr.load = min(attr.bandwidth, attr.load + volume_mbps)
            pkts = int(volume_mbps)
            if attr.queue_used + pkts <= attr.queue_size:
                attr.queue_used += pkts
            else:
                dropped = True
        return {"total_delay": total_delay, "dropped": dropped,
                "hops": len(path) - 1}

    def step_background(self, intensity: float = 0.3):
        """
        Thêm background traffic ngẫu nhiên lên 30-60% link.
        Gọi sau reset() để tạo trạng thái khởi đầu đa dạng.
        """
        all_links = list(self._link_attrs.keys())
        n = int(self._rng.integers(len(all_links) // 3,
                                   max(len(all_links) // 3 + 1,
                                       len(all_links) * 2 // 3)))
        for idx in self._rng.choice(len(all_links), size=n, replace=False):
            u, v = all_links[idx]
            attr = self._link_attrs[(u, v)]
            extra = float(self._rng.uniform(0, intensity * attr.bandwidth))
            attr.load = min(attr.bandwidth, attr.load + extra)
            attr.queue_used = min(attr.queue_size,
                                  attr.queue_used + int(extra))

    def decay_load(self, decay: float = 0.85):
        """Giảm dần load/queue sau mỗi step (leaky bucket)."""
        for attr in self._link_attrs.values():
            attr.load       = attr.load * decay
            attr.queue_used = max(0, int(attr.queue_used * decay))

    def reset(self, rng: Optional[np.random.Generator] = None):
        """Reset: sinh lại tham số link ngẫu nhiên và xóa trạng thái lưu lượng."""
        if rng is not None:
            self._rng = rng
        self.randomize_links()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def shortest_path(self, src: int, dst: int) -> List[int]:
        try:
            return nx.shortest_path(self.graph, src, dst, weight="delay")
        except nx.NetworkXNoPath:
            return []

    def summary(self) -> Dict:
        utils  = [a.utilization for a in self._link_attrs.values()]
        queues = [a.queue_util  for a in self._link_attrs.values()]
        return {
            "avg_utilization": float(np.mean(utils))  if utils else 0.0,
            "max_utilization": float(np.max(utils))   if utils else 0.0,
            "avg_queue_util":  float(np.mean(queues)) if queues else 0.0,
            "congested_links": sum(1 for a in self._link_attrs.values()
                                   if a.is_congested),
        }