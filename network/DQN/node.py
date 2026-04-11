"""
network/DQN/node.py

Node (router) với routing table và hàng đợi.
Queue của node được mô hình hoá qua queue_used_cur của link đi ra.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Node:
    node_id:        int
    queue_capacity: int = 100   # packets — dùng làm tham chiếu mặc định

    _routing_table: Dict[int, int] = field(default_factory=dict, repr=False)

    # ── Routing table ──────────────────────────────────────────────────

    def set_route(self, dst: int, next_hop: int):
        self._routing_table[dst] = next_hop

    def get_next_hop(self, dst: int) -> int:
        """Trả về -1 nếu không có route."""
        return self._routing_table.get(dst, -1)

    def reset(self):
        self._routing_table.clear()