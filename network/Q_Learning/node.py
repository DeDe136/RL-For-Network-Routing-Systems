"""
network/Q_Learning/node.py

Router node với routing table và queue đơn giản.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Node:
    node_id: int
    queue_capacity: int = 100    # packets

    _routing_table: Dict[int, int] = field(default_factory=dict, repr=False)
    _queue: int = 0

    @property
    def queue_util(self) -> float:
        return min(1.0, self._queue / self.queue_capacity)

    def set_route(self, dst: int, next_hop: int):
        self._routing_table[dst] = next_hop

    def get_next_hop(self, dst: int) -> int:
        """Trả về -1 nếu không có route."""
        return self._routing_table.get(dst, -1)

    def enqueue(self, packets: int) -> bool:
        """Thêm packets vào queue. Trả về False nếu drop."""
        if self._queue + packets <= self.queue_capacity:
            self._queue += packets
            return True
        return False  # drop

    def reset(self):
        self._routing_table.clear()
        self._queue = 0