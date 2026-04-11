"""
network/link.py

Dataclass mô tả trạng thái một link theo MDP spec.

Mỗi link có 6 thông số:
  - delay          : propagation delay (ms)
  - bandwidth      : băng thông tối đa (Mbps)
  - queue_size_cur : kích thước hàng đợi của node hiện tại (đầu gửi)
  - queue_used_cur : lượng hàng đợi đang dùng ở node hiện tại
  - queue_used     : lượng dữ liệu đã truyền qua đầu bên kia của link
  - load           : mức tải hiện tại trên link (Mbps)

link_state_vector() expose 5 thông số đã chuẩn hóa:
  [delay_norm, bw_norm, queue_size_cur_norm, utilization, queue_util]
  trong đó:
    utilization = load / bandwidth
    queue_util  = queue_used_cur / queue_size_cur
"""

from dataclasses import dataclass, field


@dataclass
class LinkAttr:
    delay:          float   # ms
    bandwidth:      float   # Mbps
    queue_size_cur: int     # packets — kích thước hàng đợi tại node gửi

    # Trạng thái thay đổi theo traffic
    queue_used_cur: float = 0.0   # hàng đợi đang dùng ở node gửi (packets)
    queue_used:     float = 0.0   # lượng đã truyền sang node nhận (Mbps)
    load:           float = 0.0   # băng thông đang dùng (Mbps)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def utilization(self) -> float:
        """load / bandwidth ∈ [0, 1]."""
        if self.bandwidth <= 0:
            return 0.0
        return min(1.0, self.load / self.bandwidth)

    @property
    def queue_util(self) -> float:
        """queue_used_cur / queue_size_cur ∈ [0, 1]."""
        if self.queue_size_cur <= 0:
            return 0.0
        return min(1.0, self.queue_used_cur / self.queue_size_cur)

    @property
    def is_congested(self) -> bool:
        return self.utilization > 0.8

    @property
    def remain_bw(self) -> float:
        """Băng thông còn trống = bandwidth - load."""
        return max(0.0, self.bandwidth - self.load)

    def reset(self):
        """Xóa trạng thái traffic về 0 (đầu episode)."""
        self.queue_used_cur = 0.0
        self.queue_used     = 0.0
        self.load           = 0.0