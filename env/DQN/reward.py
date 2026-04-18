"""
env/reward.py

Hàm reward cho bài toán định tuyến mạng.

Thứ tự ưu tiên (trọng số giảm dần):
  delay           > avg_queue_util  > utilization  > hops

  delay          : 0.35  — ưu tiên cao nhất (QoS quan trọng nhất)
  avg_queue_util : 0.25  — tránh đầy hàng đợi → drop
  utilization    : 0.20  — tránh nghẽn băng thông
  hops           : 0.10  — ít hop hơn tốt hơn

Tie-breaker (khi util/queue_util bằng nhau):
  bandwidth_bonus   : +0.05  — link bw lớn → còn nhiều capacity
  queue_size_bonus  : +0.05  — queue lớn → ít bị drop hơn

Tổng hệ số phạt : 0.35 + 0.25 + 0.20 + 0.10 = 0.90
Tổng bonus tối đa: 0.05 + 0.05 = 0.10
→ reward ∈ [-2.0, +1.10] thực tế ≈ [-2.0, +1.0]

Shaping reward (intermediate hop) — cùng thứ tự ưu tiên, hệ số nhỏ hơn:
  delay          : 0.08
  avg_queue_util : 0.07
  utilization    : 0.05
  hops penalty   : 0.025 (cố định)
  bandwidth bonus: +0.02
  queue_size bonus: +0.02
"""

BANDWIDTH_MAX  = 200.0   # Mbps  — dùng để normalize tie-breaker
QUEUE_SIZE_MAX = 100.0   # packets


def compute_final_reward(
    path_found:     bool,
    total_delay:    float,
    dropped:        bool,
    hops:           int,
    utilization:    float,          # avg utilization băng thông trên path [0,1]
    avg_queue_util: float = 0.0,   # avg queue_util trên path [0,1]
    avg_bandwidth:  float = 0.0,   # avg bandwidth trên path (Mbps)
    avg_queue_size: float = 0.0,   # avg queue_size_cur trên path (packets)
) -> float:
    """
    Reward cuối episode (terminated hoặc truncated).

    Trả về float ∈ [-2.0, +1.0].
    """
    if not path_found:
        return -2.0

    reward = 1.0

    # ── Phạt delay (ưu tiên CAO NHẤT) ───────────────────────────────
    reward -= min(1.0, total_delay / 50.0)  * 0.35

    # ── Phạt congestion (queue > util) ───────────────────────────────
    reward -= avg_queue_util * 0.25
    reward -= utilization    * 0.20

    # ── Phạt hops (thứ yếu) ──────────────────────────────────────────
    reward -= min(1.0, hops / 7.0)          * 0.10

    # ── Tie-breaker: thưởng capacity lớn ─────────────────────────────
    # Khi util/queue_util bằng nhau → ưu tiên link bw cao, queue lớn.
    # Bonus nhỏ (+0.05) không làm đảo lộn thứ tự ưu tiên trên.
    if avg_bandwidth > 0:
        reward += min(1.0, avg_bandwidth  / BANDWIDTH_MAX)  * 0.05
    if avg_queue_size > 0:
        reward += min(1.0, avg_queue_size / QUEUE_SIZE_MAX) * 0.05

    # ── Phòng thủ drop ────────────────────────────────────────────────
    if dropped:
        reward -= 0.2

    return float(reward)


def compute_shaping_reward(
    current_link_delay:       float,
    current_link_utilization: float,
    current_link_queue_util:  float,
    current_link_bandwidth:   float = 0.0,
    current_link_queue_size:  float = 0.0,
) -> float:
    """
    Reward shaping cho bước trung gian (intermediate hop).

    Cùng thứ tự ưu tiên với final reward nhưng hệ số nhỏ hơn,
    tránh lấn át tín hiệu học dài hạn.

    Trả về float ∈ [-0.5, 0.0).
    """
    shaping = 0.0

    # Phạt cố định mỗi hop — khuyến khích path ngắn
    shaping -= 0.025

    # Phạt delay link hiện tại (ưu tiên cao nhất)
    shaping -= min(1.0, current_link_delay / 10.0) * 0.08

    # Phạt congestion (queue > util)
    shaping -= current_link_queue_util  * 0.07
    shaping -= current_link_utilization * 0.05

    # Tie-breaker: thưởng nhỏ cho link capacity cao
    if current_link_bandwidth > 0:
        shaping += min(1.0, current_link_bandwidth  / BANDWIDTH_MAX)  * 0.02
    if current_link_queue_size > 0:
        shaping += min(1.0, current_link_queue_size / QUEUE_SIZE_MAX) * 0.02

    return max(-0.5, float(shaping))