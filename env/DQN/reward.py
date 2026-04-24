"""
env/DQN/reward.py

Hàm reward cho bài toán định tuyến mạng.

Thứ tự ưu tiên (trọng số giảm dần):
  dropped_data > delay > avg_queue_util > utilization > hops

  dropped_data   : 0.30  — ưu tiên cao nhất, tránh data bị drop
  delay          : 0.22  
  avg_queue_util : 0.13  — tránh đầy hàng đợi
  utilization    : 0.12  — tránh nghẽn băng thông
  hops           : 0.08  — ít hop hơn tốt hơn

Phạt dropped_data (lượng data bị drop tích lũy trên path):
  dropped_data_penalty : 0.30  — phạt theo tổng data bị drop
  Cách tính: penalty = min(1.0, total_dropped / DROP_PENALTY_SCALE) × 0.30

Tie-breaker:
  bandwidth_bonus  : +0.05  — link bw lớn → còn nhiều capacity
  queue_size_bonus : +0.05  — queue lớn → ít bị drop hơn

Tổng hệ số phạt:
  0.30 + 0.22 + 0.13 + 0.12 + 0.08 = 0.85
Tổng bonus tối đa:
  0.05 + 0.05 = 0.10
→ reward ∈ [-2.0, +1.10]  thực tế ≈ [-2.0, +1.0]

Điều kiện kết thúc episode:
  - terminated: current_node == dst (đến đích)
  - truncated : hops >= max_hops - 1 (hết bước)
  Không còn truncated vì drop — drop được phạt qua reward thay thế.

Shaping reward (intermediate hop) — cùng thứ tự, hệ số nhỏ hơn.
"""

BANDWIDTH_MAX      = 200.0   # Mbps
QUEUE_SIZE_MAX     = 100.0   # packets
DROP_PENALTY_SCALE = 50.0    # packets — 50 packets drop → phạt tối đa


def compute_final_reward(
    path_found:        bool,
    total_delay:       float,
    hops:              int,
    utilization:       float,
    avg_queue_util:    float = 0.0,
    avg_bandwidth:     float = 0.0,
    avg_queue_size:    float = 0.0,
    total_dropped_data:float = 0.0,   # tổng data bị drop trên toàn path
) -> float:
    """
    Reward cuối episode (terminated hoặc truncated).

    Trả về float ∈ [-2.0, +1.0].
    Không nhận tham số 'dropped' bool nữa — drop được phạt qua
    total_dropped_data.
    """
    if not path_found:
        return -2.0

    reward = 1.0

    # ── Phạt avg_queue_util và utilization (congestion) ─────────────
    reward -= avg_queue_util * 0.13
    reward -= utilization    * 0.12

    # ── Phạt hops và delay ───────────────────────────────────────────
    reward -= min(1.0, hops / 7.0)          * 0.08
    reward -= min(1.0, total_delay / 50.0)  * 0.22

    # ── Phạt dropped_data — ưu tiên cao nhất ─────────────────────────
    # Phạt liên tục theo lượng data bị drop, không cắt episode đột ngột.
    # DROP_PENALTY_SCALE: 50 packets drop → phạt tối đa 0.30 điểm.
    reward -= min(1.0, total_dropped_data / DROP_PENALTY_SCALE) * 0.30

    # ── Tie-breaker: thưởng capacity lớn ─────────────────────────────
    if avg_bandwidth > 0:
        reward += min(1.0, avg_bandwidth  / BANDWIDTH_MAX)  * 0.05
    if avg_queue_size > 0:
        reward += min(1.0, avg_queue_size / QUEUE_SIZE_MAX) * 0.05

    return float(reward)


def compute_shaping_reward(
    current_link_delay:       float,
    current_link_utilization: float,
    current_link_queue_util:  float,
    current_link_bandwidth:   float = 0.0,
    current_link_queue_size:  float = 0.0,
    current_link_dropped:     float = 0.0,   # dropped_data của link vừa đi qua
) -> float:
    """
    Reward shaping cho bước trung gian (intermediate hop).

    Cùng thứ tự ưu tiên với final reward nhưng hệ số nhỏ hơn.
    Trả về float ∈ [-0.5, 0.0).
    """
    shaping = 0.0

    # Phạt cố định mỗi hop — khuyến khích path ngắn
    shaping -= 0.025

    # Phạt congestion (queue > util)
    shaping -= current_link_queue_util  * 0.04
    shaping -= current_link_utilization * 0.03

    # Phạt delay link
    shaping -= min(1.0, current_link_delay / 10.0) * 0.07

    # Phạt dropped_data của link này
    shaping -= min(1.0, current_link_dropped / 50.0) * 0.10

    # Tie-breaker: thưởng capacity cao
    if current_link_bandwidth > 0:
        shaping += min(1.0, current_link_bandwidth  / BANDWIDTH_MAX)  * 0.02
    if current_link_queue_size > 0:
        shaping += min(1.0, current_link_queue_size / QUEUE_SIZE_MAX) * 0.02

    return max(-0.5, float(shaping))