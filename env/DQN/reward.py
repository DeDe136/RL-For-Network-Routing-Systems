"""
env/DQN/reward.py

Hàm reward cho bài toán định tuyến mạng.

Thứ tự ưu tiên (trọng số giảm dần):
  loop > dropped_data > delay > avg_queue_util > utilization > hops

  loop         : 0.40  — ưu tiên cao nhất, phạt mạnh khi đi lại node đã thăm
  dropped_data   : 0.30  — ưu tiên cao nhất, tránh data bị drop
  delay          : 0.22  
  avg_queue_util : 0.13  — tránh đầy hàng đợi
  utilization    : 0.12  — tránh nghẽn băng thông
  hops           : 0.08  — ít hop hơn tốt hơn

Phạt loop (compute_final_reward):
  loop_count       : số node bị đi lại trong toàn episode
  LOOP_PENALTY_MAX : 0.40 — mỗi lần loop phạt 0.40/MAX_LOOPS,
                    tối đa phạt 0.40 khi loop MAX_LOOPS lần.

Phạt loop (compute_shaping_reward):
  is_loop          : bool — True nếu hop này đi lại node đã thăm
  LOOP_SHAPING_PENALTY : 0.30 — phạt cố định mỗi lần loop ở intermediate hop.

Phạt dropped_data (lượng data bị drop tích lũy trên path):
  dropped_data_penalty : 0.30  — phạt theo tổng data bị drop
  Cách tính: penalty = min(1.0, total_dropped / DROP_PENALTY_SCALE) × 0.30

Tie-breaker:
  bandwidth_bonus  : +0.05  — link bw lớn → còn nhiều capacity
  queue_size_bonus : +0.05  — queue lớn → ít bị drop hơn

Tổng hệ số phạt final:
  0.40 + 0.30 + 0.22 + 0.13 + 0.12 + 0.08 = 1.25
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
DROP_PENALTY_SCALE = 100.0    # packets — 100 packets drop → phạt tối đa

# Traffic penalty (sau send_traffic) — phạt nhẹ load và queue_used_cur raw
LOAD_PENALTY_W       = 0.04   # trọng số phạt load / bandwidth
QUEUE_USED_PENALTY_W = 0.03   # trọng số phạt queue_used_cur / queue_size_cur

# Loop penalty
MAX_LOOPS            = 4       # số lần loop tối đa để normalize (≥ MAX_LOOPS → phạt max)
LOOP_PENALTY_MAX     = 0.40    # phạt tối đa cho loop trong final reward
LOOP_SHAPING_PENALTY = 0.30    # phạt cố định mỗi lần loop ở shaping reward


def compute_final_reward(
    path_found:         bool,
    total_delay:        float,
    hops:               int,
    utilization:        float,
    avg_queue_util:     float = 0.0,
    avg_bandwidth:      float = 0.0,
    avg_queue_size:     float = 0.0,
    total_dropped_data: float = 0.0,
    loop_count:         int   = 0,    # số node bị đi lại trong episode
) -> float:
    """
    Reward cuối episode (terminated hoặc truncated).

    loop_count: số lần agent đi vào node đã thăm trong episode này.
    Phạt tuyến tính: loop_count / MAX_LOOPS × LOOP_PENALTY_MAX,
    clamp tại LOOP_PENALTY_MAX khi loop_count >= MAX_LOOPS.

    Trả về float ∈ [-2.0, +1.10].
    """
    if not path_found:
        return -2.0

    reward = 1.0

    # ── Phạt loop — ưu tiên cao nhất ─────────────────────────────────
    # Mỗi lần loop bị phạt tuyến tính, tối đa LOOP_PENALTY_MAX.
    reward -= min(1.0, loop_count / MAX_LOOPS) * LOOP_PENALTY_MAX

    # ── Phạt avg_queue_util và utilization (congestion) ──────────────
    reward -= avg_queue_util * 0.13
    reward -= utilization    * 0.12

    # ── Phạt hops và delay ───────────────────────────────────────────
    reward -= min(1.0, hops / 7.0)          * 0.08
    reward -= min(1.0, total_delay / 50.0)  * 0.22

    # ── Phạt dropped_data ─────────────────────────────────────────────
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
    current_link_dropped:     float = 0.0,
    is_loop:                  bool  = False,  # True nếu hop này đi lại node đã thăm
) -> float:
    """
    Reward shaping cho bước trung gian (intermediate hop).

    is_loop=True → cộng thêm phạt LOOP_SHAPING_PENALTY vào shaping.
    Phạt loop ở đây lớn hơn các hạng mục khác để signal rõ ràng:
    agent cần tránh loop ngay tại bước xảy ra, không chờ đến cuối episode.

    Trả về float (không clamp cứng để loop penalty vẫn hiển thị rõ).
    """
    shaping = 0.0

    # ── Phạt loop ngay tại bước này ───────────────────────────────────
    if is_loop:
        shaping -= LOOP_SHAPING_PENALTY

    # ── Phạt cố định mỗi hop — khuyến khích path ngắn ────────────────
    shaping -= 0.025

    # ── Phạt congestion ───────────────────────────────────────────────
    shaping -= current_link_queue_util  * 0.04
    shaping -= current_link_utilization * 0.03

    # ── Phạt delay link ───────────────────────────────────────────────
    shaping -= min(1.0, current_link_delay / 10.0) * 0.07

    # ── Phạt dropped_data của link này ───────────────────────────────
    shaping -= min(1.0, current_link_dropped / 50.0) * 0.10

    # ── Tie-breaker: thưởng capacity cao ─────────────────────────────
    if current_link_bandwidth > 0:
        shaping += min(1.0, current_link_bandwidth  / BANDWIDTH_MAX)  * 0.02
    if current_link_queue_size > 0:
        shaping += min(1.0, current_link_queue_size / QUEUE_SIZE_MAX) * 0.02

    return float(shaping)

def compute_traffic_penalty(
    load:           float,
    queue_used_cur: float,
    bandwidth:      float,
    queue_size_cur: float,
) -> float:
    """
    Phạt nhẹ dựa trên load và queue_used_cur raw trả về từ send_traffic.

    Gọi ngay sau send_traffic trong step() ở mọi hop (kể cả intermediate
    lẫn terminal) trước khi cộng vào shaping / final reward, để agent nhận
    tín hiệu tức thời về mức độ tải link vừa đi qua.

    Normalize:
      load_ratio       = load / bandwidth          ∈ [0, 1]
      queue_used_ratio = queue_used_cur / queue_size_cur  ∈ [0, 1]

    Penalty = -(load_ratio × LOAD_PENALTY_W + queue_used_ratio × QUEUE_USED_PENALTY_W)
            ∈ [-0.07, 0.0]

    Trọng số nhỏ (~0.03 mỗi thành phần) để không lấn át shaping reward
    hiện có, chỉ đủ để phân biệt link nhàn rỗi vs link đang tải nặng.
    """
    load_ratio       = min(1.0, load / bandwidth)          if bandwidth      > 0 else 0.0
    queue_used_ratio = min(1.0, queue_used_cur / queue_size_cur) if queue_size_cur > 0 else 0.0

    return -(load_ratio * LOAD_PENALTY_W + queue_used_ratio * QUEUE_USED_PENALTY_W)