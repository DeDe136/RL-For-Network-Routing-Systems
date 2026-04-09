"""
env/DQN/reward.py

Hàm reward cho bài toán định tuyến.
Mục tiêu: tối thiểu delay + drop, tối đa throughput.
"""


def compute_final_reward(
    path_found: bool,
    total_delay: float,
    dropped: bool,
    hops: int,
    utilization: float,
    avg_queue_util: float = 0.0,
) -> float:
    """
    Reward cuối episode (khi terminated hoặc truncated).

    Args:
        path_found     : agent đến đích thành công (không bị drop)
        total_delay    : tổng delay trên toàn path (ms)
        dropped        : có gói bị drop không
        hops           : số hop đã đi
        utilization    : utilization băng thông trung bình trên path [0,1]
        avg_queue_util : tỉ lệ đầy hàng đợi trung bình trên path [0,1]

    Returns:
        reward float trong [-2, +1]
    """
    if not path_found:
        return -2.0             # không đến được đích — phạt nặng

    reward = 1.0

    # Phạt delay (50ms coi là xấu)
    reward -= min(1.0, total_delay / 50.0) * 0.4

    # Phạt hop count (tối đa 7 hops)
    reward -= min(1.0, hops / 7.0) * 0.2

    # Phạt utilization băng thông trung bình
    reward -= utilization * 0.1

    # Phạt queue_util trung bình — báo hiệu mức độ gần tắc nghẽn
    reward -= avg_queue_util * 0.1

    # dropped không thể xảy ra khi path_found=True (routing_env đảm bảo),
    # giữ lại để phòng thủ nếu logic env thay đổi
    if dropped:
        reward -= 0.2

    return float(reward)


def compute_shaping_reward(
    current_link_delay: float,
    current_link_utilization: float,
    current_link_queue_util: float,
) -> float:
    """
    Reward shaping cho các bước trung gian (intermediate hops).
    Chỉ phạt dựa trên link vừa đi qua — không dùng tổng tích lũy.

    Args:
        current_link_delay       : delay của link vừa chọn (ms)
        current_link_utilization : utilization băng thông của link đó [0,1]
        current_link_queue_util  : tỉ lệ đầy queue của link đó [0,1]

    Returns:
        shaping reward trong [-0.5, 0)
    """
    shaping = 0.0

    # Phạt cố định mỗi hop — khuyến khích path ngắn
    shaping -= 0.025

    # Phạt delay của riêng link này (không tích lũy)
    shaping -= min(1.0, current_link_delay / 10.0) * 0.05

    # Phạt utilization băng thông link hiện tại
    shaping -= current_link_utilization * 0.05

    # Phạt queue_util link hiện tại — cảnh báo sắp drop
    shaping -= current_link_queue_util * 0.05

    return max(-0.5, float(shaping))