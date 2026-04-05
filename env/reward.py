"""
env/reward.py

Hàm reward cho bài toán định tuyến.

Mục tiêu: tối thiểu delay + drop, tối đa throughput.
Reward nằm trong khoảng [-1, +1].
"""


def compute_reward(
    total_delay: float,
    dropped: bool,
    hops: int,
    utilization: float,
    path_found: bool,
) -> float:
    """
    Args:
        total_delay   : tổng delay trên path (ms)
        dropped       : có gói bị drop không
        hops          : số hop
        utilization   : utilization trung bình các link trên path
        path_found    : agent đã đến đích chưa

    Returns:
        reward float trong [-2, +1]
    """
    if not path_found:
        return -2.0                     # không đến được đích — phạt nặng

    # Thưởng cơ bản khi đến đích
    reward = 1.0

    # Phạt theo delay (chuẩn hoá với 50ms là "xấu")
    reward -= min(1.0, total_delay / 50.0) * 0.4

    # Phạt nếu có drop
    if dropped:
        reward -= 0.3

    # Phạt hop count thừa (càng ngắn càng tốt, giả sử tối đa 6 hops)
    reward -= min(1.0, hops / 6.0) * 0.2

    # Phạt utilization cao
    reward -= utilization * 0.1

    return float(reward)