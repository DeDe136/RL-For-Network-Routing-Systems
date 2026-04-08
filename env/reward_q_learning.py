"""
env/reward_q_learning.py

Hàm reward cho Q-learning dựa trên delay.
"""

def compute_step_reward_q_learning(
    current_link_delay: float,
) -> float:
    """
    Reward shaping cho từng bước (intermediate hops) dựa trên delay.
    Dùng để guide agent trong quá trình training.

    Args:
        current_link_delay   : delay của link vừa chọn (ms)

    Returns:
        step reward trong [-1, 0]:
          - Link delay thấp → reward gần 0
          - Link delay cao → reward âm
    """

    # Delay tham chiếu cho normalization (ms), default 10ms
    reference_delay: float = 10.0
    
    # Phạt dựa vào delay của link hiện tại
    # step_reward = -(delay / reference_delay), capped ở -1
    step_reward = -(min(current_link_delay, reference_delay) / reference_delay)
    
    return max(-1.0, float(step_reward))
