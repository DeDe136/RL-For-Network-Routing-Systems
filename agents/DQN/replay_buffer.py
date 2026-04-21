"""
agents/DQN/replay_buffer.py

Hai loại Experience Replay Buffer:

  ReplayBuffer           : Uniform replay FIFO — không có purge.
  PrioritizedReplayBuffer: Prioritized Experience Replay (PER) — Schaul et al. 2015.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PrioritizedReplayBuffer — Tại sao và cơ chế hoạt động
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vấn đề của Uniform Replay:
  - Sample đều mọi transition → bỏ qua những lần agent sai nhiều nhất.
  - Transition có TD-error lớn (sai nhiều) = cơ hội học quan trọng
    nhưng chỉ được sample với xác suất 1/N.

Giải pháp — PER:
  Mỗi transition i có priority p_i. Xác suất sample:
    P(i) = p_i^α / Σ_k p_k^α

  α (per_alpha) ∈ [0,1]:
    α=0 → uniform (như cũ)
    α=1 → hoàn toàn theo priority
    Giá trị tốt: α=0.6

  Priority ban đầu khi push: max priority hiện có (đảm bảo transition
  mới luôn được thử ít nhất 1 lần).

  Priority cập nhật sau update:
    p_i = |δ_i| + ε   (δ = TD-error, ε = per_eps nhỏ tránh p=0)

Importance Sampling weights (IS weights):
  Sample có bias → cần bù trừ khi tính loss:
    w_i = (N · P(i))^(-β) / max_j w_j   (chuẩn hoá)

  β (per_beta) tăng dần từ per_beta_start → 1.0 trong quá trình training:
    - β nhỏ ở đầu: cho phép bias nhỏ, học nhanh hơn
    - β=1 ở cuối: unbiased hoàn toàn

Cấu trúc SumTree:
  Binary tree lưu priorities. Node lá = transition.
  Node cha = tổng priorities con.
  → Sample O(log N), update O(log N), thay vì O(N).
"""

from collections import deque
import random
import numpy as np
from typing import Tuple, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ReplayBuffer (giữ nguyên interface)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReplayBuffer:
    """
    Bộ đệm vòng FIFO — Uniform Experience Replay.

    Cơ chế push: nếu buffer đã đầy (len == capacity) → bỏ phần tử cũ
    nhất (popleft) trước khi append phần tử mới.
    Không có purge(), không có purge_threshold.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Thêm transition vào buffer theo cơ chế FIFO:
          - Nếu buffer đã đầy → bỏ phần tử đầu tiên (cũ nhất).
          - Push phần tử mới vào cuối.
        """
        if len(self.buffer) == self.capacity:
            self.buffer.popleft()
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> Tuple:
        """Lấy ngẫu nhiên batch_size transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        return len(self.buffer) == self.capacity

    @property
    def fill_ratio(self) -> float:
        return len(self.buffer) / self.capacity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SumTree — cấu trúc dữ liệu cho PER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _SumTree:
    """
    Binary SumTree cho Prioritized Experience Replay.

    Cấu trúc:
      - capacity lá (transitions) → tree có 2*capacity - 1 node.
      - Node [0..capacity-2]: node trong, lưu tổng priorities con.
      - Node [capacity-1..2*capacity-2]: lá, lưu priority của transition.

    Vị trí lá thứ i → tree[capacity - 1 + i].
    Write pointer xoay vòng: _ptr ∈ [0, capacity).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.zeros(capacity, dtype=object)   # transitions
        self._ptr     = 0      # vị trí ghi tiếp theo (circular)
        self._size    = 0      # số transitions thực sự có

    # ── Cập nhật priority ────────────────────────────────────────────

    def _propagate(self, idx: int, delta: float):
        """Cập nhật tổng từ lá lên gốc."""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, tree_idx: int, priority: float):
        """Cập nhật priority cho lá tree_idx."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    # ── Thêm transition ──────────────────────────────────────────────

    def add(self, priority: float, data) -> int:
        """
        Thêm transition với priority vào vị trí con trỏ hiện tại.
        Trả về tree_idx của lá vừa ghi.
        """
        tree_idx = self._ptr + self.capacity - 1
        self.data[self._ptr] = data
        self.update(tree_idx, priority)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return tree_idx

    # ── Sample theo priority ─────────────────────────────────────────

    def _retrieve(self, idx: int, value: float) -> int:
        """Tìm lá có cumulative priority ≥ value (binary search trên tree)."""
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    def sample(self, value: float) -> Tuple[int, float, object]:
        """
        Sample transition ứng với cumulative priority = value.

        Returns:
            (tree_idx, priority, data)
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    @property
    def total(self) -> float:
        """Tổng tất cả priorities (= tree[0])."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Priority lớn nhất trong số _size transitions."""
        if self._size == 0:
            return 1.0
        leaves = self.tree[self.capacity - 1: self.capacity - 1 + self._size]
        return float(np.max(leaves)) if len(leaves) > 0 else 1.0

    def __len__(self) -> int:
        return self._size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PrioritizedReplayBuffer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (Schaul et al. 2015).

    Interface:
      push(state, action, reward, next_state, done)
      sample(batch_size, beta) → (states, actions, rewards, next_states,
                                   dones, weights, tree_indices)
      update_priorities(tree_indices, td_errors)

    Hyperparameters:
      alpha     : mức độ ưu tiên hoá [0=uniform, 1=fully prioritized]
      beta      : IS correction strength, tăng dần → 1.0
      per_eps   : epsilon nhỏ thêm vào priority tránh p=0
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 per_eps: float = 1e-6):
        """
        Args:
            capacity: số transitions tối đa.
            alpha   : exponent ưu tiên hoá (0=uniform, 1=fully PER).
            per_eps : epsilon nhỏ tránh priority = 0.
        """
        self.capacity = capacity
        self.alpha    = alpha
        self.per_eps  = per_eps
        self._tree    = _SumTree(capacity)

    # ── Push ─────────────────────────────────────────────────────────

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Thêm transition với max priority (đảm bảo được sample ít nhất 1 lần).
        """
        max_p = self._tree.max_priority
        self._tree.add(max_p ** self.alpha,
                       (state, action, reward, next_state, float(done)))

    # ── Sample ───────────────────────────────────────────────────────

    def sample(self, batch_size: int,
               beta: float = 0.4) -> Tuple:
        """
        Sample batch_size transitions theo priority.

        Args:
            batch_size: số transitions.
            beta      : IS weight exponent (0=không bù, 1=fully unbiased).

        Returns:
            (states, actions, rewards, next_states, dones,
             is_weights, tree_indices)
             is_weights : (batch_size,) numpy float32 — dùng nhân loss.
             tree_indices: list[int] — dùng để update_priorities sau update.
        """
        assert len(self) >= batch_size, \
            f"Buffer chỉ có {len(self)} < batch_size={batch_size}"

        batch_size     = min(batch_size, len(self))
        segment        = self._tree.total / batch_size

        tree_indices   = []
        priorities     = []
        transitions    = []

        for i in range(batch_size):
            lo     = segment * i
            hi     = segment * (i + 1)
            value  = np.random.uniform(lo, hi)
            # Clamp value vào [per_eps, total - per_eps] tránh edge case
            value  = max(self.per_eps, min(value, self._tree.total - self.per_eps))
            idx, p, data = self._tree.sample(value)
            tree_indices.append(idx)
            priorities.append(p)
            transitions.append(data)

        # IS weights: w_i = (N · P(i))^(-β) / max_j w_j
        N          = len(self)
        probs      = np.array(priorities, dtype=np.float64) / self._tree.total
        probs      = np.clip(probs, 1e-12, 1.0)          # tránh log(0)
        raw_w      = (N * probs) ** (-beta)
        is_weights = (raw_w / raw_w.max()).astype(np.float32)

        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            is_weights,
            tree_indices,
        )

    # ── Update priorities ─────────────────────────────────────────────

    def update_priorities(self, tree_indices, td_errors: np.ndarray):
        """
        Cập nhật priority sau mỗi update step.

        Args:
            tree_indices: list[int] trả về từ sample().
            td_errors   : |δ| per transition (numpy array hoặc list).
        """
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float64))
        for idx, err in zip(tree_indices, td_errors):
            priority = (err + self.per_eps) ** self.alpha
            self._tree.update(idx, priority)

    # ── Utilities ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._tree)

    @property
    def is_full(self) -> bool:
        return len(self) == self.capacity

    @property
    def fill_ratio(self) -> float:
        return len(self) / self.capacity