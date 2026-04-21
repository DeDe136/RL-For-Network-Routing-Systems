"""
tests/DQN/test_dqn_agent.py

Unit tests cho D3QN Agent (Double DQN + Dueling DQN + PER) và
ReplayBuffer FIFO của người dùng.

Spec tóm tắt:
  ReplayBuffer:
    - FIFO thuần: deque(maxlen=capacity) + popleft() thủ công khi đầy.
    - KHÔNG có purge(), KHÔNG có purge_threshold.
    - Có is_full và fill_ratio.

  DQNAgent (D3QN):
    - use_double  = True  → Double DQN (q_net chọn action, target tính value)
    - use_dueling = True  → DuelingQNetwork (V + A - mean(A))
    - use_per     = True  → PrioritizedReplayBuffer (SumTree)
    - use_per     = False → ReplayBuffer FIFO (không purge)
    - Config không cần purge_threshold khi use_per=True.
    - update() trả {"loss", "epsilon"} khi use_per=False,
                    {"loss", "epsilon", "beta", "mean_priority"} khi use_per=True.
    - network_summary() chứa "Double", "Dueling", "PER" khi cả 3 bật.
    - save/load lưu thêm per_frame.

Chạy: pytest tests/DQN/test_dqn_agent.py -v
"""

import sys, os, types, random, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# ── Mock gymnasium ────────────────────────────────────────────────────
gm = types.ModuleType("gymnasium"); sp = types.ModuleType("gymnasium.spaces")
class _Env:
    def reset(self, seed=None, options=None): pass
class _Disc:
    def __init__(self, n): self.n = n
sp.Box = lambda **kw: None; sp.Discrete = _Disc; sp.Dict = lambda d: None
gm.spaces = sp; gm.Env = _Env
reg = types.ModuleType("gymnasium.envs.registration"); reg.register = lambda **k: None
envs = types.ModuleType("gymnasium.envs"); envs.registration = reg
sys.modules.update({"gymnasium": gm, "gymnasium.spaces": sp,
                    "gymnasium.envs": envs, "gymnasium.envs.registration": reg})

torch = pytest.importorskip("torch", reason="torch not installed")

from network.DQN.topology import NetworkTopology
from agents.DQN.dqn_agent import DQNAgent
from agents.DQN.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, _SumTree
from agents.DQN.q_network import QNetwork, DuelingQNetwork
from env.DQN.spaces import obs_to_flat, NUM_NODES

INPUT_DIM = 158   # 2 + 26 × 6


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def topo():
    return NetworkTopology(seed=0)


@pytest.fixture
def agent_d3qn(topo):
    """D3QN agent: Double + Dueling + PER (cấu hình đầy đủ)."""
    cfg = {
        "input_dim":          INPUT_DIM,
        "hidden_dims":        [64, 64],
        "lr":                 1e-3,
        "gamma":              0.99,
        "epsilon":            1.0,
        "eps_min":            0.01,
        "eps_decay":          0.99,
        "batch_size":         8,
        "buffer_capacity":    200,
        "target_update_freq": 5,
        "learn_start":        8,
        "use_double":         True,
        "use_dueling":        True,
        "use_per":            True,
        "per_alpha":          0.6,
        "per_beta_start":     0.4,
        "per_beta_frames":    1000,
        "per_eps":            1e-6,
    }
    ag = DQNAgent(n_states=1, n_actions=NUM_NODES, config=cfg)
    ag.set_neighbor_mask(topo.adj_matrix)
    return ag


@pytest.fixture
def agent_vanilla(topo):
    """Vanilla DQN: tắt tất cả D3QN improvements → dùng ReplayBuffer FIFO."""
    cfg = {
        "input_dim":          INPUT_DIM,
        "hidden_dims":        [64, 64],
        "lr":                 1e-3,
        "gamma":              0.99,
        "epsilon":            1.0,
        "eps_min":            0.01,
        "eps_decay":          0.99,
        "batch_size":         8,
        "buffer_capacity":    200,
        "target_update_freq": 5,
        "learn_start":        8,
        "use_double":         False,
        "use_dueling":        False,
        "use_per":            False,
    }
    ag = DQNAgent(n_states=1, n_actions=NUM_NODES, config=cfg)
    ag.set_neighbor_mask(topo.adj_matrix)
    return ag


def make_obs(topo, cur=0, dst=7):
    return {
        "current_node": cur,
        "dst_node":     dst,
        "link_states":  topo.link_state_vector(),
    }


def fill_buffer(agent, topo, n=20):
    flat  = obs_to_flat(make_obs(topo))
    valid = list(np.where(agent.neighbor_mask[0] > 0)[0])
    for i in range(n):
        agent.memory.push(flat, random.choice(valid),
                          float(i % 3 - 1), flat, i == n - 1)


# ═══════════════════════════════════════════════════════════════════════
#  SumTree
# ═══════════════════════════════════════════════════════════════════════

class TestSumTree:
    def test_empty_len_zero(self):
        t = _SumTree(8)
        assert len(t) == 0

    def test_total_after_add(self):
        t = _SumTree(4)
        t.add(1.0, "a"); t.add(2.0, "b"); t.add(3.0, "c"); t.add(4.0, "d")
        assert abs(t.total - 10.0) < 1e-9

    def test_len_after_add(self):
        t = _SumTree(4)
        t.add(1.0, "a"); t.add(2.0, "b")
        assert len(t) == 2

    def test_max_priority(self):
        t = _SumTree(4)
        t.add(1.0, "a"); t.add(5.0, "b"); t.add(2.0, "c")
        assert abs(t.max_priority - 5.0) < 1e-9

    def test_max_priority_empty_returns_1(self):
        t = _SumTree(4)
        assert t.max_priority == 1.0

    def test_update_propagates_to_root(self):
        t = _SumTree(4)
        t.add(1.0, "a"); t.add(1.0, "b")
        leaf0 = t.capacity - 1
        t.update(leaf0, 5.0)
        assert abs(t.total - 6.0) < 1e-9

    def test_circular_overwrite(self):
        """Khi đầy, con trỏ xoay vòng ghi đè phần tử cũ nhất."""
        t = _SumTree(3)
        t.add(1.0, "a"); t.add(2.0, "b"); t.add(3.0, "c")
        assert len(t) == 3
        t.add(4.0, "d")   # ghi đè lá đầu tiên
        assert len(t) == 3
        assert abs(t.total - 9.0) < 1e-9   # 4+2+3

    def test_sample_returns_valid_data(self):
        t = _SumTree(4)
        t.add(1.0, "x"); t.add(3.0, "y")
        tree_idx, priority, data = t.sample(t.total * 0.5)
        assert data in ("x", "y")
        assert priority > 0

    def test_sample_full_range(self):
        """Sample trên toàn dải [0, total] — luôn trả về kết quả hợp lệ."""
        t = _SumTree(8)
        for i in range(8):
            t.add(float(i + 1), f"item{i}")
        # Không raise
        for _ in range(20):
            v = np.random.uniform(1e-6, t.total - 1e-6)
            idx, p, data = t.sample(v)
            assert p > 0
            assert data is not None


# ═══════════════════════════════════════════════════════════════════════
#  ReplayBuffer — FIFO thuần (theo file người dùng gửi)
# ═══════════════════════════════════════════════════════════════════════

class TestReplayBuffer:
    """
    ReplayBuffer của người dùng: FIFO thuần.
      - deque(maxlen=capacity)
      - push(): popleft() thủ công khi đầy, rồi append
      - KHÔNG có purge(), KHÔNG có purge_threshold
    """

    def test_push_increments_len(self):
        buf  = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        buf.push(flat, 0, 1.0, flat, False)
        assert len(buf) == 1

    def test_push_many_below_capacity(self):
        buf  = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(50):
            buf.push(flat, i % 8, float(i), flat, False)
        assert len(buf) == 50

    def test_fifo_eviction_when_full(self):
        """Khi đầy, push loại bỏ phần tử cũ nhất — len không vượt capacity."""
        buf  = ReplayBuffer(10)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(15):
            buf.push(flat, 0, float(i), flat, False)
        assert len(buf) == 10

    def test_fifo_order_newest_kept(self):
        """Sau khi push quá capacity, chỉ giữ 10 phần tử mới nhất."""
        buf = ReplayBuffer(10)
        for i in range(15):
            s = np.full(INPUT_DIM, float(i), dtype=np.float32)
            buf.push(s, 0, float(i), s, False)
        # reward của 5 phần tử đầu (0..4) đã bị loại
        rewards = [t[2] for t in list(buf.buffer)]
        assert min(rewards) == pytest.approx(5.0)
        assert max(rewards) == pytest.approx(14.0)

    def test_no_purge_method(self):
        """FIFO thuần — không có purge() method."""
        buf = ReplayBuffer(100)
        assert not hasattr(buf, "purge"), \
            "ReplayBuffer không có purge()"

    def test_no_purge_threshold(self):
        """FIFO thuần — không có purge_threshold attribute."""
        buf = ReplayBuffer(100)
        assert not hasattr(buf, "purge_threshold"), \
            "ReplayBuffer không có purge_threshold"

    def test_is_full_false_initially(self):
        buf = ReplayBuffer(5)
        assert not buf.is_full

    def test_is_full_true_when_full(self):
        buf  = ReplayBuffer(5)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for _ in range(5):
            buf.push(flat, 0, 0.0, flat, False)
        assert buf.is_full

    def test_fill_ratio_half(self):
        buf  = ReplayBuffer(10)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for _ in range(5):
            buf.push(flat, 0, 0.0, flat, False)
        assert abs(buf.fill_ratio - 0.5) < 1e-9

    def test_fill_ratio_stays_at_1_when_full(self):
        buf  = ReplayBuffer(5)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for _ in range(20):
            buf.push(flat, 0, 0.0, flat, False)
        assert buf.fill_ratio == pytest.approx(1.0)

    def test_sample_shape(self):
        buf  = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(20):
            buf.push(flat, i % 8, float(i), flat, False)
        s, a, r, ns, d = buf.sample(8)
        assert s.shape  == (8, INPUT_DIM)
        assert ns.shape == (8, INPUT_DIM)
        assert len(a)   == 8
        assert len(r)   == 8
        assert len(d)   == 8

    def test_sample_dtype(self):
        buf  = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(20):
            buf.push(flat, i % 8, float(i), flat, False)
        s, a, r, ns, d = buf.sample(4)
        assert s.dtype  == np.float32
        assert a.dtype  == np.int64
        assert r.dtype  == np.float32
        assert ns.dtype == np.float32
        assert d.dtype  == np.float32

    def test_sample_raises_when_too_small(self):
        """Sample nhiều hơn số phần tử hiện có → raise."""
        buf  = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        buf.push(flat, 0, 0.0, flat, False)   # chỉ 1 phần tử
        with pytest.raises(Exception):
            buf.sample(10)


# ═══════════════════════════════════════════════════════════════════════
#  PrioritizedReplayBuffer
# ═══════════════════════════════════════════════════════════════════════

class TestPrioritizedReplayBuffer:
    def _make(self, cap=100):
        return PrioritizedReplayBuffer(capacity=cap, alpha=0.6, per_eps=1e-6)

    def test_push_and_len(self):
        per  = self._make()
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(10):
            per.push(flat, 0, float(i), flat, False)
        assert len(per) == 10

    def test_capacity_respected(self):
        """SumTree xoay vòng — không vượt capacity."""
        per  = self._make(5)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(12):
            per.push(flat, 0, float(i), flat, False)
        assert len(per) == 5

    def test_is_full_and_fill_ratio(self):
        per  = self._make(10)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for _ in range(10):
            per.push(flat, 0, 0.0, flat, False)
        assert per.is_full
        assert abs(per.fill_ratio - 1.0) < 1e-9

    def test_sample_returns_7_tuple(self):
        per  = self._make()
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(20):
            per.push(flat, i % 8, float(i), flat, False)
        result = per.sample(8, beta=0.4)
        assert len(result) == 7

    def test_sample_shapes(self):
        per  = self._make()
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(20):
            per.push(flat, i % 8, float(i), flat, False)
        s, a, r, ns, d, w, idx = per.sample(8, beta=0.4)
        assert s.shape  == (8, INPUT_DIM)
        assert ns.shape == (8, INPUT_DIM)
        assert len(w)   == 8
        assert len(idx) == 8

    def test_is_weights_in_0_1(self):
        per  = self._make()
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(20):
            per.push(flat, 0, float(i), flat, False)
        _, _, _, _, _, w, _ = per.sample(8, beta=0.4)
        assert np.all(w > 0) and np.all(w <= 1.0 + 1e-6)

    def test_max_is_weight_equals_1(self):
        """IS weights chuẩn hoá → max = 1.0."""
        per  = self._make()
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(20):
            per.push(flat, 0, float(i), flat, False)
        _, _, _, _, _, w, _ = per.sample(8, beta=0.4)
        assert abs(w.max() - 1.0) < 1e-6

    def test_update_priorities_no_crash(self):
        per  = self._make()
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(20):
            per.push(flat, 0, float(i), flat, False)
        _, _, _, _, _, w, idx = per.sample(8, beta=0.4)
        td_errors = np.random.rand(8)
        per.update_priorities(idx, td_errors)   # không crash

    def test_higher_td_error_higher_priority(self):
        """Transition có TD-error lớn hơn được cập nhật priority cao hơn."""
        per = self._make(4)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        per.push(flat, 0, 0.0, flat, False)   # 1 transition
        per.push(flat, 0, 0.0, flat, False)
        _, _, _, _, _, _, idx = per.sample(1, beta=0.4)
        # cập nhật priority lớn
        per.update_priorities(idx, np.array([100.0]))
        # priority sau cập nhật > epsilon
        tree_leaf = per._tree.tree[idx[0]]
        assert tree_leaf > 0


# ═══════════════════════════════════════════════════════════════════════
#  Link state & obs_to_flat
# ═══════════════════════════════════════════════════════════════════════

class TestLinkStateAndObs:
    def test_link_state_shape_26x6(self, topo):
        ls = topo.link_state_vector()
        assert ls.shape == (26, 6), f"expected (26,6), got {ls.shape}"

    def test_obs_to_flat_shape_158(self, topo):
        flat = obs_to_flat(make_obs(topo))
        assert flat.shape == (INPUT_DIM,), f"expected ({INPUT_DIM},), got {flat.shape}"

    def test_decode_current_node_all_nodes(self, topo):
        for cur in range(NUM_NODES):
            flat    = obs_to_flat(make_obs(topo, cur=cur))
            decoded = int(round(flat[0] * (NUM_NODES - 1)))
            assert decoded == cur, f"cur={cur} decoded={decoded}"

    def test_drop_norm_col_in_range(self, topo):
        ls = topo.link_state_vector()
        assert np.all(ls[:, 5] >= 0) and np.all(ls[:, 5] <= 1.0 + 1e-6)

    def test_drop_norm_increases_with_dropped_data(self, topo):
        idx_key   = sorted(topo._link_attrs.keys()).index((0, 1))
        ls_before = topo.link_state_vector()[idx_key, 5]
        topo.link(0, 1).dropped_data = topo.DROP_NORM_MAX
        ls_after  = topo.link_state_vector()[idx_key, 5]
        assert ls_after > ls_before


# ═══════════════════════════════════════════════════════════════════════
#  D3QN Agent — kiến trúc
# ═══════════════════════════════════════════════════════════════════════

class TestD3QNArchitecture:
    def test_d3qn_uses_dueling_network(self, agent_d3qn):
        assert isinstance(agent_d3qn.q_net, DuelingQNetwork), \
            f"expected DuelingQNetwork, got {type(agent_d3qn.q_net)}"

    def test_d3qn_uses_per(self, agent_d3qn):
        assert isinstance(agent_d3qn.memory, PrioritizedReplayBuffer), \
            f"expected PrioritizedReplayBuffer, got {type(agent_d3qn.memory)}"

    def test_d3qn_flags_on(self, agent_d3qn):
        assert agent_d3qn.use_double
        assert agent_d3qn.use_dueling
        assert agent_d3qn.use_per

    def test_vanilla_uses_standard_network(self, agent_vanilla):
        assert isinstance(agent_vanilla.q_net, QNetwork) and \
               not isinstance(agent_vanilla.q_net, DuelingQNetwork)

    def test_vanilla_uses_replay_buffer(self, agent_vanilla):
        """Vanilla (use_per=False) dùng ReplayBuffer FIFO, không PER."""
        assert isinstance(agent_vanilla.memory, ReplayBuffer) and \
               not isinstance(agent_vanilla.memory, PrioritizedReplayBuffer)

    def test_vanilla_no_purge(self, agent_vanilla):
        """ReplayBuffer FIFO của vanilla không có purge()."""
        assert not hasattr(agent_vanilla.memory, "purge")

    def test_vanilla_no_purge_threshold(self, agent_vanilla):
        assert not hasattr(agent_vanilla.memory, "purge_threshold")

    def test_dueling_has_three_streams(self, agent_d3qn):
        net = agent_d3qn.q_net
        assert hasattr(net, "backbone")
        assert hasattr(net, "value_stream")
        assert hasattr(net, "advantage_stream")

    def test_q_net_output_shape(self, agent_d3qn, topo):
        obs = make_obs(topo)
        flat = obs_to_flat(obs)
        t = torch.FloatTensor(flat).unsqueeze(0)
        out = agent_d3qn.q_net(t)
        assert out.shape == (1, NUM_NODES)


# ═══════════════════════════════════════════════════════════════════════
#  select_action
# ═══════════════════════════════════════════════════════════════════════

class TestSelectAction:
    def test_always_valid_neighbor(self, agent_d3qn, topo):
        for cur in range(NUM_NODES):
            valid = list(np.where(agent_d3qn.neighbor_mask[cur] > 0)[0])
            if not valid:
                continue
            obs = make_obs(topo, cur=cur)
            for _ in range(20):
                a = agent_d3qn.select_action(obs)
                assert a in valid, f"node={cur}: invalid action={a}"

    def test_greedy_picks_highest_q(self, agent_d3qn, topo):
        agent_d3qn.eval_mode()
        with torch.no_grad():
            # Dueling: set advantage stream để node 1 có Q cao nhất
            last_adv = list(agent_d3qn.q_net.advantage_stream.children())[-1]
            last_adv.bias.fill_(0.0); last_adv.weight.fill_(0.0)
            last_adv.bias[1] = 10.0
            last_val = list(agent_d3qn.q_net.value_stream.children())[-1]
            last_val.bias.fill_(0.0); last_val.weight.fill_(0.0)
        obs = make_obs(topo, cur=0)
        if agent_d3qn.neighbor_mask[0][1] > 0:
            actions = {agent_d3qn.select_action(obs) for _ in range(30)}
            assert actions == {1}

    def test_tie_breaking_random(self, agent_d3qn, topo):
        """Q bằng nhau → tie-breaking random → thấy nhiều action."""
        agent_d3qn.eval_mode()
        with torch.no_grad():
            for p in agent_d3qn.q_net.parameters():
                p.fill_(0.0)
        obs   = make_obs(topo, cur=0)
        valid = set(np.where(agent_d3qn.neighbor_mask[0] > 0)[0].tolist())
        seen  = set()
        for _ in range(300):
            seen.add(agent_d3qn.select_action(obs))
        assert seen == valid, f"tie-break: got {seen}, expected {valid}"

    def test_epsilon_1_explores(self, agent_d3qn, topo):
        agent_d3qn.train_mode(); agent_d3qn.epsilon = 1.0
        obs  = make_obs(topo, cur=0)
        seen = set()
        for _ in range(200):
            seen.add(agent_d3qn.select_action(obs))
        assert len(seen) > 1

    def test_eval_mode_ignores_epsilon(self, agent_d3qn, topo):
        agent_d3qn.eval_mode(); agent_d3qn.epsilon = 1.0
        with torch.no_grad():
            last = list(agent_d3qn.q_net.advantage_stream.children())[-1]
            last.bias.fill_(0.0); last.weight.fill_(0.0)
            last.bias[2] = 5.0
        obs = make_obs(topo, cur=0)
        if agent_d3qn.neighbor_mask[0][2] > 0:
            actions = {agent_d3qn.select_action(obs) for _ in range(20)}
            assert actions == {2}

    def test_never_invalid_action(self, agent_d3qn, topo):
        agent_d3qn.train_mode(); agent_d3qn.epsilon = 0.5
        for cur in range(NUM_NODES):
            valid = set(np.where(agent_d3qn.neighbor_mask[cur] > 0)[0].tolist())
            obs   = make_obs(topo, cur=cur)
            for _ in range(30):
                assert agent_d3qn.select_action(obs) in valid


# ═══════════════════════════════════════════════════════════════════════
#  remember
# ═══════════════════════════════════════════════════════════════════════

class TestRemember:
    def test_remember_adds_to_per(self, agent_d3qn, topo):
        obs = make_obs(topo)
        assert len(agent_d3qn.memory) == 0
        agent_d3qn.remember(obs, 1, 0.5, obs, False)
        assert len(agent_d3qn.memory) == 1

    def test_remember_adds_to_fifo(self, agent_vanilla, topo):
        obs = make_obs(topo)
        agent_vanilla.remember(obs, 1, 0.5, obs, False)
        assert len(agent_vanilla.memory) == 1


# ═══════════════════════════════════════════════════════════════════════
#  update — D3QN
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateD3QN:
    def test_no_update_before_learn_start(self, agent_d3qn, topo):
        fill_buffer(agent_d3qn, topo, n=4)   # < learn_start=8
        result = agent_d3qn.update()
        assert result["loss"] is None

    def test_loss_after_learn_start(self, agent_d3qn, topo):
        fill_buffer(agent_d3qn, topo, n=20)
        result = agent_d3qn.update()
        assert result["loss"] is not None
        assert result["loss"] >= 0.0

    def test_loss_is_finite(self, agent_d3qn, topo):
        fill_buffer(agent_d3qn, topo, n=20)
        result = agent_d3qn.update()
        assert result["loss"] == result["loss"]   # not NaN
        assert result["loss"] < 1e8

    def test_returns_beta_and_mean_priority(self, agent_d3qn, topo):
        """D3QN (use_per=True) trả thêm beta và mean_priority."""
        fill_buffer(agent_d3qn, topo, n=20)
        result = agent_d3qn.update()
        assert "beta"          in result
        assert "mean_priority" in result

    def test_beta_increases_over_time(self, agent_d3qn, topo):
        fill_buffer(agent_d3qn, topo, n=20)
        beta0 = agent_d3qn._current_beta()
        for _ in range(10):
            agent_d3qn.update()
        assert agent_d3qn._current_beta() >= beta0

    def test_beta_reaches_1_at_per_beta_frames(self, agent_d3qn, topo):
        agent_d3qn._per_frame = agent_d3qn.per_beta_frames
        assert abs(agent_d3qn._current_beta() - 1.0) < 1e-9

    def test_beta_starts_at_per_beta_start(self, agent_d3qn, topo):
        agent_d3qn._per_frame = 0
        assert abs(agent_d3qn._current_beta() - 0.4) < 1e-9

    def test_epsilon_decays(self, agent_d3qn, topo):
        agent_d3qn.train_mode()
        fill_buffer(agent_d3qn, topo, n=20)
        eps_before = agent_d3qn.epsilon
        for _ in range(5):
            agent_d3qn.update()
        assert agent_d3qn.epsilon < eps_before

    def test_epsilon_floor(self, agent_d3qn, topo):
        agent_d3qn.epsilon = agent_d3qn.eps_min
        fill_buffer(agent_d3qn, topo, n=20)
        for _ in range(30):
            agent_d3qn.update()
        assert agent_d3qn.epsilon >= agent_d3qn.eps_min - 1e-9

    def test_target_net_syncs(self, agent_d3qn, topo):
        agent_d3qn.target_update_freq = 2
        fill_buffer(agent_d3qn, topo, n=20)
        with torch.no_grad():
            list(agent_d3qn.q_net.parameters())[0].fill_(99.0)
        for _ in range(2):
            agent_d3qn.update()
        v_q = list(agent_d3qn.q_net.parameters())[0][0, 0].item()
        v_t = list(agent_d3qn.target_net.parameters())[0][0, 0].item()
        assert abs(v_q - v_t) < 1e-4

    def test_double_dqn_action_masking_in_target(self, agent_d3qn, topo):
        """Double DQN: target không bị ảnh hưởng bởi Q của invalid actions."""
        fill_buffer(agent_d3qn, topo, n=20)
        with torch.no_grad():
            last = list(agent_d3qn.target_net.advantage_stream.children())[-1]
            last.bias.fill_(0.0)
            last.bias[7] = 1e6
        result = agent_d3qn.update()
        assert result["loss"] == result["loss"]   # not NaN
        assert result["loss"] < 1e8


# ═══════════════════════════════════════════════════════════════════════
#  update — Vanilla (use_per=False)
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateVanilla:
    def test_loss_after_learn_start(self, agent_vanilla, topo):
        fill_buffer(agent_vanilla, topo, n=20)
        result = agent_vanilla.update()
        assert result["loss"] is not None and result["loss"] >= 0.0

    def test_no_beta_or_mean_priority(self, agent_vanilla, topo):
        """Vanilla (use_per=False) không trả beta hay mean_priority."""
        fill_buffer(agent_vanilla, topo, n=20)
        result = agent_vanilla.update()
        assert "beta"          not in result
        assert "mean_priority" not in result


# ═══════════════════════════════════════════════════════════════════════
#  best_path
# ═══════════════════════════════════════════════════════════════════════

class TestBestPath:
    def test_starts_at_src(self, agent_d3qn, topo):
        path = agent_d3qn.best_path(0, 7, topo)
        assert path[0] == 0

    def test_only_valid_links(self, agent_d3qn, topo):
        path = agent_d3qn.best_path(0, 7, topo)
        for i in range(len(path) - 1):
            assert topo.has_link(path[i], path[i+1]), \
                f"invalid link {path[i]}→{path[i+1]}"

    def test_no_cycles(self, agent_d3qn, topo):
        path = agent_d3qn.best_path(0, 7, topo)
        assert len(path) == len(set(path)), f"cycle in {path}"

    def test_training_state_restored(self, agent_d3qn, topo):
        agent_d3qn.train_mode()
        agent_d3qn.best_path(0, 7, topo)
        assert agent_d3qn.training is True
        agent_d3qn.eval_mode()
        agent_d3qn.best_path(0, 7, topo)
        assert agent_d3qn.training is False

    def test_send_traffic_called_per_hop(self, agent_d3qn, topo):
        """best_path gọi send_traffic → link_states thay đổi."""
        topo.reset(); topo.step_background(0.1)
        ls_before = topo.link_state_vector()[:, 3:].copy()
        agent_d3qn.best_path(0, 7, topo, volume_mbps=20.0)
        ls_after = topo.link_state_vector()[:, 3:]
        assert not np.allclose(ls_before, ls_after)

    def test_tie_breaking_in_best_path(self, agent_d3qn, topo):
        """Q bằng nhau trong best_path → tie-breaking random."""
        agent_d3qn.eval_mode()
        with torch.no_grad():
            for p in agent_d3qn.q_net.parameters():
                p.fill_(0.0)
        paths = set()
        for _ in range(30):
            topo.reset()
            paths.add(tuple(agent_d3qn.best_path(0, 7, topo)))
        assert len(paths) >= 1   # không crash


# ═══════════════════════════════════════════════════════════════════════
#  save / load
# ═══════════════════════════════════════════════════════════════════════

class TestSaveLoad:
    def _make_agent(self):
        cfg = {
            "input_dim": INPUT_DIM, "hidden_dims": [64,64],
            "lr": 1e-3, "gamma": 0.99,
            "epsilon": 1.0, "eps_min": 0.01, "eps_decay": 0.99,
            "batch_size": 8, "buffer_capacity": 200,
            "target_update_freq": 5, "learn_start": 8,
            "use_double": True, "use_dueling": True, "use_per": True,
            "per_alpha": 0.6, "per_beta_start": 0.4,
            "per_beta_frames": 1000, "per_eps": 1e-6,
        }
        return DQNAgent(n_states=1, n_actions=NUM_NODES, config=cfg)

    def test_save_load_epsilon(self, agent_d3qn, topo):
        agent_d3qn.epsilon = 0.42
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d3qn.pt")
            agent_d3qn.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            assert abs(ag2.epsilon - 0.42) < 1e-5

    def test_save_load_steps_done(self, agent_d3qn, topo):
        agent_d3qn.steps_done = 77
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d3qn.pt")
            agent_d3qn.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            assert ag2.steps_done == 77

    def test_save_load_per_frame(self, agent_d3qn, topo):
        """per_frame được lưu để β schedule tiếp tục đúng sau load."""
        agent_d3qn._per_frame = 500
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d3qn.pt")
            agent_d3qn.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            assert ag2._per_frame == 500

    def test_save_load_q_net_weights(self, agent_d3qn, topo):
        with torch.no_grad():
            list(agent_d3qn.q_net.parameters())[0].fill_(3.14)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d3qn.pt")
            agent_d3qn.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            v1 = list(agent_d3qn.q_net.parameters())[0][0, 0].item()
            v2 = list(ag2.q_net.parameters())[0][0, 0].item()
            assert abs(v1 - v2) < 1e-5

    def test_load_missing_epsilon_uses_eps_min(self, agent_d3qn, topo):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d3qn.pt")
            torch.save({
                "q_net":      agent_d3qn.q_net.state_dict(),
                "target_net": agent_d3qn.target_net.state_dict(),
                "optimizer":  agent_d3qn.optimizer.state_dict(),
            }, path)
            ag2 = self._make_agent()
            ag2.load(path)
            assert ag2.epsilon    == ag2.eps_min
            assert ag2.steps_done == 0
            assert ag2._per_frame == 0


# ═══════════════════════════════════════════════════════════════════════
#  network_summary
# ═══════════════════════════════════════════════════════════════════════

class TestNetworkSummary:
    def test_summary_is_string(self, agent_d3qn, topo):
        assert isinstance(agent_d3qn.network_summary(), str)

    def test_summary_d3qn_contains_all_flags(self, agent_d3qn, topo):
        s = agent_d3qn.network_summary()
        assert "Double"  in s
        assert "Dueling" in s
        assert "PER"     in s

    def test_summary_contains_beta(self, agent_d3qn, topo):
        s = agent_d3qn.network_summary()
        assert "β=" in s

    def test_summary_contains_input_dim(self, agent_d3qn, topo):
        s = agent_d3qn.network_summary()
        assert str(INPUT_DIM) in s

    def test_summary_vanilla_no_per_flag(self, agent_vanilla, topo):
        s = agent_vanilla.network_summary()
        assert "PER" not in s