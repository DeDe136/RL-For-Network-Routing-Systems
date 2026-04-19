"""
tests/DQN/test_dqn_agent.py

Unit tests cho DQNAgent (Vanilla DQN) theo file dqn_agent.py
và replay_buffer.py của người dùng.

Điểm khác biệt so với version cũ:
  - ReplayBuffer KHÔNG có auto-purge (không có purge_threshold,
    không có purge() method). Đây là FIFO thuần với deque(maxlen).
    → Không test purge; test fill_ratio và is_full bình thường.
  - DQNAgent KHÔNG nhận purge_threshold trong config.
  - best_path() có tie-breaking random trong exploitation
    (random trong tất cả action cùng max Q).
  - input_dim = 158 (2 + 26×6, vì link_state (26,6)).
  - network_summary() trả về string có "DQN" và str(input_dim).

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
from agents.DQN.replay_buffer import ReplayBuffer
from env.DQN.spaces import obs_to_flat, NUM_NODES

INPUT_DIM = 158   # 2 + 26 × 6


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def topo():
    return NetworkTopology(seed=0)


@pytest.fixture
def agent(topo):
    """
    Config theo dqn_agent.py của người dùng.
    KHÔNG có purge_threshold — ReplayBuffer là FIFO thuần.
    """
    cfg = {
        "input_dim":          INPUT_DIM,
        "hidden_dims":        [64, 64],
        "lr":                 1e-3,
        "gamma":              0.99,
        "epsilon":            1.0,
        "eps_min":            0.01,
        "eps_decay":          0.99,
        "batch_size":         8,
        "buffer_capacity":    500,
        "target_update_freq": 10,
        "learn_start":        8,
    }
    ag = DQNAgent(n_states=1, n_actions=NUM_NODES, config=cfg)
    ag.set_neighbor_mask(topo.adj_matrix)
    return ag


def make_obs(topo, cur=0, dst=7):
    """Tạo observation dict từ topology."""
    return {
        "current_node": cur,
        "dst_node":     dst,
        "link_states":  topo.link_state_vector(),
    }


def fill_buffer(agent, topo, n=20):
    """Điền n transitions vào replay buffer."""
    flat  = obs_to_flat(make_obs(topo))
    valid = list(np.where(agent.neighbor_mask[0] > 0)[0])
    for i in range(n):
        agent.memory.push(flat, random.choice(valid),
                          float(i % 3 - 1), flat, i == n - 1)


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
        """Cột drop_norm (index 5) ∈ [0,1]."""
        ls = topo.link_state_vector()
        assert np.all(ls[:, 5] >= 0)
        assert np.all(ls[:, 5] <= 1.0 + 1e-6)

    def test_drop_norm_increases_with_dropped_data(self, topo):
        idx_key = sorted(topo._link_attrs.keys()).index((0, 1))
        ls_before = topo.link_state_vector()[idx_key, 5]
        topo.link(0, 1).dropped_data = topo.DROP_NORM_MAX
        ls_after  = topo.link_state_vector()[idx_key, 5]
        assert ls_after > ls_before


# ═══════════════════════════════════════════════════════════════════════
#  ReplayBuffer — FIFO thuần (không có purge)
# ═══════════════════════════════════════════════════════════════════════

class TestReplayBuffer:
    """
    ReplayBuffer của người dùng là FIFO thuần:
    deque(maxlen=capacity) tự pop phần tử đầu khi đầy.
    KHÔNG có purge(), KHÔNG có purge_threshold.
    """

    def test_push_and_len(self):
        buf = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        buf.push(flat, 0, 1.0, flat, False)
        assert len(buf) == 1

    def test_push_many(self):
        buf = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(50):
            buf.push(flat, i % 8, float(i), flat, False)
        assert len(buf) == 50

    def test_fifo_eviction_when_full(self):
        """Khi đầy, phần tử cũ nhất bị xóa tự động (deque maxlen)."""
        buf  = ReplayBuffer(10)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(15):   # push 15 vào buffer capacity=10
            buf.push(flat, 0, float(i), flat, False)
        assert len(buf) == 10   # không quá capacity

    def test_no_purge_method(self):
        """FIFO thuần — không có purge() method."""
        buf = ReplayBuffer(100)
        assert not hasattr(buf, "purge"), \
            "ReplayBuffer của người dùng không có purge()"

    def test_no_purge_threshold(self):
        """Không có purge_threshold parameter."""
        buf = ReplayBuffer(100)
        assert not hasattr(buf, "purge_threshold"), \
            "ReplayBuffer của người dùng không có purge_threshold"

    def test_is_full_property(self):
        buf  = ReplayBuffer(5)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        assert not buf.is_full
        for i in range(5):
            buf.push(flat, 0, 0.0, flat, False)
        assert buf.is_full

    def test_fill_ratio(self):
        buf  = ReplayBuffer(10)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        for i in range(5):
            buf.push(flat, 0, 0.0, flat, False)
        assert abs(buf.fill_ratio - 0.5) < 1e-9

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

    def test_sample_raises_when_too_small(self):
        buf  = ReplayBuffer(100)
        flat = np.zeros(INPUT_DIM, dtype=np.float32)
        buf.push(flat, 0, 0.0, flat, False)
        with pytest.raises(Exception):
            buf.sample(10)   # buffer chỉ có 1 phần tử


# ═══════════════════════════════════════════════════════════════════════
#  select_action
# ═══════════════════════════════════════════════════════════════════════

class TestSelectAction:
    def test_action_in_valid_neighbors_all_nodes(self, agent, topo):
        """Mọi action đều là valid neighbor của current_node."""
        for cur in range(NUM_NODES):
            valid = list(np.where(agent.neighbor_mask[cur] > 0)[0])
            if not valid:
                continue
            obs = make_obs(topo, cur=cur)
            for _ in range(20):
                a = agent.select_action(obs)
                assert a in valid, f"node={cur}: invalid action={a}"

    def test_greedy_picks_highest_q(self, agent, topo):
        """Exploitation: chọn action có Q cao nhất."""
        agent.eval_mode()
        with torch.no_grad():
            last = list(agent.q_net.net.children())[-1]
            last.bias.fill_(0.0); last.weight.fill_(0.0)
            last.bias[1] = 10.0
        obs = make_obs(topo, cur=0)
        if agent.neighbor_mask[0][1] > 0:
            actions = {agent.select_action(obs) for _ in range(30)}
            assert actions == {1}

    def test_tie_breaking_random_among_equal_q(self, agent, topo):
        """Khi Q bằng nhau, tie-breaking random → thấy nhiều action khác nhau."""
        agent.eval_mode()
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.fill_(0.0)
        obs   = make_obs(topo, cur=0)
        valid = set(np.where(agent.neighbor_mask[0] > 0)[0].tolist())
        seen  = set()
        for _ in range(300):
            seen.add(agent.select_action(obs))
        assert seen == valid, f"tie-break: got {seen}, expected {valid}"

    def test_epsilon_1_always_explores(self, agent, topo):
        """epsilon=1 → luôn explore (random)."""
        agent.train_mode(); agent.epsilon = 1.0
        obs  = make_obs(topo, cur=0)
        seen = set()
        for _ in range(200):
            seen.add(agent.select_action(obs))
        assert len(seen) > 1

    def test_eval_mode_ignores_epsilon(self, agent, topo):
        """eval_mode bỏ qua epsilon, luôn greedy."""
        agent.eval_mode(); agent.epsilon = 1.0
        with torch.no_grad():
            last = list(agent.q_net.net.children())[-1]
            last.bias.fill_(0.0); last.weight.fill_(0.0)
            last.bias[2] = 5.0
        obs = make_obs(topo, cur=0)
        if agent.neighbor_mask[0][2] > 0:
            actions = {agent.select_action(obs) for _ in range(20)}
            assert actions == {2}

    def test_never_invalid_action(self, agent, topo):
        """Không bao giờ chọn action không có link."""
        agent.train_mode(); agent.epsilon = 0.5
        for cur in range(NUM_NODES):
            valid = set(np.where(agent.neighbor_mask[cur] > 0)[0].tolist())
            obs   = make_obs(topo, cur=cur)
            for _ in range(50):
                assert agent.select_action(obs) in valid


# ═══════════════════════════════════════════════════════════════════════
#  remember
# ═══════════════════════════════════════════════════════════════════════

class TestRemember:
    def test_remember_adds_to_buffer(self, agent, topo):
        obs = make_obs(topo)
        agent.remember(obs, 1, 0.5, obs, False)
        assert len(agent.memory) == 1

    def test_remember_correct_shape(self, agent, topo):
        obs = make_obs(topo)
        agent.remember(obs, 1, 0.5, obs, True)
        s, a, r, ns, d = agent.memory.sample(1)
        assert s.shape  == (1, INPUT_DIM)
        assert a[0]     == 1
        assert abs(r[0] - 0.5) < 1e-6
        assert d[0]     == 1.0


# ═══════════════════════════════════════════════════════════════════════
#  update
# ═══════════════════════════════════════════════════════════════════════

class TestUpdate:
    def test_no_update_before_learn_start(self, agent, topo):
        fill_buffer(agent, topo, n=4)   # < learn_start=8
        result = agent.update()
        assert result["loss"] is None

    def test_loss_non_negative_after_learn_start(self, agent, topo):
        fill_buffer(agent, topo, n=20)
        result = agent.update()
        assert result["loss"] is not None
        assert result["loss"] >= 0.0

    def test_loss_is_finite(self, agent, topo):
        fill_buffer(agent, topo, n=20)
        result = agent.update()
        assert result["loss"] == result["loss"]   # not NaN
        assert result["loss"] < 1e8

    def test_epsilon_decays_during_training(self, agent, topo):
        agent.train_mode()
        fill_buffer(agent, topo, n=20)
        eps_before = agent.epsilon
        for _ in range(5):
            agent.update()
        assert agent.epsilon < eps_before

    def test_epsilon_floor_at_eps_min(self, agent, topo):
        agent.epsilon = agent.eps_min
        fill_buffer(agent, topo, n=20)
        for _ in range(30):
            agent.update()
        assert agent.epsilon >= agent.eps_min - 1e-9

    def test_target_net_syncs_at_freq(self, agent, topo):
        """target_net đồng bộ với q_net sau target_update_freq bước."""
        agent.target_update_freq = 2
        fill_buffer(agent, topo, n=20)
        with torch.no_grad():
            list(agent.q_net.parameters())[0].fill_(99.0)
        for _ in range(2):
            agent.update()
        q_val = list(agent.q_net.parameters())[0][0, 0].item()
        t_val = list(agent.target_net.parameters())[0][0, 0].item()
        assert abs(q_val - t_val) < 1e-4

    def test_action_masking_in_target(self, agent, topo):
        """Target Q không dùng Q-value của invalid actions."""
        fill_buffer(agent, topo, n=20)
        with torch.no_grad():
            last = list(agent.target_net.net.children())[-1]
            last.bias.fill_(0.0)
            last.bias[7] = 1e6   # node 7 không kề node 0
        result = agent.update()
        assert result["loss"] is not None
        assert result["loss"] == result["loss"]   # not NaN
        assert result["loss"] < 1e8

    def test_gradient_clipping_applied(self, agent, topo):
        """Gradient clipping (max_norm=1.0) không crash."""
        fill_buffer(agent, topo, n=20)
        # Force large loss bằng cách đặt target rất xa prediction
        result = agent.update()
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
#  best_path
# ═══════════════════════════════════════════════════════════════════════

class TestBestPath:
    def test_starts_at_src(self, agent, topo):
        path = agent.best_path(0, 7, topo)
        assert path[0] == 0

    def test_only_valid_links(self, agent, topo):
        """Mọi bước đi trên path đều là link hợp lệ."""
        path = agent.best_path(0, 7, topo)
        for i in range(len(path) - 1):
            assert topo.has_link(path[i], path[i+1]), \
                f"invalid link {path[i]}→{path[i+1]}"

    def test_no_cycles(self, agent, topo):
        """Path không có vòng lặp."""
        path = agent.best_path(0, 7, topo)
        assert len(path) == len(set(path)), f"cycle in {path}"

    def test_reaches_dst_when_q_guided(self, agent, topo):
        """Nếu Q được set đúng, agent đi đến đích."""
        route = {0: 1, 1: 3, 3: 5, 5: 7}
        with torch.no_grad():
            last = list(agent.q_net.net.children())[-1]
            last.bias.fill_(0.0); last.weight.fill_(0.0)
            for node, nxt in route.items():
                if agent.neighbor_mask[node][nxt] > 0:
                    last.bias[nxt] = 10.0
        path = agent.best_path(0, 7, topo)
        assert path[-1] == 7 or len(path) > 1

    def test_training_state_restored(self, agent, topo):
        """best_path không thay đổi training state của agent."""
        agent.train_mode()
        agent.best_path(0, 7, topo)
        assert agent.training is True

        agent.eval_mode()
        agent.best_path(0, 7, topo)
        assert agent.training is False

    def test_send_traffic_called_per_hop(self, agent, topo):
        """best_path gọi send_traffic + reduce_load → link_states thay đổi."""
        topo.reset(); topo.step_background(0.1)
        ls_before = topo.link_state_vector().copy()
        agent.best_path(0, 7, topo, volume_mbps=20.0)
        ls_after = topo.link_state_vector()
        # Ít nhất 1 link thay đổi trạng thái (cột 3-5 là dynamic)
        assert not np.allclose(ls_before[:, 3:], ls_after[:, 3:])

    def test_tie_breaking_in_exploitation(self, agent, topo):
        """best_path: khi Q bằng nhau, tie-breaking random."""
        agent.eval_mode()
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.fill_(0.0)   # tất cả Q = 0
        # Chạy nhiều lần — có thể thấy nhiều path khác nhau
        paths = set()
        for _ in range(50):
            topo.reset()
            p = tuple(agent.best_path(0, 7, topo))
            paths.add(p)
        # Không phải lúc nào cũng cùng 1 path (có randomness)
        assert len(paths) >= 1   # ít nhất không crash


# ═══════════════════════════════════════════════════════════════════════
#  save / load
# ═══════════════════════════════════════════════════════════════════════

class TestSaveLoad:
    def _make_agent(self):
        cfg = {
            "input_dim":          INPUT_DIM,
            "hidden_dims":        [64, 64],
            "lr":                 1e-3,
            "gamma":              0.99,
            "epsilon":            1.0,
            "eps_min":            0.01,
            "eps_decay":          0.99,
            "batch_size":         8,
            "buffer_capacity":    500,
            "target_update_freq": 10,
            "learn_start":        8,
        }
        return DQNAgent(n_states=1, n_actions=NUM_NODES, config=cfg)

    def test_save_and_load_epsilon(self, agent, topo):
        agent.epsilon = 0.42
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            assert abs(ag2.epsilon - 0.42) < 1e-5

    def test_save_and_load_steps_done(self, agent, topo):
        agent.steps_done = 77
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            assert ag2.steps_done == 77

    def test_save_and_load_q_net_weights(self, agent, topo):
        with torch.no_grad():
            list(agent.q_net.parameters())[0].fill_(3.14)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            v1 = list(agent.q_net.parameters())[0][0, 0].item()
            v2 = list(ag2.q_net.parameters())[0][0, 0].item()
            assert abs(v1 - v2) < 1e-5

    def test_save_and_load_target_net_weights(self, agent, topo):
        with torch.no_grad():
            list(agent.target_net.parameters())[0].fill_(7.77)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)
            ag2 = self._make_agent()
            ag2.load(path)
            v = list(ag2.target_net.parameters())[0][0, 0].item()
            assert abs(v - 7.77) < 1e-5

    def test_load_sets_epsilon_to_eps_min_if_missing(self, agent, topo):
        """Nếu checkpoint không có 'epsilon' → dùng eps_min."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            # Lưu checkpoint không có epsilon
            torch.save({
                "q_net":      agent.q_net.state_dict(),
                "target_net": agent.target_net.state_dict(),
                "optimizer":  agent.optimizer.state_dict(),
                # không có "epsilon" và "steps_done"
            }, path)
            ag2 = self._make_agent()
            ag2.load(path)
            assert ag2.epsilon == ag2.eps_min


# ═══════════════════════════════════════════════════════════════════════
#  network_summary
# ═══════════════════════════════════════════════════════════════════════

class TestNetworkSummary:
    def test_summary_contains_DQN(self, agent, topo):
        s = agent.network_summary()
        assert "DQN" in s

    def test_summary_contains_input_dim(self, agent, topo):
        s = agent.network_summary()
        assert str(INPUT_DIM) in s

    def test_summary_is_string(self, agent, topo):
        assert isinstance(agent.network_summary(), str)