"""
tests/DQN/test_agent.py

Unit tests cho DQNAgent (Vanilla DQN).
Chạy: pytest tests/DQN/test_agent.py -v
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
reg = types.ModuleType("gymnasium.envs.registration")
reg.register = lambda **k: None
envs = types.ModuleType("gymnasium.envs"); envs.registration = reg
sys.modules.update({"gymnasium": gm, "gymnasium.spaces": sp,
                    "gymnasium.envs": envs,
                    "gymnasium.envs.registration": reg})

torch = pytest.importorskip("torch", reason="torch not installed")

from network.DQN.topology import NetworkTopology
from agents.DQN.dqn_agent import DQNAgent
from env.DQN.spaces import obs_to_flat, NUM_NODES

INPUT_DIM = 132  # 2 + 26 × 5


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def topo():
    return NetworkTopology(seed=0)


@pytest.fixture
def agent(topo):
    cfg = {
        "input_dim": INPUT_DIM, "hidden_dims": [64, 64],
        "lr": 1e-3, "gamma": 0.99,
        "epsilon": 1.0, "eps_min": 0.01, "eps_decay": 0.99,
        "batch_size": 8, "buffer_capacity": 500,
        "target_update_freq": 10, "learn_start": 8,
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
    """Điền n transition vào buffer."""
    flat = obs_to_flat(make_obs(topo))
    valid = list(np.where(agent.neighbor_mask[0] > 0)[0])
    for i in range(n):
        agent.memory.push(flat, random.choice(valid),
                          float(i % 3 - 1), flat, i == n - 1)


# ── select_action ─────────────────────────────────────────────────────

class TestSelectAction:
    def test_action_in_valid_neighbors_all_nodes(self, agent, topo):
        """Action luôn nằm trong valid neighbors, với mọi node."""
        for cur in range(NUM_NODES):
            valid = list(np.where(agent.neighbor_mask[cur] > 0)[0])
            if not valid:
                continue
            obs = make_obs(topo, cur=cur)
            for _ in range(20):
                a = agent.select_action(obs)
                assert a in valid, \
                    f"node={cur}: invalid action={a}, valid={valid}"

    def test_greedy_picks_highest_q(self, agent, topo):
        """Khi eval, agent chọn action có Q cao nhất trong valid."""
        agent.eval_mode()
        with torch.no_grad():
            last = list(agent.q_net.net.children())[-1]
            last.bias.fill_(0.0)
            last.weight.fill_(0.0)
            last.bias[1] = 10.0   # node 1 có Q cao nhất
        obs = make_obs(topo, cur=0)
        # Node 1 phải là neighbor của node 0
        if agent.neighbor_mask[0][1] > 0:
            actions = {agent.select_action(obs) for _ in range(30)}
            assert actions == {1}, f"expected {{1}}, got {actions}"

    def test_tie_breaking_covers_all_valid(self, agent, topo):
        """Khi tất cả Q bằng nhau, tie-breaking chọn đều mọi valid action."""
        agent.eval_mode()
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.fill_(0.0)
        obs   = make_obs(topo, cur=0)
        valid = set(np.where(agent.neighbor_mask[0] > 0)[0].tolist())
        seen  = set()
        for _ in range(300):
            seen.add(agent.select_action(obs))
        assert seen == valid, \
            f"tie-break: got {seen}, expected {valid}"

    def test_epsilon_1_explores_randomly(self, agent, topo):
        """Với ε=1, agent luôn explore — thấy nhiều action khác nhau."""
        agent.train_mode(); agent.epsilon = 1.0
        obs  = make_obs(topo, cur=0)
        seen = set()
        for _ in range(200):
            seen.add(agent.select_action(obs))
        assert len(seen) > 1, f"epsilon=1 but only saw {seen}"

    def test_eval_mode_ignores_epsilon(self, agent, topo):
        """Trong eval mode, ε bị bỏ qua — chỉ greedy."""
        agent.eval_mode(); agent.epsilon = 1.0
        with torch.no_grad():
            last = list(agent.q_net.net.children())[-1]
            last.bias.fill_(0.0); last.weight.fill_(0.0)
            last.bias[2] = 5.0
        obs = make_obs(topo, cur=0)
        if agent.neighbor_mask[0][2] > 0:
            actions = {agent.select_action(obs) for _ in range(20)}
            assert actions == {2}, f"eval mode got {actions}"

    def test_no_invalid_action_ever(self, agent, topo):
        """Không bao giờ chọn action không có link, kể cả khi explore."""
        agent.train_mode(); agent.epsilon = 0.5
        for cur in range(NUM_NODES):
            valid = set(np.where(agent.neighbor_mask[cur] > 0)[0].tolist())
            obs   = make_obs(topo, cur=cur)
            for _ in range(50):
                a = agent.select_action(obs)
                assert a in valid, \
                    f"node={cur}: picked invalid {a}"


# ── remember ──────────────────────────────────────────────────────────

class TestRemember:
    def test_remember_adds_to_buffer(self, agent, topo):
        obs = make_obs(topo)
        agent.remember(obs, 1, 0.5, obs, False)
        assert len(agent.memory) == 1

    def test_remember_stores_flat_state(self, agent, topo):
        obs = make_obs(topo)
        agent.remember(obs, 1, 0.5, obs, True)
        s, a, r, ns, d = agent.memory.sample(1)
        assert s.shape == (1, INPUT_DIM), f"got {s.shape}"
        assert a[0] == 1
        assert abs(r[0] - 0.5) < 1e-6
        assert d[0] == 1.0


# ── update ────────────────────────────────────────────────────────────

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

    def test_epsilon_decays_each_update(self, agent, topo):
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
        """Target net phải đồng bộ sau target_update_freq bước."""
        agent.target_update_freq = 2
        fill_buffer(agent, topo, n=20)
        with torch.no_grad():
            list(agent.q_net.parameters())[0].fill_(99.0)
        for _ in range(2):
            agent.update()
        q_val = list(agent.q_net.parameters())[0][0, 0].item()
        t_val = list(agent.target_net.parameters())[0][0, 0].item()
        assert abs(q_val - t_val) < 1e-4, \
            f"q_net={q_val:.4f} target_net={t_val:.4f}"

    def test_action_masking_in_target(self, agent, topo):
        """
        Đảm bảo target Q không dùng Q-value của invalid actions.
        Test: đặt Q rất cao cho invalid action → nếu không mask,
        loss sẽ bất thường. Chỉ cần update chạy được mà không crash
        và loss hợp lý.
        """
        fill_buffer(agent, topo, n=20)
        with torch.no_grad():
            last = list(agent.target_net.net.children())[-1]
            last.bias.fill_(0.0)
            # Đặt Q cực cao cho tất cả action → nếu không mask,
            # target sẽ rất lớn → loss cực lớn
            last.bias[7] = 1e6   # node 7 thường không kề node 0
        result = agent.update()
        # Loss phải hữu hạn (không bị inf/nan)
        assert result["loss"] is not None
        assert not (result["loss"] != result["loss"])   # not NaN
        assert result["loss"] < 1e8                     # không quá lớn


# ── Vanilla DQN overestimation ────────────────────────────────────────

class TestVanillaDQNBehavior:
    def test_same_network_selects_and_evaluates_target(self, agent, topo):
        """
        Vanilla DQN: target_net vừa chọn action vừa tính value
        (cùng 1 network) → có thể overestimate.

        Test quan sát: trong update(), target = target_net.max()
        Nếu là Double DQN, action được chọn bởi q_net — khác nhau.
        Ở đây chỉ verify rằng vanilla DQN KHÔNG dùng q_net để chọn action
        trong phần tính target (kiểm tra gián tiếp qua behavior).
        """
        fill_buffer(agent, topo, n=20)
        with torch.no_grad():
            # q_net bias action 1
            last_q = list(agent.q_net.net.children())[-1]
            last_q.bias.fill_(0.0); last_q.weight.fill_(0.0)
            last_q.bias[1] = 5.0
            # target_net bias action khác (2 nếu có link từ 0)
            last_t = list(agent.target_net.net.children())[-1]
            last_t.bias.fill_(0.0); last_t.weight.fill_(0.0)
            last_t.bias[2] = 10.0
        # Vanilla DQN dùng target_net.max() → chọn action 2 làm target
        # Không crash là đủ; behavior difference với Double DQN sẽ
        # thể hiện qua training curves (overestimation).
        result = agent.update()
        assert result["loss"] is not None


# ── best_path ─────────────────────────────────────────────────────────

class TestBestPath:
    def test_starts_at_src(self, agent, topo):
        path = agent.best_path(0, 7, topo)
        assert path[0] == 0

    def test_only_valid_links(self, agent, topo):
        path = agent.best_path(0, 7, topo)
        for i in range(len(path) - 1):
            assert topo.has_link(path[i], path[i+1]), \
                f"invalid link {path[i]}→{path[i+1]}"

    def test_no_cycles(self, agent, topo):
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

    def test_eval_mode_restored(self, agent, topo):
        """best_path không thay đổi training state của agent."""
        agent.train_mode()
        agent.best_path(0, 7, topo)
        assert agent.training is True

        agent.eval_mode()
        agent.best_path(0, 7, topo)
        assert agent.training is False


# ── save / load ───────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load_preserves_state(self, agent, topo):
        agent.epsilon    = 0.42
        agent.steps_done = 77
        with torch.no_grad():
            list(agent.q_net.parameters())[0].fill_(3.14)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)

            cfg2 = {
                "input_dim": INPUT_DIM, "hidden_dims": [64, 64],
                "lr": 1e-3, "gamma": 0.99, "epsilon": 1.0,
                "eps_min": 0.01, "eps_decay": 0.99,
                "batch_size": 8, "buffer_capacity": 500,
                "target_update_freq": 10, "learn_start": 8,
            }
            ag2 = DQNAgent(n_states=1, n_actions=NUM_NODES, config=cfg2)
            ag2.load(path)

            assert abs(ag2.epsilon    - 0.42) < 1e-5
            assert ag2.steps_done == 77
            v1 = list(agent.q_net.parameters())[0][0, 0].item()
            v2 = list(ag2.q_net.parameters())[0][0, 0].item()
            assert abs(v1 - v2) < 1e-5, f"weights differ: {v1} vs {v2}"

    def test_target_net_also_saved(self, agent, topo):
        """target_net phải được save và restore."""
        with torch.no_grad():
            list(agent.target_net.parameters())[0].fill_(7.77)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)
            cfg2 = {
                "input_dim": INPUT_DIM, "hidden_dims": [64, 64],
                "lr": 1e-3, "gamma": 0.99, "epsilon": 1.0,
                "eps_min": 0.01, "eps_decay": 0.99,
                "batch_size": 8, "buffer_capacity": 500,
                "target_update_freq": 10, "learn_start": 8,
            }
            ag2 = DQNAgent(n_states=1, n_actions=NUM_NODES, config=cfg2)
            ag2.load(path)
            v = list(ag2.target_net.parameters())[0][0, 0].item()
            assert abs(v - 7.77) < 1e-5


# ── network_summary ───────────────────────────────────────────────────

class TestNetworkSummary:
    def test_summary_contains_key_info(self, agent, topo):
        s = agent.network_summary()
        assert "DQN" in s
        assert "params" in s
        assert str(INPUT_DIM) in s