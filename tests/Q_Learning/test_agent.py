"""
tests/Q_Learning/test_agent.py
pytest tests/Q_Learning/test_agent.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import tempfile

from agents.Q_Learning.ql_agent_for_delay import QLearningAgent
from network.Q_Learning.topology import NetworkTopology


NUM_NODES = 8


@pytest.fixture
def agent():
    topo = NetworkTopology()
    ag   = QLearningAgent(config={
        "alpha": 0.1, "gamma": 0.99,
        "epsilon": 1.0, "eps_min": 0.05, "eps_decay": 0.99,
    })
    ag.set_neighbor_mask(topo.adj_matrix)
    return ag, topo


# ── Q-table init ──────────────────────────────────────────────────────

class TestQTableInit:
    def test_shape(self, agent):
        ag, _ = agent
        assert ag.Q.shape == (NUM_NODES, NUM_NODES, NUM_NODES)

    def test_initial_zeros(self, agent):
        ag, _ = agent
        assert np.all(ag.Q == 0.0)

    def test_neighbor_mask_set(self, agent):
        ag, topo = agent
        # Node 0 có neighbors 1 và 2
        valid = ag._valid_actions(0)
        assert 1 in valid and 2 in valid

    def test_no_self_loop_in_valid(self, agent):
        ag, _ = agent
        for node in range(NUM_NODES):
            valid = ag._valid_actions(node)
            assert node not in valid


# ── Action selection ──────────────────────────────────────────────────

class TestActionSelection:
    def test_returns_valid_neighbor(self, agent):
        ag, topo = agent
        ag.epsilon = 1.0     # full exploration
        for cur in range(NUM_NODES):
            valid = list(ag._valid_actions(cur))
            if not valid:
                continue
            dst  = (cur + 3) % NUM_NODES
            action = ag.select_action((cur, dst))
            assert action in valid, \
                f"Node {cur}: action {action} not in valid {valid}"

    def test_greedy_returns_best_q(self, agent):
        ag, _ = agent
        ag.epsilon = 0.0
        ag.training = False
        # Đặt Q cao cho action 1 từ node 0
        ag.Q[0, 7, 1] = 10.0
        action = ag.select_action((0, 7))
        assert action == 1

    def test_epsilon_1_always_random(self, agent):
        ag, _ = agent
        ag.epsilon = 1.0
        actions = set()
        for _ in range(200):
            a = ag.select_action((0, 7))
            actions.add(a)
        # Với 200 lần, phải thấy ít nhất 2 actions khác nhau
        assert len(actions) >= 2

    def test_eval_mode_greedy(self, agent):
        ag, _ = agent
        ag.eval_mode()
        ag.Q[0, 7, 2] = 5.0
        ag.Q[0, 7, 1] = 1.0
        action = ag.select_action((0, 7))
        assert action == 2


# ── Update (Bellman) ──────────────────────────────────────────────────

class TestBellmanUpdate:
    def test_q_changes_after_update(self, agent):
        ag, _ = agent
        q_before = ag.Q[0, 7, 1].copy()
        ag.update((0, 7), 1, reward=1.0, next_state=(1, 7), done=False)
        assert ag.Q[0, 7, 1] != q_before

    def test_terminal_update_ignores_next(self, agent):
        ag, _ = agent
        ag.Q[:] = 0.0
        ag.update((0, 7), 1, reward=1.0, next_state=(7, 7), done=True)
        # TD target = reward = 1.0
        expected = 0.0 + 0.1 * (1.0 - 0.0)
        assert abs(ag.Q[0, 7, 1] - expected) < 1e-9

    def test_epsilon_decays(self, agent):
        ag, _ = agent
        eps_before = ag.epsilon
        ag.update((0, 7), 1, 0.5, (1, 7), False)
        assert ag.epsilon < eps_before

    def test_epsilon_not_below_min(self, agent):
        ag, _ = agent
        ag.epsilon = ag.eps_min
        for _ in range(100):
            ag.update((0, 7), 1, 0.5, (1, 7), False)
        assert ag.epsilon >= ag.eps_min

    def test_td_error_returned(self, agent):
        ag, _ = agent
        result = ag.update((0, 7), 1, 1.0, (1, 7), False)
        assert "td_error" in result
        assert result["td_error"] >= 0.0

    def test_convergence_on_simple_mdp(self, agent):
        """
        Đơn giản: agent luôn ở node 0, dst=1, action 1 luôn cho reward +1.
        Q[0,1,1] phải hội tụ về ~ 1/(1-gamma) nếu không terminal,
        hoặc về 1.0 nếu terminal.
        Ở đây dùng terminal → Q[0,1,1] → 1.0
        """
        ag, _ = agent
        ag.epsilon = 0.0
        ag.alpha   = 0.5
        for _ in range(200):
            ag.update((0, 1), 1, reward=1.0, next_state=(1, 1), done=True)
        assert ag.Q[0, 1, 1] > 0.9


# ── best_path ─────────────────────────────────────────────────────────

class TestBestPath:
    def test_path_starts_at_src(self, agent):
        ag, _ = agent
        path = ag.best_path(0, 7)
        assert path[0] == 0

    def test_path_ends_at_dst_when_q_learned(self, agent):
        ag, topo = agent
        # Đặt Q-table đơn giản để agent đi theo shortest path 0→1→3→5→7
        route = {
            (0, 7): 1,
            (1, 7): 3,
            (3, 7): 5,
            (5, 7): 7,
        }
        for (cur, dst), nxt in route.items():
            ag.Q[cur, dst, nxt] = 10.0
        path = ag.best_path(0, 7)
        assert path[-1] == 7

    def test_path_uses_only_valid_links(self, agent):
        ag, topo = agent
        path = ag.best_path(0, 7)
        for i in range(len(path) - 1):
            assert topo.has_link(path[i], path[i+1]), \
                f"Invalid link {path[i]}→{path[i+1]} in path"


# ── Save / Load ───────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load_preserves_qtable(self, agent):
        ag, _ = agent
        ag.Q[0, 7, 1] = 3.14
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "qtable.npy")
            ag.save(path)
            ag2 = QLearningAgent()
            ag2.load(path)
            assert abs(ag2.Q[0, 7, 1] - 3.14) < 1e-6
            assert ag2.Q.shape == ag.Q.shape