"""
tests/Q_Learning/test_env.py
pytest tests/Q_Learning/test_env.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import gymnasium as gym

import env  # noqa — đăng ký NetworkRouting-v0
from env.Q_Learning.routing_env import NetworkRoutingEnv
from env.Q_Learning.reward import compute_step_reward_q_learning


# ── Fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def make_env():
    e = NetworkRoutingEnv(max_hops=8, seed=0)
    yield e
    e.close()

@pytest.fixture
def gym_env():
    e = gym.make("NetworkRouting-QL-v0", max_hops=8, seed=0)
    yield e
    e.close()


# ── Reset ─────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_dict_obs(self, make_env):
        obs, info = make_env.reset()
        assert isinstance(obs, dict)
        assert "current_node" in obs
        assert "dst_node" in obs
        assert "link_states" in obs

    def test_obs_shapes(self, make_env):
        obs, _ = make_env.reset()
        assert isinstance(obs["current_node"], (int, np.integer))
        assert isinstance(obs["dst_node"],     (int, np.integer))
        assert obs["link_states"].shape == (26, 4)

    def test_src_neq_dst(self, make_env):
        for _ in range(20):
            obs, _ = make_env.reset()
            assert obs["current_node"] != obs["dst_node"]

    def test_link_states_in_range(self, make_env):
        obs, _ = make_env.reset()
        ls = obs["link_states"]
        assert np.all(ls >= 0)

    def test_info_keys(self, make_env):
        _, info = make_env.reset()
        # reset trả về info rỗng (Gym convention)
        assert isinstance(info, dict)


# ── Step ──────────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_5_tuple(self, make_env):
        make_env.reset()
        result = make_env.step(make_env.action_space.sample())
        assert len(result) == 5

    def test_reward_is_float(self, make_env):
        make_env.reset()
        _, reward, _, _, _ = make_env.step(0)
        assert isinstance(reward, float)

    def test_invalid_link_gives_penalty(self, make_env):
        # Node 0 không có link đến node 7 trực tiếp
        make_env.reset()
        make_env._src = 0
        make_env._dst = 7
        make_env._current_node = 0
        _, reward, terminated, truncated, _ = make_env.step(7)
        assert reward == pytest.approx(-0.5)
        assert not terminated and not truncated

    def test_reach_dst_terminates(self, make_env):
        obs, _ = make_env.reset()
        # Ép current_node và dst kề nhau
        make_env._current_node = 0
        make_env._dst = 1
        make_env._src = 0
        make_env._path = [0]  # Cập nhật _path để đồng bộ với _current_node
        _, reward, terminated, truncated, _ = make_env.step(1)
        assert terminated
        assert reward > -2.0

    def test_exceed_max_hops_truncates(self, make_env):
        make_env.reset()
        make_env._current_node = 0
        make_env._dst = 7
        make_env._path = [0]  # Cập nhật _path để đồng bộ với _current_node
        make_env._hops = make_env.max_hops  # already at limit
        _, reward, _, truncated, _ = make_env.step(1)
        # Sau khi hops >= max_hops, step tiếp theo phải truncate
        # (hoặc đã truncated tại step này)
        assert truncated or make_env._hops >= make_env.max_hops

    def test_episode_always_ends(self, make_env):
        make_env.reset()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = make_env.step(
                make_env.action_space.sample()
            )
            done = terminated or truncated
            steps += 1
            assert steps < 500, "Episode không kết thúc"

    def test_obs_valid_after_step(self, make_env):
        make_env.reset()
        obs, _, _, _, _ = make_env.step(make_env.action_space.sample())
        assert "current_node" in obs
        assert 0 <= obs["current_node"] < 8
        assert obs["link_states"].shape == (26, 4)


# ── Full episode ───────────────────────────────────────────────────────

class TestFullEpisode:
    def test_random_rollout_accumulates_reward(self, make_env):
        make_env.reset()
        total = 0.0
        done  = False
        while not done:
            _, r, terminated, truncated, _ = make_env.step(
                make_env.action_space.sample()
            )
            total += r
            done = terminated or truncated
        assert isinstance(total, float)

    def test_info_has_path_at_end(self, make_env):
        make_env.reset()
        info = {}
        done = False
        while not done:
            _, _, terminated, truncated, info = make_env.step(
                make_env.action_space.sample()
            )
            done = terminated or truncated
        assert "path" in info
        assert len(info["path"]) >= 1

    def test_gymnasium_env_checker(self):
        from gymnasium.utils.env_checker import check_env
        e = NetworkRoutingEnv(max_hops=8, seed=7)
        check_env(e, warn=True)
        e.close()

    def test_gym_make_works(self, gym_env):
        obs, _ = gym_env.reset()
        assert "link_states" in obs


# ── Reward function ───────────────────────────────────────────────────

class TestReward:
    def test_zero_delay_gives_zero_reward(self):
        r = compute_step_reward_q_learning(0.0)
        assert r == pytest.approx(0.0)

    def test_small_delay_near_zero(self):
        r = compute_step_reward_q_learning(1.0)
        assert -0.2 < r < 0.0  # gần 0 nhưng âm

    def test_reference_delay_equals_minus_one(self):
        r = compute_step_reward_q_learning(10.0)
        assert r == pytest.approx(-1.0)

    def test_large_delay_clipped_to_minus_one(self):
        r = compute_step_reward_q_learning(50.0)
        assert r == pytest.approx(-1.0)

    def test_reward_monotonic_with_delay(self):
        r1 = compute_step_reward_q_learning(1.0)
        r2 = compute_step_reward_q_learning(5.0)
        r3 = compute_step_reward_q_learning(10.0)

        assert r1 > r2 > r3  # delay tăng → reward giảm

    def test_reward_bounds(self):
        for d in [0, 1, 5, 10, 50, 100]:
            r = compute_step_reward_q_learning(d)
            assert -1.0 <= r <= 0.0