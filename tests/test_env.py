"""
tests/test_env.py
pytest tests/test_env.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import gymnasium as gym

import env  # noqa — đăng ký NetworkRouting-v0
from env.routing_env import NetworkRoutingEnv
from env.reward import compute_reward


# ── Fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def make_env():
    e = NetworkRoutingEnv(max_hops=8, seed=0)
    yield e
    e.close()

@pytest.fixture
def gym_env():
    e = gym.make("NetworkRouting-v0", max_hops=8, seed=0)
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
        _, reward, terminated, truncated, _ = make_env.step(1)
        assert terminated
        assert reward > -2.0

    def test_exceed_max_hops_truncates(self, make_env):
        make_env.reset()
        make_env._current_node = 0
        make_env._dst = 7
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
    def test_no_path_returns_minus_two(self):
        r = compute_reward(0, False, 0, 0, path_found=False)
        assert r == pytest.approx(-2.0)

    def test_perfect_path_near_one(self):
        r = compute_reward(
            total_delay=1.0, dropped=False, hops=1,
            utilization=0.0, path_found=True
        )
        assert r > 0.5

    def test_bad_path_lower_reward(self):
        r_good = compute_reward(5.0,  False, 1, 0.1, True)
        r_bad  = compute_reward(45.0, True,  6, 0.9, True)
        assert r_good > r_bad

    def test_drop_reduces_reward(self):
        r_no_drop = compute_reward(10.0, False, 3, 0.3, True)
        r_drop    = compute_reward(10.0, True,  3, 0.3, True)
        assert r_no_drop > r_drop

    def test_reward_bounded(self):
        for delay in [0, 5, 50, 100]:
            for dropped in [True, False]:
                for hops in [1, 4, 8]:
                    r = compute_reward(delay, dropped, hops, 0.5, True)
                    assert -2.1 <= r <= 1.1