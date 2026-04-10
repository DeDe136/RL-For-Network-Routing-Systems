"""
tests/DQN/test_env.py
pytest tests/DQN/test_env.py -v
"""

import sys, os, types, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

gm = types.ModuleType("gymnasium"); sp = types.ModuleType("gymnasium.spaces")
class _Env:
    def reset(self, seed=None, options=None): pass
class _Disc:
    def __init__(self, n): self.n = n
    def sample(self): return random.randint(0, self.n-1)
sp.Box = lambda **kw: None; sp.Discrete = _Disc; sp.Dict = lambda d: None
gm.spaces = sp; gm.Env = _Env
reg = types.ModuleType("gymnasium.envs.registration"); reg.register = lambda **k: None
envs = types.ModuleType("gymnasium.envs"); envs.registration = reg
sys.modules.update({"gymnasium": gm, "gymnasium.spaces": sp,
                    "gymnasium.envs": envs,
                    "gymnasium.envs.registration": reg})

from env.DQN.routing_env import NetworkRoutingEnv
from env.DQN.reward import compute_final_reward, compute_shaping_reward
from env.DQN.spaces import obs_to_flat


@pytest.fixture
def env():
    e = NetworkRoutingEnv(max_hops=8, bg_intensity=0.3, seed=0)
    yield e
    e.close()


class TestReset:
    def test_obs_keys(self, env):
        obs, info = env.reset()
        for k in ("current_node", "dst_node", "link_states"):
            assert k in obs

    def test_link_states_shape(self, env):
        obs, _ = env.reset()
        assert obs["link_states"].shape == (26, 5)

    def test_src_neq_dst(self, env):
        for _ in range(20):
            obs, _ = env.reset()
            assert obs["current_node"] != obs["dst_node"]

    def test_link_states_vary_across_episodes(self, env):
        """randomize_links → tham số vật lý thay đổi mỗi episode."""
        ls1 = env.reset()[0]["link_states"][:, :3].copy()
        ls2 = env.reset()[0]["link_states"][:, :3].copy()
        assert not np.allclose(ls1, ls2)


class TestStep:
    def test_step_returns_5_tuple(self, env):
        env.reset()
        result = env.step(list(env.topo.neighbors(env._current_node))[0])
        assert len(result) == 5

    def test_reward_is_float(self, env):
        env.reset()
        nbr = list(env.topo.neighbors(env._current_node))[0]
        _, reward, _, _, _ = env.step(nbr)
        assert isinstance(reward, float)

    def test_reach_dst_terminates(self, env):
        env.reset()
        env._current_node = 0; env._dst = 1
        env._src = 0; env._path = [0]
        _, reward, term, trunc, _ = env.step(1)
        assert term and not trunc
        assert reward > -2.0

    def test_drop_truncates(self, env):
        env.reset(); env._current_node = 0; env._dst = 7; env._path = [0]
        lk = env.topo.link(0, 1)
        lk.load = lk.bandwidth; lk.queue_used = lk.queue_size
        _, reward, term, trunc, info = env.step(1)
        assert trunc and not term
        assert reward < 0
        assert info["dropped"]

    def test_max_hops_truncates(self, env):
        env.reset(); env._current_node = 0; env._dst = 7
        env._hops = env.max_hops; env._path = [0]
        _, reward, _, trunc, _ = env.step(1)
        assert trunc
        assert reward < 0

    def test_never_both_terminated_and_truncated(self, env):
        for _ in range(80):
            env.reset(); done = False
            while not done:
                nbrs = env.topo.neighbors(env._current_node)
                action = random.choice(nbrs) if nbrs else 0
                _, _, t, tr, _ = env.step(action)
                assert not (t and tr), "terminated and truncated both True"
                done = t or tr

    def test_episode_always_ends(self, env):
        env.reset(); done = False; steps = 0
        while not done:
            nbrs = env.topo.neighbors(env._current_node)
            action = random.choice(nbrs) if nbrs else 0
            _, _, t, tr, _ = env.step(action)
            done = t or tr; steps += 1
            assert steps < 500, "episode did not end"


class TestObsToFlat:
    def test_flat_shape_132(self, env):
        obs, _ = env.reset()
        flat = obs_to_flat(obs)
        assert flat.shape == (132,), f"got {flat.shape}"

    def test_decode_node_roundtrip(self, env):
        for _ in range(20):
            obs, _ = env.reset()
            flat = obs_to_flat(obs)
            decoded = int(round(flat[0] * 7))
            assert decoded == obs["current_node"]


class TestReward:
    def test_no_path_minus_two(self):
        r = compute_final_reward(False, 0, False, 0, 0.0, 0.0)
        assert r == -2.0

    def test_good_path_positive(self):
        r = compute_final_reward(True, 5.0, False, 2, 0.1, 0.1)
        assert r > 0

    def test_bad_path_lower_reward(self):
        r_good = compute_final_reward(True, 5.0,  False, 2, 0.1, 0.1)
        r_bad  = compute_final_reward(True, 45.0, False, 6, 0.9, 0.9)
        assert r_good > r_bad

    def test_reward_bounded(self):
        for pf in [True, False]:
            for h in [1, 4, 8]:
                r = compute_final_reward(pf, 30.0, False, h, 0.5, 0.5)
                assert -2.1 <= r <= 1.1

    def test_shaping_penalizes_high_queue(self):
        r_low  = compute_shaping_reward(2.0, 0.1, 0.0)
        r_high = compute_shaping_reward(2.0, 0.1, 0.9)
        assert r_high < r_low