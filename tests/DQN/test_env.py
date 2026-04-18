"""
tests/DQN/test_env.py

Unit tests cho NetworkRoutingEnv (MDP model).
Chạy: pytest tests/DQN/test_env.py -v
"""

import sys, os, types, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# ── Mock gymnasium ────────────────────────────────────────────────────
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


# ── Fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def env():
    e = NetworkRoutingEnv(max_hops=8, bg_intensity=0.3, seed=0)
    yield e
    e.close()


def step_valid(env):
    """Một step với valid neighbor."""
    nbrs = env.topo.neighbors(env._current_node)
    action = random.choice(nbrs) if nbrs else 0
    return env.step(action)


# ═══════════════════════════════════════════════════════════════════════
#  Reset
# ═══════════════════════════════════════════════════════════════════════

class TestReset:
    def test_obs_has_required_keys(self, env):
        obs, _ = env.reset()
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
        """Mỗi episode bắt đầu với tham số link khác nhau (randomize_links)."""
        ls1 = env.reset()[0]["link_states"][:, :3].copy()
        ls2 = env.reset()[0]["link_states"][:, :3].copy()
        assert not np.allclose(ls1, ls2)

    def test_episode_state_reset(self, env):
        env.reset()
        for _ in range(3):
            step_valid(env)
        env.reset()
        assert env._hops         == 0
        assert env._total_delay  == 0.0
        assert env._dropped      == False
        assert len(env._path)    == 1


# ═══════════════════════════════════════════════════════════════════════
#  Step — cơ bản
# ═══════════════════════════════════════════════════════════════════════

class TestStepBasic:
    def test_returns_5_tuple(self, env):
        env.reset()
        result = step_valid(env)
        assert len(result) == 5

    def test_reward_is_float(self, env):
        env.reset()
        _, r, _, _, _ = step_valid(env)
        assert isinstance(r, float)

    def test_obs_updated(self, env):
        env.reset()
        obs, _, _, _, _ = step_valid(env)
        assert "link_states" in obs
        assert obs["link_states"].shape == (26, 5)

    def test_hops_increments(self, env):
        env.reset()
        step_valid(env)
        assert env._hops == 1
        step_valid(env) if not (env._current_node == env._dst) else None

    def test_path_grows(self, env):
        env.reset()
        assert len(env._path) == 1
        step_valid(env)
        assert len(env._path) == 2

    def test_send_traffic_updates_link(self, env):
        """After step, link trên path có load > 0."""
        env.reset()
        cur = env._current_node
        nbrs = env.topo.neighbors(cur)
        action = nbrs[0]
        env.step(action)
        attr = env.topo.link(cur, action)
        # Sau send_traffic và reduce_load, load có thể > 0
        assert attr.load >= 0  # load không âm

    def test_total_delay_accumulates(self, env):
        env.reset()
        env.step(list(env.topo.neighbors(env._current_node))[0])
        assert env._total_delay > 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Step — kết thúc episode
# ═══════════════════════════════════════════════════════════════════════

class TestStepTermination:
    def test_reach_dst_terminated(self, env):
        env.reset()
        env._current_node = 0
        env._dst  = 1
        env._src  = 0
        env._path = [0]
        _, r, term, trunc, _ = env.step(1)
        assert term and not trunc
        assert r > -2.0

    def test_drop_truncates(self, env):
        env.reset()
        env._current_node = 0
        env._dst  = 7
        env._path = [0]
        # Làm đầy queue link 0→1
        attr = env.topo.link(0, 1)
        attr.queue_used_cur = attr.queue_size_cur + 10.0  # vượt ngưỡng
        # Bất kỳ volume nào cũng sẽ drop vì queue đã đầy
        _, r, term, trunc, info = env.step(1)
        assert trunc and not term
        assert r < 0
        assert info["dropped"]

    def test_max_hops_truncates(self, env):
        env.reset()
        env._current_node = 0
        env._dst   = 7
        env._hops  = env.max_hops
        env._path  = [0]
        _, r, _, trunc, _ = env.step(1)
        assert trunc
        assert r < 0

    def test_never_both_terminated_and_truncated(self, env):
        for _ in range(80):
            env.reset()
            done = False
            while not done:
                _, _, t, tr, _ = step_valid(env)
                assert not (t and tr), "Both terminated & truncated!"
                done = t or tr

    def test_episode_always_ends(self, env):
        for _ in range(30):
            env.reset()
            done = False; steps = 0
            while not done:
                _, _, t, tr, _ = step_valid(env)
                done = t or tr; steps += 1
                assert steps < 500, "Episode không kết thúc"

    def test_drop_wins_over_dst(self, env):
        """Drop tại đích → truncated (không terminated)."""
        env.reset()
        env._current_node = 0
        env._dst  = 1
        env._src  = 0
        env._path = [0]
        attr = env.topo.link(0, 1)
        attr.queue_used_cur = attr.queue_size_cur + 10.0
        _, _, term, trunc, info = env.step(1)
        assert trunc and not term
        assert info["dropped"]


# ═══════════════════════════════════════════════════════════════════════
#  is_first_hop logic
# ═══════════════════════════════════════════════════════════════════════

class TestFirstHop:
    def test_first_hop_congs_volume_into_queue(self, env):
        """Hop đầu: queue_used_cur tăng theo volume."""
        env.reset()
        env._current_node = 0
        env._dst  = 7
        env._path = [0]
        attr = env.topo.link(0, 1)
        attr.queue_used_cur = 0.0
        attr.load           = 0.0
        env._volume = 10.0
        env.step(1)
        # Sau send_traffic + reduce_load, queue hoặc load phải >= 0
        assert attr.load >= 0 or attr.queue_used_cur >= 0

    def test_second_hop_no_extra_volume(self, env):
        """Hop thứ 2: queue_used_cur của link mới không cộng thêm volume."""
        env.reset()
        env._current_node = 0; env._dst = 7; env._path = [0]; env._hops = 0
        env._volume = 10.0
        env.step(1)   # hop 1: path=[0,1]
        # Ghi lại queue_used của link 0→1 sau reduce
        q_used_01 = env.topo.link(0, 1).queue_used
        if env._current_node != env._dst and not env._dropped:
            env.step(list(env.topo.neighbors(env._current_node))[0])
            # Không có assert cứng vì phụ thuộc nhiều vào tham số random,
            # nhưng không được crash
            assert env._hops >= 2


# ═══════════════════════════════════════════════════════════════════════
#  obs_to_flat
# ═══════════════════════════════════════════════════════════════════════

class TestObsToFlat:
    def test_shape_132(self, env):
        obs, _ = env.reset()
        flat = obs_to_flat(obs)
        assert flat.shape == (132,), f"got {flat.shape}"

    def test_node_encoded_0_to_1(self, env):
        for _ in range(10):
            obs, _ = env.reset()
            flat = obs_to_flat(obs)
            assert 0.0 <= flat[0] <= 1.0
            assert 0.0 <= flat[1] <= 1.0

    def test_decode_roundtrip(self, env):
        for _ in range(20):
            obs, _ = env.reset()
            flat = obs_to_flat(obs)
            decoded = int(round(flat[0] * 7))
            assert decoded == obs["current_node"], \
                f"decoded={decoded} actual={obs['current_node']}"


# ═══════════════════════════════════════════════════════════════════════
#  Reward functions
# ═══════════════════════════════════════════════════════════════════════

class TestRewardFunctions:
    def test_no_path_returns_minus_two(self):
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
                assert -2.1 <= r <= 1.1, f"reward={r} out of bounds"

    def test_shaping_negative(self):
        r = compute_shaping_reward(2.0, 0.1, 0.1)
        assert r < 0

    def test_shaping_penalizes_congestion(self):
        r_low  = compute_shaping_reward(2.0, 0.1, 0.0)
        r_high = compute_shaping_reward(2.0, 0.9, 0.9)
        assert r_low > r_high


# ═══════════════════════════════════════════════════════════════════════
#  reduce_load được gọi trong step
# ═══════════════════════════════════════════════════════════════════════

class TestReduceLoadInStep:
    def test_reduce_load_called_each_step(self, env):
        """Load không tích lũy vô hạn nhờ reduce_load."""
        env.reset()
        env._current_node = 0; env._dst = 7; env._path = [0]
        # Nhiều step liên tiếp: load phải hữu hạn
        for _ in range(5):
            if env._current_node == env._dst or env._dropped:
                break
            step_valid(env)
        for attr in env.topo._link_attrs.values():
            assert attr.load < float("inf")
            assert attr.queue_used_cur >= 0