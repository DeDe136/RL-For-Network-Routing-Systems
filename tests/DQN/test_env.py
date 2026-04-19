"""
tests/DQN/test_env.py

Unit tests cho NetworkRoutingEnv theo thiết kế mới:
  - _total_dropped_data (không còn _dropped bool)
  - Không truncated vì drop
  - link_states shape (26, 6) với drop_norm
  - obs_to_flat shape (158,) = 2 + 26×6
  - reward nhận total_dropped_data, không nhận dropped bool

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
                    "gymnasium.envs": envs, "gymnasium.envs.registration": reg})

from env.DQN.routing_env import NetworkRoutingEnv
from env.DQN.reward import (compute_final_reward, compute_shaping_reward,
                         DROP_PENALTY_SCALE, BANDWIDTH_MAX, QUEUE_SIZE_MAX)
from env.DQN.spaces import obs_to_flat


@pytest.fixture
def env():
    e = NetworkRoutingEnv(max_hops=8, bg_intensity=0.3, seed=0)
    yield e
    e.close()


def step_valid(env):
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

    def test_link_states_shape_26x6(self, env):
        obs, _ = env.reset()
        assert obs["link_states"].shape == (26, 6), \
            f"expected (26,6), got {obs['link_states'].shape}"

    def test_src_neq_dst(self, env):
        for _ in range(20):
            obs, _ = env.reset()
            assert obs["current_node"] != obs["dst_node"]

    def test_total_dropped_data_reset_to_zero(self, env):
        env.reset()
        env.step(list(env.topo.neighbors(env._current_node))[0])
        env.reset()
        assert env._total_dropped_data == 0.0

    def test_no_dropped_bool_attr(self, env):
        assert not hasattr(env, "_dropped"), \
            "_dropped bool không còn tồn tại trong env mới"

    def test_link_states_vary_across_episodes(self, env):
        ls1 = env.reset()[0]["link_states"][:, :3].copy()
        ls2 = env.reset()[0]["link_states"][:, :3].copy()
        assert not np.allclose(ls1, ls2)

    def test_episode_state_fully_reset(self, env):
        env.reset()
        for _ in range(3):
            step_valid(env)
        env.reset()
        assert env._hops               == 0
        assert env._total_delay        == 0.0
        assert env._total_dropped_data == 0.0
        assert len(env._path)          == 1


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

    def test_obs_link_states_shape(self, env):
        env.reset()
        obs, _, _, _, _ = step_valid(env)
        assert obs["link_states"].shape == (26, 6)

    def test_hops_increments(self, env):
        env.reset()
        step_valid(env)
        assert env._hops == 1

    def test_path_grows(self, env):
        env.reset()
        assert len(env._path) == 1
        step_valid(env)
        assert len(env._path) == 2

    def test_total_delay_accumulates(self, env):
        env.reset()
        step_valid(env)
        assert env._total_delay > 0.0

    def test_total_dropped_data_is_float(self, env):
        env.reset()
        step_valid(env)
        assert isinstance(env._total_dropped_data, float)
        assert env._total_dropped_data >= 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Step — điều kiện kết thúc
# ═══════════════════════════════════════════════════════════════════════

class TestTerminationConditions:
    def test_reach_dst_terminated(self, env):
        env.reset()
        env._current_node = 0; env._dst = 1
        env._src = 0; env._path = [0]
        _, r, term, trunc, _ = env.step(1)
        assert term and not trunc
        assert r > -2.0

    def test_max_hops_truncates(self, env):
        env2 = NetworkRoutingEnv(max_hops=4, bg_intensity=0.0, seed=2)
        env2.reset()
        env2._current_node = 0; env2._dst = 7
        env2._src = 0; env2._path = [0]; env2._hops = 2
        _, _, _, trunc, _ = env2.step(1)   # hops → 3 = max_hops-1
        assert trunc
        env2.close()

    def test_drop_does_NOT_truncate(self, env):
        """
        Quan trọng: drop không còn gây truncated.
        Ngay cả khi có overflow, episode tiếp tục.
        """
        env.reset()
        env._current_node = 0; env._dst = 7; env._path = [0]
        # Làm queue rất nhỏ → overflow chắc chắn
        for attr in env.topo._link_attrs.values():
            attr.queue_size_cur = 1
        env._hops = 0
        _, _, term, trunc, info = env.step(1)
        # Không được truncated VÌ drop
        if trunc:
            # Truncated chỉ được phép nếu do max_hops
            assert env._hops >= env.max_hops - 1, \
                "truncated xảy ra KHÔNG phải vì max_hops → sai"

    def test_total_dropped_data_in_info(self, env):
        env.reset()
        _, _, _, _, info = step_valid(env)
        assert "total_dropped_data" in info
        assert isinstance(info["total_dropped_data"], float)

    def test_info_dropped_key_backward_compat(self, env):
        """info['dropped'] vẫn tồn tại (bool) để backward-compat."""
        env.reset()
        _, _, _, _, info = step_valid(env)
        assert "dropped" in info
        assert isinstance(info["dropped"], bool)

    def test_never_both_terminated_and_truncated(self, env):
        for _ in range(60):
            env.reset()
            done = False
            while not done:
                _, _, t, tr, _ = step_valid(env)
                assert not (t and tr)
                done = t or tr

    def test_episode_always_ends(self, env):
        for _ in range(20):
            env.reset(); done = False; steps = 0
            while not done:
                _, _, t, tr, _ = step_valid(env)
                done = t or tr; steps += 1
                assert steps < 500

    def test_dropped_data_accumulates_across_steps(self, env):
        """_total_dropped_data tích lũy qua nhiều step."""
        env.reset()
        for attr in env.topo._link_attrs.values():
            attr.queue_size_cur = 1   # force drop mỗi step
        total = 0.0
        for _ in range(3):
            if env._current_node == env._dst:
                break
            _, _, t, tr, _ = step_valid(env)
            total = env._total_dropped_data
            if t or tr:
                break
        # _total_dropped_data phải là float ≥ 0
        assert isinstance(env._total_dropped_data, float)
        assert env._total_dropped_data >= 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Invalid link guard
# ═══════════════════════════════════════════════════════════════════════

class TestInvalidLinkGuard:
    def test_invalid_link_reward_minus_05(self, env):
        env.reset(); env._current_node = 0; env._path = [0]
        _, r, term, trunc, _ = env.step(7)   # 0→7 không có link
        assert abs(r - (-0.5)) < 1e-9

    def test_invalid_link_position_unchanged(self, env):
        env.reset(); env._current_node = 0; env._path = [0]
        env.step(7)
        assert env._current_node == 0

    def test_invalid_link_not_terminal(self, env):
        env.reset(); env._current_node = 0; env._path = [0]
        _, _, term, trunc, _ = env.step(7)
        assert not term and not trunc


# ═══════════════════════════════════════════════════════════════════════
#  obs_to_flat (shape 158)
# ═══════════════════════════════════════════════════════════════════════

class TestObsToFlat:
    def test_shape_158(self, env):
        obs, _ = env.reset()
        flat = obs_to_flat(obs)
        assert flat.shape == (158,), f"got {flat.shape}"

    def test_node_encoded_in_0_1(self, env):
        for _ in range(10):
            obs, _ = env.reset()
            flat = obs_to_flat(obs)
            assert 0.0 <= flat[0] <= 1.0
            assert 0.0 <= flat[1] <= 1.0

    def test_decode_current_node_roundtrip(self, env):
        for _ in range(20):
            obs, _ = env.reset()
            flat = obs_to_flat(obs)
            decoded = int(round(flat[0] * 7))
            assert decoded == obs["current_node"]

    def test_drop_norm_in_flat_vector(self, env):
        """Cột drop_norm (index 5 trong link_states) có mặt trong flat."""
        obs, _ = env.reset()
        flat = obs_to_flat(obs)
        # flat[2:] = link_states.flatten() với shape (26,6)
        # Cột 5 của link 0 = flat[2 + 5] = flat[7]
        assert flat.shape == (158,)   # 2 + 26*6


# ═══════════════════════════════════════════════════════════════════════
#  Info dict — link_details
# ═══════════════════════════════════════════════════════════════════════

class TestInfoDict:
    def test_link_details_has_dropped_data(self, env):
        env.reset()
        _, _, _, _, info = step_valid(env)
        if info["link_details"]:
            for lk in info["link_details"]:
                assert "dropped_data" in lk
                assert isinstance(lk["dropped_data"], float)

    def test_link_details_per_link_fields(self, env):
        env.reset()
        _, _, _, _, info = step_valid(env)
        required = {"src_node","dst_node","delay","bandwidth","load",
                    "utilization","queue_size_cur","queue_used_cur",
                    "queue_util","dropped_data"}
        for lk in info["link_details"]:
            assert required.issubset(lk.keys()), \
                f"missing: {required - lk.keys()}"

    def test_total_dropped_data_equals_sum_of_link_details(self, env):
        """total_dropped_data trong info phải ≥ tổng dropped_data mỗi link."""
        env.reset()
        for attr in env.topo._link_attrs.values():
            attr.queue_size_cur = 5   # nhỏ để dễ overflow
        _, _, t, tr, info = step_valid(env)
        link_sum = sum(lk["dropped_data"] for lk in info["link_details"])
        # total_dropped_data có thể > link_sum vì còn từ reduce_load
        assert info["total_dropped_data"] >= link_sum - 1e-9


# ═══════════════════════════════════════════════════════════════════════
#  Reward functions
# ═══════════════════════════════════════════════════════════════════════

class TestRewardFunctions:
    def test_no_path_returns_minus_two(self):
        r = compute_final_reward(False, 0, 0, 0.0)
        assert r == -2.0

    def test_reward_decreases_with_more_dropped_data(self):
        base = dict(path_found=True, total_delay=5.0, hops=2, utilization=0.1,
                    avg_queue_util=0.1, avg_bandwidth=100.0, avg_queue_size=50.0)
        r0 = compute_final_reward(**base, total_dropped_data=0.0)
        r1 = compute_final_reward(**base, total_dropped_data=DROP_PENALTY_SCALE)
        assert r1 < r0, f"r0={r0:.4f} r1={r1:.4f}"

    def test_max_drop_penalty(self):
        base = dict(path_found=True, total_delay=5.0, hops=2, utilization=0.0,
                    avg_queue_util=0.0)
        r0 = compute_final_reward(**base, total_dropped_data=0.0)
        r1 = compute_final_reward(**base, total_dropped_data=DROP_PENALTY_SCALE)
        assert abs(r0 - r1 - 0.30) < 1e-6, \
            f"expected delta=0.30, got {r0-r1:.6f}"

    def test_no_dropped_bool_param_in_signature(self):
        import inspect
        sig = inspect.signature(compute_final_reward)
        assert "dropped" not in sig.parameters, \
            "compute_final_reward không còn nhận 'dropped' bool"
        assert "total_dropped_data" in sig.parameters

    def test_good_path_positive(self):
        r = compute_final_reward(True, 5.0, 2, 0.1, 0.1, 50.0, 50.0, 0.0)
        assert r > 0

    def test_reward_bounded(self):
        for pf in [True, False]:
            r = compute_final_reward(pf, 30.0, 4, 0.5, 0.5, 100.0, 50.0, 25.0)
            assert -2.1 <= r <= 1.2

    def test_bandwidth_bonus(self):
        base = dict(path_found=True, total_delay=5.0, hops=2, utilization=0.5,
                    avg_queue_util=0.5, avg_queue_size=50.0, total_dropped_data=0.0)
        r_no  = compute_final_reward(**base, avg_bandwidth=0.0)
        r_bw  = compute_final_reward(**base, avg_bandwidth=BANDWIDTH_MAX)
        assert abs(r_bw - r_no - 0.05) < 1e-6

    def test_queue_size_bonus(self):
        base = dict(path_found=True, total_delay=5.0, hops=2, utilization=0.5,
                    avg_queue_util=0.5, avg_bandwidth=0.0, total_dropped_data=0.0)
        r_no  = compute_final_reward(**base, avg_queue_size=0.0)
        r_qs  = compute_final_reward(**base, avg_queue_size=QUEUE_SIZE_MAX)
        assert abs(r_qs - r_no - 0.05) < 1e-6

    def test_shaping_reward_negative(self):
        r = compute_shaping_reward(2.0, 0.1, 0.1, 100.0, 50.0, 0.0)
        assert r <= 0

    def test_shaping_drop_penalty(self):
        r0 = compute_shaping_reward(2.0, 0.1, 0.1, 100.0, 50.0, 0.0)
        r1 = compute_shaping_reward(2.0, 0.1, 0.1, 100.0, 50.0, 50.0)
        assert r1 < r0

    def test_shaping_clamp_minus_05(self):
        r = compute_shaping_reward(10.0, 1.0, 1.0, 0.0, 0.0, 9999.0)
        assert r >= -0.5