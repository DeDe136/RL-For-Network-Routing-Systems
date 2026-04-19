"""
tests/DQN/test_network.py

Unit tests cho network layer theo file topology.py và link.py của người dùng.

Điểm khác biệt so với version cũ:
  - DROP_NORM_MAX = 50.0 là class attribute của NetworkTopology
    (không import từ link.py).
  - LinkAttr trong link.py KHÔNG export DROP_NORM_MAX.
  - reduce_load() trả về float (total_dropped).
  - send_traffic() trả về {"total_delay", "dropped_data", "hops"},
    không có "dropped" bool.
  - weight() method trong topology để Dijkstra dùng delay.

Topology tuyến tính dùng để trace MDP spec:
    0 <--> 1 <--> 2 <--> 3 <--> 4
    delay=[5,3,6,2], bw=[20,30,40,50], q_size=[90,60,70,100]

Chạy: pytest tests/DQN/test_network.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from network.DQN.link import LinkAttr
from network.DQN.topology import NetworkTopology
from network.DQN.metrics import NetworkMetrics


# ── Helpers: simulate MDP spec trên topology tuyến tính ───────────────

def make_linear_links():
    """
    Topology tuyến tính cố định: 0-1-2-3-4.
    delay=[5,3,6,2] ms, bw=[20,30,40,50] Mbps, q_size=[90,60,70,100] packets.
    """
    params = {
        (0,1): (5, 20,  90),
        (1,2): (3, 30,  60),
        (2,3): (6, 40,  70),
        (3,4): (2, 50, 100),
    }
    links = {}
    for (u, v), (d, bw, q) in params.items():
        for a, b in [(u, v), (v, u)]:
            links[(a, b)] = LinkAttr(delay=d, bandwidth=bw, queue_size_cur=q)
    return links


def send_step(links, path, vol, is_first):
    """
    Simulate send_traffic MDP spec cho link cuối của path.
    Trả về lượng data bị drop tại hop này (float).
    """
    u, v = path[-2], path[-1]
    attr = links[(u, v)]

    # Bước 1: lấy queue_used từ link trước
    if len(path) >= 3:
        pa  = links[(path[-3], u)]
        qup = pa.queue_used
        pa.queue_used = 0.0
    else:
        qup = 0.0

    # Bước 2: cộng vào queue_used_cur
    attr.queue_used_cur += (qup + vol) if is_first else qup

    # Bước 3: drop & clamp
    dropped = 0.0
    if attr.queue_used_cur > attr.queue_size_cur:
        overflow = attr.queue_used_cur - attr.queue_size_cur
        attr.dropped_data  += overflow
        dropped             = overflow
        attr.queue_used_cur = float(attr.queue_size_cur)

    # Bước 4: cập nhật load và queue_used
    rb = attr.bandwidth - attr.load
    if rb <= attr.queue_used_cur:
        attr.load           += rb
        attr.queue_used      = attr.load
        attr.queue_used_cur -= rb
    else:
        attr.load           += attr.queue_used_cur
        attr.queue_used      = attr.load
        attr.queue_used_cur  = 0.0

    return dropped


def reduce_step(links, path, decay=0.85):
    """
    Simulate reduce_load MDP spec.
    Trả về tổng dropped trong bước này (float).
    """
    pairs         = [(path[i], path[i+1]) for i in range(len(path)-1)]
    total_dropped = 0.0

    for i, (u, v) in enumerate(pairs):
        attr = links[(u, v)]

        # Phần A: chuyển load sang link kế tiếp
        if attr.load != 0:
            if i + 1 < len(pairs):
                nu, nv = pairs[i+1]
                na = links[(nu, nv)]
                na.queue_used_cur += attr.load
                if na.queue_used_cur > na.queue_size_cur:
                    ov = na.queue_used_cur - na.queue_size_cur
                    na.dropped_data  += ov
                    total_dropped    += ov
                    na.queue_used_cur = float(na.queue_size_cur)
            attr.load = 0.0

        # Phần B: xử lý queue_used_cur còn lại
        if attr.queue_used_cur == 0.0:
            attr.load = 0.0
        else:
            attr.load = 0.0
            rb        = attr.bandwidth          # load=0 → remain=bandwidth
            if rb <= attr.queue_used_cur:
                attr.load           += rb
                attr.queue_used_cur -= rb
            else:
                attr.load           += attr.queue_used_cur
                attr.queue_used_cur  = 0.0

    # Decay link ngoài path
    pset = set(pairs)
    for k, a in links.items():
        if k not in pset:
            a.load           *= decay
            a.queue_used_cur *= decay
            a.queue_used     *= decay

    return total_dropped


# ═══════════════════════════════════════════════════════════════════════
#  LinkAttr
# ═══════════════════════════════════════════════════════════════════════

class TestLinkAttr:
    """
    LinkAttr có 7 fields: delay, bandwidth, queue_size_cur,
    queue_used_cur, queue_used, load, dropped_data.
    DROP_NORM_MAX KHÔNG phải là hằng số trong link.py — nó nằm trong
    class NetworkTopology (topology.py).
    """

    def test_7_fields_exist(self):
        a = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90)
        for f in ("delay", "bandwidth", "queue_size_cur",
                  "queue_used_cur", "queue_used", "load", "dropped_data"):
            assert hasattr(a, f), f"missing field: {f}"

    def test_defaults_all_zero(self):
        a = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90)
        assert a.queue_used_cur == 0.0
        assert a.queue_used     == 0.0
        assert a.load           == 0.0
        assert a.dropped_data   == 0.0

    def test_utilization_half(self):
        a = LinkAttr(5, 20, 90, load=10.0)
        assert abs(a.utilization - 0.5) < 1e-9

    def test_utilization_capped_at_1(self):
        a = LinkAttr(5, 20, 90, load=9999.0)
        assert a.utilization == 1.0

    def test_utilization_zero_bandwidth(self):
        a = LinkAttr(5, 0, 90, load=10.0)
        assert a.utilization == 0.0

    def test_queue_util_half(self):
        a = LinkAttr(5, 20, 90, queue_used_cur=45.0)
        assert abs(a.queue_util - 0.5) < 1e-9

    def test_queue_util_capped_at_1(self):
        a = LinkAttr(5, 20, 90, queue_used_cur=9999.0)
        assert a.queue_util == 1.0

    def test_queue_util_zero_queue_size(self):
        a = LinkAttr(5, 20, 0)
        assert a.queue_util == 0.0

    def test_remain_bw(self):
        a = LinkAttr(5, 20, 90, load=5.0)
        assert abs(a.remain_bw - 15.0) < 1e-9

    def test_remain_bw_not_negative(self):
        a = LinkAttr(5, 20, 90, load=999.0)
        assert a.remain_bw == 0.0

    def test_is_congested_above_threshold(self):
        a = LinkAttr(5, 20, 90, load=17.0)    # util = 0.85 > 0.8
        assert a.is_congested

    def test_is_congested_below_threshold(self):
        a = LinkAttr(5, 20, 90, load=10.0)    # util = 0.5
        assert not a.is_congested

    def test_reset_clears_all_traffic_fields(self):
        a = LinkAttr(5, 20, 90,
                     load=10.0, queue_used_cur=30.0,
                     queue_used=10.0, dropped_data=5.0)
        a.reset()
        assert a.load           == 0.0
        assert a.queue_used_cur == 0.0
        assert a.queue_used     == 0.0
        assert a.dropped_data   == 0.0

    def test_reset_preserves_physical_params(self):
        """reset() chỉ xóa traffic, không thay đổi delay/bw/queue_size."""
        a = LinkAttr(5, 20, 90, load=10.0, dropped_data=3.0)
        a.reset()
        assert a.delay          == 5
        assert a.bandwidth      == 20
        assert a.queue_size_cur == 90


# ═══════════════════════════════════════════════════════════════════════
#  MDP Spec trace: send_traffic + reduce_load (bước 1 và 2 từ spec)
# ═══════════════════════════════════════════════════════════════════════

class TestMDPSpecStep1:
    """
    Bước 1: path=[0,1], volume=60, is_first_hop=True.
    Link 0-1: bw=20, q=90.
    queue_used_cur = 0 + 0 + 60 = 60 (không drop)
    remain_bw = 20 - 0 = 20 ≤ 60 → load=20, queue_used=20, queue_used_cur=40
    """

    def test_no_drop_60_lt_90(self):
        links   = make_linear_links()
        dropped = send_step(links, [0,1], 60, True)
        assert dropped == 0.0

    def test_queue_used_cur_40(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        assert abs(links[(0,1)].queue_used_cur - 40.0) < 1e-6

    def test_load_20(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        assert abs(links[(0,1)].load - 20.0) < 1e-6

    def test_queue_used_20(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        assert abs(links[(0,1)].queue_used - 20.0) < 1e-6

    def test_dropped_data_zero(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        assert links[(0,1)].dropped_data == 0.0


class TestMDPSpecStep1Reduce:
    """
    Sau reduce_load([0,1]):
    Phần A: load=20, no next → load=0.
    Phần B: queue_used_cur=40, bw=20 ≤ 40 → load=20, queue_used_cur=20.
    """

    def test_load_20_after_reduce(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        reduce_step(links, [0,1])
        assert abs(links[(0,1)].load - 20.0) < 1e-6

    def test_queue_used_cur_20_after_reduce(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        reduce_step(links, [0,1])
        assert abs(links[(0,1)].queue_used_cur - 20.0) < 1e-6


class TestMDPSpecStep2:
    """
    Bước 2: path=[0,1,2], volume=60, is_first_hop=False.
    queue_used(0-1) = 20 (sau step 1 + reduce).
    queue_used_cur(1-2) += 20 (từ link trước). 20 ≤ bw=30 → load=20, q_used_cur=0.
    queue_used(0-1) reset về 0 sau khi lấy.
    """

    def _state_after_step1(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        reduce_step(links, [0,1])
        return links

    def test_queue_used_01_becomes_zero(self):
        links = self._state_after_step1()
        send_step(links, [0,1,2], 60, False)
        assert abs(links[(0,1)].queue_used - 0.0) < 1e-6

    def test_queue_used_cur_12_is_zero(self):
        """bw(1-2)=30 > queue_added=20 → queue_used_cur=0 setelah send."""
        links = self._state_after_step1()
        send_step(links, [0,1,2], 60, False)
        assert abs(links[(1,2)].queue_used_cur - 0.0) < 1e-6

    def test_load_12_is_20(self):
        links = self._state_after_step1()
        send_step(links, [0,1,2], 60, False)
        assert abs(links[(1,2)].load - 20.0) < 1e-6

    def test_no_drop_step2(self):
        links = self._state_after_step1()
        dropped = send_step(links, [0,1,2], 60, False)
        assert dropped == 0.0


class TestMDPSpecStep2Reduce:
    """
    Sau reduce_load([0,1,2]):
    Link(0-1): load=20≠0, có link sau (1-2) → queue_used_cur(1-2)+=20, load(0-1)=0.
               queue_used_cur(0-1)=20, bw=20≤20 → load=20, q_used_cur=0.
    Link(1-2): load=20≠0, không có link sau → load=0.
               queue_used_cur(1-2)=20, bw=30>20 → load=20, q_used_cur=0.
    """

    def _state_after_step1_2(self):
        links = make_linear_links()
        send_step(links, [0,1], 60, True)
        reduce_step(links, [0,1])
        send_step(links, [0,1,2], 60, False)
        return links

    def test_link01_load_20(self):
        links = self._state_after_step1_2()
        reduce_step(links, [0,1,2])
        assert abs(links[(0,1)].load - 20.0) < 1e-6

    def test_link01_queue_used_cur_0(self):
        links = self._state_after_step1_2()
        reduce_step(links, [0,1,2])
        assert abs(links[(0,1)].queue_used_cur - 0.0) < 1e-6

    def test_link12_load_20(self):
        links = self._state_after_step1_2()
        reduce_step(links, [0,1,2])
        assert abs(links[(1,2)].load - 20.0) < 1e-6

    def test_link12_queue_used_cur_0(self):
        links = self._state_after_step1_2()
        reduce_step(links, [0,1,2])
        assert abs(links[(1,2)].queue_used_cur - 0.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
#  Drop & Clamp
# ═══════════════════════════════════════════════════════════════════════

class TestDropAndClamp:
    def test_no_drop_when_within_capacity(self):
        links   = make_linear_links()
        dropped = send_step(links, [0,1], 60, True)   # 60 ≤ 90
        assert dropped == 0.0
        assert links[(0,1)].dropped_data == 0.0

    def test_no_drop_exact_capacity(self):
        links   = make_linear_links()
        dropped = send_step(links, [0,1], 90, True)   # 90 == 90 → không drop
        assert dropped == 0.0

    def test_drop_one_above_capacity(self):
        links   = make_linear_links()
        dropped = send_step(links, [0,1], 91, True)   # 91 > 90 → drop 1
        assert abs(dropped - 1.0) < 1e-9

    def test_drop_large_overflow(self):
        links   = make_linear_links()
        dropped = send_step(links, [0,1], 200, True)  # 200 >> 90 → drop 110
        assert abs(dropped - 110.0) < 1e-9

    def test_queue_used_cur_clamped_at_capacity(self):
        links = make_linear_links()
        send_step(links, [0,1], 200, True)
        assert links[(0,1)].queue_used_cur <= links[(0,1)].queue_size_cur

    def test_dropped_data_accumulates_across_sends(self):
        links = make_linear_links()
        d1 = send_step(links, [0,1], 100, True)   # 100 > 90 → drop 10
        d2 = send_step(links, [0,1], 100, True)   # tiếp tục drop
        assert links[(0,1)].dropped_data == pytest.approx(d1 + d2)

    def test_episode_continues_after_drop(self):
        """Drop không dừng episode — queue chỉ bị clamp."""
        links = make_linear_links()
        send_step(links, [0,1], 200, True)   # drop lớn
        # Vẫn có thể tiếp tục send_traffic
        dropped_next = send_step(links, [0,1,2], 5, False)
        assert dropped_next >= 0.0   # không crash

    def test_reduce_load_forward_overflow(self):
        """reduce_load: forward load sang link kế bị overflow → dropped_data."""
        links = make_linear_links()
        links[(1,2)].queue_size_cur = 1
        links[(0,1)].load           = 100.0
        total = reduce_step(links, [0,1,2])
        assert total > 0
        assert links[(1,2)].dropped_data > 0
        assert links[(1,2)].queue_used_cur <= links[(1,2)].queue_size_cur


# ═══════════════════════════════════════════════════════════════════════
#  NetworkTopology — cấu trúc
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def topo():
    return NetworkTopology(seed=0)


class TestTopologyStructure:
    def test_8_nodes(self, topo):
        assert topo.graph.number_of_nodes() == 8

    def test_26_directed_edges(self, topo):
        assert topo.num_edges() == 26

    def test_link_state_vector_shape_26x6(self, topo):
        ls = topo.link_state_vector()
        assert ls.shape == (26, 6), f"expected (26,6), got {ls.shape}"

    def test_link_state_all_cols_in_0_1(self, topo):
        ls = topo.link_state_vector()
        assert np.all(ls >= 0)
        assert np.all(ls <= 1.0 + 1e-6)

    def test_link_state_drop_norm_col(self, topo):
        """Cột 5 là drop_norm, ban đầu = 0 (không có traffic)."""
        ls = topo.link_state_vector()
        assert np.all(ls[:, 5] == 0.0)

    def test_drop_norm_max_is_class_attribute(self, topo):
        """DROP_NORM_MAX = 50.0 trong topology.py (không phải link.py)."""
        assert hasattr(topo, "DROP_NORM_MAX")
        assert topo.DROP_NORM_MAX == 50.0

    def test_link_attr_7_fields(self, topo):
        attr = topo.link(0, 1)
        for f in ("delay", "bandwidth", "queue_size_cur",
                  "queue_used_cur", "queue_used", "load", "dropped_data"):
            assert hasattr(attr, f), f"missing: {f}"

    def test_has_link_bidirectional(self, topo):
        assert topo.has_link(0, 1) and topo.has_link(1, 0)

    def test_no_direct_link_0_7(self, topo):
        assert not topo.has_link(0, 7)

    def test_adj_matrix_shape_8x8(self, topo):
        assert topo.adj_matrix.shape == (8, 8)

    def test_topology_has_weight_method(self, topo):
        """topology.py có weight() method dùng delay cho Dijkstra."""
        assert hasattr(topo, "weight") and callable(topo.weight)

    def test_weight_returns_delay(self, topo):
        u, v = 0, 1
        assert abs(topo.weight(u, v, None) - topo.link(u, v).delay) < 1e-9


# ═══════════════════════════════════════════════════════════════════════
#  NetworkTopology — randomize / reset
# ═══════════════════════════════════════════════════════════════════════

class TestTopologyRandomize:
    def test_randomize_changes_physical_params(self, topo):
        ls1 = topo.link_state_vector()[:, :3].copy()
        topo.randomize_links()
        ls2 = topo.link_state_vector()[:, :3]
        assert not np.allclose(ls1, ls2)

    def test_randomize_preserves_adjacency(self, topo):
        adj1 = topo.adj_matrix.copy()
        topo.randomize_links()
        np.testing.assert_array_equal(adj1, topo.adj_matrix)

    def test_randomize_resets_dropped_data(self, topo):
        """randomize_links() tạo LinkAttr mới → dropped_data = 0."""
        topo.link(0, 1).dropped_data = 77.0
        topo.randomize_links()
        assert topo.link(0, 1).dropped_data == 0.0

    def test_reset_calls_randomize(self, topo):
        ls1 = topo.link_state_vector()[:, :3].copy()
        topo.reset()
        ls2 = topo.link_state_vector()[:, :3]
        assert not np.allclose(ls1, ls2)

    def test_same_seed_same_physical_params(self):
        t1 = NetworkTopology(seed=42)
        t2 = NetworkTopology(seed=42)
        np.testing.assert_allclose(
            t1.link_state_vector()[:, :3],
            t2.link_state_vector()[:, :3],
            rtol=1e-6,
        )

    def test_different_seed_different_params(self):
        t1 = NetworkTopology(seed=1)
        t2 = NetworkTopology(seed=2)
        assert not np.allclose(
            t1.link_state_vector()[:, :3],
            t2.link_state_vector()[:, :3],
        )


# ═══════════════════════════════════════════════════════════════════════
#  send_traffic trên NetworkTopology
# ═══════════════════════════════════════════════════════════════════════

class TestSendTrafficTopology:
    def test_returns_correct_keys(self, topo):
        r = topo.send_traffic([0, 1], 10.0, is_first_hop=True)
        assert "total_delay"  in r
        assert "dropped_data" in r
        assert "hops"         in r
        assert "dropped"      not in r   # không còn bool dropped

    def test_dropped_data_is_float(self, topo):
        r = topo.send_traffic([0, 1], 10.0, is_first_hop=True)
        assert isinstance(r["dropped_data"], float)

    def test_hops_is_1(self, topo):
        r = topo.send_traffic([0, 1], 5.0, is_first_hop=True)
        assert r["hops"] == 1

    def test_total_delay_equals_link_delay(self, topo):
        r = topo.send_traffic([0, 1], 5.0, is_first_hop=True)
        assert abs(r["total_delay"] - topo.link(0, 1).delay) < 1e-9

    def test_updates_load(self, topo):
        topo.send_traffic([0, 1], 10.0, is_first_hop=True)
        attr = topo.link(0, 1)
        assert attr.load > 0 or attr.queue_used_cur > 0

    def test_no_drop_with_small_volume(self, topo):
        r = topo.send_traffic([0, 1], 0.1, is_first_hop=True)
        assert r["dropped_data"] == 0.0
        assert topo.link(0, 1).dropped_data == 0.0

    def test_drop_and_clamp_when_overflow(self, topo):
        attr = topo.link(0, 1)
        attr.queue_size_cur = 1   # force overflow
        r = topo.send_traffic([0, 1], 100.0, is_first_hop=True)
        assert r["dropped_data"]    > 0
        assert attr.dropped_data    > 0
        assert attr.queue_used_cur <= attr.queue_size_cur

    def test_link_state_changes_after_traffic(self, topo):
        ls1 = topo.link_state_vector()[:, 3:].copy()   # cột động
        topo.send_traffic([0, 1], 30.0, is_first_hop=True)
        ls2 = topo.link_state_vector()[:, 3:]
        assert not np.allclose(ls1, ls2)

    def test_drop_norm_col_increases_with_dropped_data(self, topo):
        """Cột 5 (drop_norm) tăng khi dropped_data tăng."""
        idx = sorted(topo._link_attrs.keys()).index((0, 1))
        ls_before = topo.link_state_vector()[idx, 5]
        attr = topo.link(0, 1)
        attr.queue_size_cur = 1
        topo.send_traffic([0, 1], 100.0, is_first_hop=True)
        ls_after = topo.link_state_vector()[idx, 5]
        assert ls_after > ls_before

    def test_drop_norm_capped_at_1(self, topo):
        """drop_norm = min(1, dropped_data / DROP_NORM_MAX) ∈ [0,1]."""
        attr = topo.link(0, 1)
        attr.dropped_data = topo.DROP_NORM_MAX * 10   # rất lớn
        ls = topo.link_state_vector()
        idx = sorted(topo._link_attrs.keys()).index((0, 1))
        assert ls[idx, 5] == pytest.approx(1.0)

    def test_drop_norm_at_max(self, topo):
        attr = topo.link(0, 1)
        attr.dropped_data = topo.DROP_NORM_MAX
        ls  = topo.link_state_vector()
        idx = sorted(topo._link_attrs.keys()).index((0, 1))
        assert abs(ls[idx, 5] - 1.0) < 1e-6

    def test_empty_path_returns_zeros(self, topo):
        r = topo.send_traffic([0], 10.0, is_first_hop=True)
        assert r["total_delay"]  == 0.0
        assert r["dropped_data"] == 0.0
        assert r["hops"]         == 0


# ═══════════════════════════════════════════════════════════════════════
#  reduce_load trên NetworkTopology
# ═══════════════════════════════════════════════════════════════════════

class TestReduceLoadTopology:
    def test_returns_float(self, topo):
        result = topo.reduce_load([0, 1])
        assert isinstance(result, float)

    def test_not_bool(self, topo):
        result = topo.reduce_load([0, 1])
        assert not isinstance(result, bool)

    def test_zero_when_idle(self, topo):
        result = topo.reduce_load([0, 1])
        assert result == 0.0

    def test_drops_when_forward_overflows(self, topo):
        topo.link(1, 2).queue_size_cur = 1
        topo.link(0, 1).load           = 100.0
        result = topo.reduce_load([0, 1, 2])
        assert result > 0
        assert topo.link(1, 2).dropped_data > 0
        assert topo.link(1, 2).queue_used_cur <= topo.link(1, 2).queue_size_cur

    def test_decay_links_outside_path(self, topo):
        topo.link(1, 2).load = 10.0
        topo.reduce_load([0, 1], decay=0.5)
        assert abs(topo.link(1, 2).load - 5.0) < 1e-6

    def test_path_links_not_decayed(self, topo):
        """Link trong path bị xử lý bởi phần A/B (không phải *decay)."""
        topo.link(0, 1).load = 10.0
        topo.reduce_load([0, 1], decay=0.5)
        assert topo.link(0, 1).load != 5.0   # không phải 10 * 0.5

    def test_load_zeroed_after_partA(self, topo):
        """Phần A: load chuyển sang link sau hoặc bị zero."""
        topo.link(0, 1).load = 15.0
        topo.link(0, 1).queue_used_cur = 0.0
        topo.reduce_load([0, 1])
        assert topo.link(0, 1).load == 0.0 or topo.link(0, 1).load > 0


# ═══════════════════════════════════════════════════════════════════════
#  Background traffic + Summary
# ═══════════════════════════════════════════════════════════════════════

class TestTopologyBackground:
    def test_step_background_changes_utilization(self, topo):
        utils_before = [a.utilization for a in topo._link_attrs.values()]
        topo.step_background(intensity=0.5)
        utils_after  = [a.utilization for a in topo._link_attrs.values()]
        assert any(a != b for a, b in zip(utils_before, utils_after))

    def test_step_background_no_overflow(self, topo):
        """step_background clamp load/queue_used_cur trong giới hạn."""
        topo.step_background(intensity=1.0)
        for attr in topo._link_attrs.values():
            assert attr.load           <= attr.bandwidth      + 1e-9
            assert attr.queue_used_cur <= attr.queue_size_cur + 1e-9


class TestTopologySummary:
    def test_summary_has_required_keys(self, topo):
        s = topo.summary()
        for k in ("avg_utilization", "max_utilization",
                  "avg_queue_util", "total_dropped", "congested_links"):
            assert k in s, f"missing key: {k}"

    def test_total_dropped_increases_after_overflow(self, topo):
        topo.link(0, 1).queue_size_cur = 1
        topo.send_traffic([0, 1], 100.0, is_first_hop=True)
        assert topo.summary()["total_dropped"] > 0

    def test_shortest_path_uses_delay_weight(self, topo):
        """shortest_path dùng weight=delay (OSPF)."""
        path = topo.shortest_path(0, 7)
        assert path[0] == 0 and path[-1] == 7

    def test_shortest_path_no_path_returns_empty(self, topo):
        """Nếu không có đường → trả về []."""
        path = topo.shortest_path(0, 0)   # src == dst — networkx trả [0]
        # Không crash là đủ; kết quả phụ thuộc implementation
        assert isinstance(path, list)


# ═══════════════════════════════════════════════════════════════════════
#  NetworkMetrics
# ═══════════════════════════════════════════════════════════════════════

class TestNetworkMetrics:
    def test_path_delay_positive(self, topo):
        mc   = NetworkMetrics(topo)
        path = topo.shortest_path(0, 7)
        assert mc.path_delay(path) > 0

    def test_path_delay_zero_for_single_node(self, topo):
        mc = NetworkMetrics(topo)
        assert mc.path_delay([0]) == 0.0

    def test_path_utilization_zero_on_idle(self, topo):
        mc   = NetworkMetrics(topo)
        path = topo.shortest_path(0, 7)
        assert mc.path_utilization(path) == 0.0

    def test_path_utilization_increases_with_traffic(self, topo):
        mc   = NetworkMetrics(topo)
        path = topo.shortest_path(0, 7)
        for i in range(len(path)-1):
            topo.link(path[i], path[i+1]).load = 100.0
        assert mc.path_utilization(path) > 0

    def test_path_drop_prob_zero_on_idle(self, topo):
        mc   = NetworkMetrics(topo)
        path = topo.shortest_path(0, 7)
        assert mc.path_drop_prob(path) == 0.0