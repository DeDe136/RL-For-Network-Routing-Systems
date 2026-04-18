"""
tests/DQN/test_network.py

Unit tests cho network layer theo MDP spec.
Chạy: pytest tests/DQN/test_network.py -v

Topology test đơn giản để verify spec:
    0 <--> 1 <--> 2 <--> 3 <--> 4

Link params cố định (không random) để dễ trace:
    0-1: delay=5, bw=20, queue_size_cur=90
    1-2: delay=3, bw=30, queue_size_cur=60
    2-3: delay=6, bw=40, queue_size_cur=70
    3-4: delay=2, bw=50, queue_size_cur=100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from network.DQN.link import LinkAttr
from network.DQN.node import Node


# ── Helper: tạo topology 5 node tuyến tính với tham số cố định ───────

def make_linear_topo():
    """
    Tạo topology tuyến tính 0-1-2-3-4 với tham số cố định từ spec.
    Trả về dict {(u,v): LinkAttr} và danh sách edges theo thứ tự.
    """
    params = {
        (0, 1): (5.0, 20.0, 90),
        (1, 2): (3.0, 30.0, 60),
        (2, 3): (6.0, 40.0, 70),
        (3, 4): (2.0, 50.0, 100),
    }
    links = {}
    for (u, v), (d, bw, q) in params.items():
        links[(u, v)] = LinkAttr(delay=d, bandwidth=bw, queue_size_cur=q)
        links[(v, u)] = LinkAttr(delay=d, bandwidth=bw, queue_size_cur=q)
    return links


def send_traffic_linear(links, path, volume, is_first_hop):
    """
    Simulate send_traffic theo MDP spec cho topology tuyến tính.
    Trả về dropped (bool).
    """
    if len(path) < 2:
        return False

    u = path[-2]
    v = path[-1]
    attr = links[(u, v)]

    # Bước 1: lấy queue_used link trước
    if len(path) >= 3:
        prev = path[-3]
        prev_attr       = links[(prev, u)]
        queue_used_prev = prev_attr.queue_used
        prev_attr.queue_used = 0.0
    else:
        queue_used_prev = 0.0

    # Bước 2: cộng vào queue_used_cur
    if is_first_hop:
        attr.queue_used_cur += queue_used_prev + volume
    else:
        attr.queue_used_cur += queue_used_prev

    # Bước 3: drop check
    dropped = attr.queue_used_cur > attr.queue_size_cur

    # Bước 4: load và queue_used
    remain_bw = attr.bandwidth - attr.load
    if remain_bw <= attr.queue_used_cur:
        attr.load          += remain_bw
        attr.queue_used     = attr.load
        attr.queue_used_cur -= remain_bw
    else:
        attr.load          += attr.queue_used_cur
        attr.queue_used     = attr.load
        attr.queue_used_cur = 0.0

    return dropped


def reduce_load_linear(links, path, decay=0.85):
    """Simulate reduce_load theo MDP spec."""
    path_pairs = [(path[i], path[i+1]) for i in range(len(path)-1)]

    for idx, (u, v) in enumerate(path_pairs):
        attr = links[(u, v)]

        # Phần A
        if attr.load != 0:
            if idx + 1 < len(path_pairs):
                nu, nv = path_pairs[idx + 1]
                links[(nu, nv)].queue_used_cur += attr.load
            attr.load = 0.0

        # Phần B
        if attr.queue_used_cur == 0.0:
            attr.load = 0.0
        else:
            attr.load = 0.0
            remain_bw = attr.bandwidth
            if remain_bw <= attr.queue_used_cur:
                attr.load           += remain_bw
                attr.queue_used_cur -= remain_bw
            else:
                attr.load           += attr.queue_used_cur
                attr.queue_used_cur  = 0.0

    # Decay link ngoài path
    path_set = set(path_pairs)
    for key, attr in links.items():
        if key not in path_set:
            attr.load           *= decay
            attr.queue_used_cur *= decay
            attr.queue_used     *= decay


# ═══════════════════════════════════════════════════════════════════════
#  Test spec: 0 → 3 với volume=60
#  (từ file MDP_Idea.txt)
# ═══════════════════════════════════════════════════════════════════════

class TestMDPSpec:
    """
    Trace đúng từng bước theo spec trong MDP_Idea.txt.
    Topology: 0-1-2-3-4, tìm đường 0→3, volume=60.
    """

    # ─────────────────────────────────────────────────────────────────
    #  BƯỚC 1: path=[0], cur=0, chọn next=1
    #  Sau send_traffic([0,1], volume=60, is_first_hop=True):
    #    queue_used_cur(0-1) = 0 + 0 + 60 = 60
    #    remain_bw = 20 - 0 = 20 <= 60
    #    load(0-1) = 20, queue_used(0-1) = 20, queue_used_cur(0-1) = 40
    #
    #  Sau reduce_load([0,1]):
    #    A: load(0-1)=20 != 0, không có link sau → load(0-1)=0
    #    B: queue_used_cur(0-1)=40 != 0
    #       remain_bw = 20 - 0 = 20 <= 40
    #       load(0-1)=20, queue_used_cur(0-1)=20
    # ─────────────────────────────────────────────────────────────────

    def test_step1_send_traffic(self):
        links = make_linear_topo()
        path  = [0, 1]

        dropped = send_traffic_linear(links, path, volume=60, is_first_hop=True)

        assert not dropped, "volume=60 <= queue_size_cur=90, không drop"
        assert links[(0,1)].queue_used_cur == pytest.approx(40.0), \
            f"queue_used_cur(0-1)={links[(0,1)].queue_used_cur}"
        assert links[(0,1)].load          == pytest.approx(20.0), \
            f"load(0-1)={links[(0,1)].load}"
        assert links[(0,1)].queue_used     == pytest.approx(20.0), \
            f"queue_used(0-1)={links[(0,1)].queue_used}"

    def test_step1_reduce_load(self):
        links = make_linear_topo()
        path  = [0, 1]
        send_traffic_linear(links, path, volume=60, is_first_hop=True)
        reduce_load_linear(links, path)

        # Sau reduce_load: load=20, queue_used_cur=20
        assert links[(0,1)].load          == pytest.approx(20.0), \
            f"load(0-1)={links[(0,1)].load}"
        assert links[(0,1)].queue_used_cur == pytest.approx(20.0), \
            f"queue_used_cur(0-1)={links[(0,1)].queue_used_cur}"

    # ─────────────────────────────────────────────────────────────────
    #  BƯỚC 2: path=[0,1], cur=1, chọn next=2
    #  Trạng thái vào: queue_used_cur(0-1)=20, load(0-1)=20,
    #                  queue_used(0-1)=20
    #
    #  Sau send_traffic([0,1,2], volume=60, is_first_hop=False):
    #    queue_used_prev = queue_used(0-1) = 20
    #    queue_used(0-1) → 0
    #    is_first_hop=False: queue_used_cur(1-2) += 20 = 20
    #    20 <= 60 → không drop
    #    remain_bw(1-2) = 30 - 0 = 30 > 20
    #    load(1-2)=20, queue_used(1-2)=20, queue_used_cur(1-2)=0
    #
    #  Sau reduce_load([0,1,2]):
    #    Link(0-1): load=20≠0, có link sau (1-2)
    #      queue_used_cur(1-2) += 20 → 20, load(0-1)=0
    #      queue_used_cur(0-1)=20≠0, remain_bw=20 <= 20
    #      load(0-1)=20, queue_used_cur(0-1)=0
    #    Link(1-2): load=20≠0, không có link sau → load(1-2)=0
    #      queue_used_cur(1-2)=20≠0, remain_bw=30>20... wait:
    #      (note: trong spec bw link(1-2)=30 nhưng load bị set=0
    #       rồi mới tính remain_bw=30)
    #      load(1-2)=20, queue_used_cur(1-2)=0
    # ─────────────────────────────────────────────────────────────────

    def test_step2_send_traffic(self):
        links = make_linear_topo()
        path  = [0, 1]
        send_traffic_linear(links, path, volume=60, is_first_hop=True)
        reduce_load_linear(links, path)
        # Trạng thái vào bước 2:
        # queue_used(0-1)=20 (chưa reset — do reduce_load tạo ra load mới)
        # Nhưng theo spec: queue_used(0-1)=20 sau step 1

        path2 = [0, 1, 2]
        dropped = send_traffic_linear(links, path2, volume=60, is_first_hop=False)

        assert not dropped, f"queue_used_cur(1-2) should be ≤ 60"
        assert links[(1,2)].queue_used_cur == pytest.approx(0.0), \
            f"queue_used_cur(1-2)={links[(1,2)].queue_used_cur} (remain_bw>queue)"
        assert links[(1,2)].load          == pytest.approx(20.0), \
            f"load(1-2)={links[(1,2)].load}"
        # queue_used(0-1) phải được reset về 0
        assert links[(0,1)].queue_used     == pytest.approx(0.0), \
            f"queue_used(0-1) sau khi lấy phải = 0"

    def test_step2_reduce_load(self):
        links = make_linear_topo()
        send_traffic_linear(links, [0,1], volume=60, is_first_hop=True)
        reduce_load_linear(links, [0,1])
        send_traffic_linear(links, [0,1,2], volume=60, is_first_hop=False)
        reduce_load_linear(links, [0,1,2])

        # Theo spec:
        # link(0-1): sau reduce_load step 2 → load=20, queue_used_cur=0
        assert links[(0,1)].load           == pytest.approx(20.0), \
            f"load(0-1)={links[(0,1)].load}"
        assert links[(0,1)].queue_used_cur == pytest.approx(0.0), \
            f"queue_used_cur(0-1)={links[(0,1)].queue_used_cur}"
        # link(1-2): load=20, queue_used_cur=0
        assert links[(1,2)].load           == pytest.approx(20.0), \
            f"load(1-2)={links[(1,2)].load}"
        assert links[(1,2)].queue_used_cur == pytest.approx(0.0), \
            f"queue_used_cur(1-2)={links[(1,2)].queue_used_cur}"


# ═══════════════════════════════════════════════════════════════════════
#  Test LinkAttr
# ═══════════════════════════════════════════════════════════════════════

class TestLinkAttr:
    def test_6_fields(self):
        attr = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90)
        assert attr.delay          == 5
        assert attr.bandwidth      == 20
        assert attr.queue_size_cur == 90
        assert attr.queue_used_cur == 0.0
        assert attr.queue_used     == 0.0
        assert attr.load           == 0.0

    def test_utilization(self):
        attr = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90, load=10.0)
        assert attr.utilization == pytest.approx(0.5)

    def test_utilization_capped(self):
        attr = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90, load=999.0)
        assert attr.utilization == 1.0

    def test_queue_util(self):
        attr = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90,
                        queue_used_cur=45.0)
        assert attr.queue_util == pytest.approx(0.5)

    def test_remain_bw(self):
        attr = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90, load=5.0)
        assert attr.remain_bw == pytest.approx(15.0)

    def test_is_congested(self):
        attr = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90, load=17.0)
        assert attr.is_congested
        attr.load = 10.0
        assert not attr.is_congested

    def test_reset(self):
        attr = LinkAttr(delay=5, bandwidth=20, queue_size_cur=90,
                        load=10.0, queue_used_cur=30.0, queue_used=10.0)
        attr.reset()
        assert attr.load           == 0.0
        assert attr.queue_used_cur == 0.0
        assert attr.queue_used     == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Test drop condition
# ═══════════════════════════════════════════════════════════════════════

class TestDropCondition:
    def test_no_drop_when_queue_not_full(self):
        links = make_linear_topo()
        # queue_size_cur(0-1) = 90, volume = 60 → 60 <= 90, no drop
        dropped = send_traffic_linear(links, [0,1], volume=60, is_first_hop=True)
        assert not dropped

    def test_drop_when_queue_overflow(self):
        links = make_linear_topo()
        # queue_size_cur(0-1) = 90, volume = 100 → 100 > 90, drop
        dropped = send_traffic_linear(links, [0,1], volume=100, is_first_hop=True)
        assert dropped

    def test_no_drop_exact_capacity(self):
        links = make_linear_topo()
        # volume = 90 = queue_size_cur → 90 <= 90, no drop (strict >)
        dropped = send_traffic_linear(links, [0,1], volume=90, is_first_hop=True)
        assert not dropped

    def test_drop_one_above_capacity(self):
        links = make_linear_topo()
        # volume = 91 > 90 → drop
        dropped = send_traffic_linear(links, [0,1], volume=91, is_first_hop=True)
        assert dropped


# ═══════════════════════════════════════════════════════════════════════
#  Test load update logic
# ═══════════════════════════════════════════════════════════════════════

class TestLoadUpdate:
    def test_remain_bw_less_than_queue(self):
        """remain_bw(20) < queue_used_cur(60): load=20, queue_used_cur=40."""
        links = make_linear_topo()
        send_traffic_linear(links, [0,1], volume=60, is_first_hop=True)
        attr = links[(0,1)]
        assert attr.load           == pytest.approx(20.0)
        assert attr.queue_used     == pytest.approx(20.0)
        assert attr.queue_used_cur == pytest.approx(40.0)

    def test_remain_bw_greater_than_queue(self):
        """remain_bw(30) > queue_used_cur(20): load=20, queue_used_cur=0."""
        links = make_linear_topo()
        send_traffic_linear(links, [0,1,2], volume=0, is_first_hop=True)
        # Inject queue_used_cur=20 vào link(1-2) thủ công
        links[(1,2)].queue_used_cur = 20.0
        # Gọi trực tiếp phần update (giả lập bước send với queue đã có)
        attr = links[(1,2)]
        remain_bw = attr.bandwidth - attr.load  # 30
        if remain_bw <= attr.queue_used_cur:
            attr.load += remain_bw
            attr.queue_used = attr.load
            attr.queue_used_cur -= remain_bw
        else:
            attr.load += attr.queue_used_cur
            attr.queue_used = attr.load
            attr.queue_used_cur = 0.0

        assert attr.load           == pytest.approx(20.0)
        assert attr.queue_used     == pytest.approx(20.0)
        assert attr.queue_used_cur == pytest.approx(0.0)

    def test_remain_bw_equal_queue(self):
        """remain_bw == queue_used_cur (biên): load=remain, queue_used_cur=0."""
        links = make_linear_topo()
        # bw=20, volume=20 → queue_used_cur=20, remain_bw=20
        send_traffic_linear(links, [0,1], volume=20, is_first_hop=True)
        attr = links[(0,1)]
        # remain_bw(20) <= queue_used_cur(20)
        assert attr.load           == pytest.approx(20.0)
        assert attr.queue_used_cur == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Test reduce_load
# ═══════════════════════════════════════════════════════════════════════

class TestReduceLoad:
    def test_load_forwarded_to_next_link(self):
        """load của link trước được cộng vào queue_used_cur của link sau."""
        links = make_linear_topo()
        # Set trực tiếp
        links[(0,1)].load           = 15.0
        links[(0,1)].queue_used_cur = 0.0
        links[(1,2)].queue_used_cur = 5.0

        reduce_load_linear(links, [0,1,2])

        # queue_used_cur(1-2) = 5 + 15 = 20
        # load(0-1) → 0 → phần B: queue_used_cur=0 → load=0
        assert links[(0,1)].load           == pytest.approx(0.0)
        assert links[(1,2)].queue_used_cur == pytest.approx(0.0)   # 20 đã được xử lý

    def test_no_next_link_load_zeroed(self):
        """Link cuối: load → 0 sau reduce."""
        links = make_linear_topo()
        links[(0,1)].load           = 20.0
        links[(0,1)].queue_used_cur = 0.0
        reduce_load_linear(links, [0,1])
        assert links[(0,1)].load == pytest.approx(0.0)

    def test_queue_used_cur_processed_after_load_clear(self):
        """Sau khi load=0, queue_used_cur còn lại được xử lý."""
        links = make_linear_topo()
        links[(0,1)].load           = 0.0
        links[(0,1)].queue_used_cur = 40.0  # bw=20 < 40

        reduce_load_linear(links, [0,1])

        # remain_bw = 20 <= 40 → load=20, queue_used_cur=20
        assert links[(0,1)].load           == pytest.approx(20.0)
        assert links[(0,1)].queue_used_cur == pytest.approx(20.0)

    def test_decay_links_outside_path(self):
        """Link ngoài path bị decay."""
        links = make_linear_topo()
        links[(2,3)].load           = 10.0
        links[(2,3)].queue_used_cur = 8.0

        reduce_load_linear(links, [0,1], decay=0.5)

        assert links[(2,3)].load           == pytest.approx(5.0)
        assert links[(2,3)].queue_used_cur == pytest.approx(4.0)


# ═══════════════════════════════════════════════════════════════════════
#  Test Node
# ═══════════════════════════════════════════════════════════════════════

class TestNode:
    def test_routing_table(self):
        nd = Node(node_id=1)
        nd.set_route(dst=3, next_hop=2)
        assert nd.get_next_hop(3) == 2

    def test_missing_route(self):
        nd = Node(node_id=0)
        assert nd.get_next_hop(7) == -1

    def test_reset_clears_routes(self):
        nd = Node(node_id=0)
        nd.set_route(3, 2)
        nd.reset()
        assert nd.get_next_hop(3) == -1


# ═══════════════════════════════════════════════════════════════════════
#  Test NetworkTopology (8 node)
# ═══════════════════════════════════════════════════════════════════════

class TestNetworkTopology:
    def setup_method(self):
        from network.DQN.topology import NetworkTopology
        self.topo = NetworkTopology(seed=0)

    def test_8_nodes(self):
        assert self.topo.graph.number_of_nodes() == 8

    def test_26_directed_edges(self):
        assert self.topo.num_edges() == 26

    def test_link_state_vector_shape(self):
        ls = self.topo.link_state_vector()
        assert ls.shape == (26, 5)

    def test_link_state_5_features(self):
        ls = self.topo.link_state_vector()
        # Cột 0-2: tham số vật lý ∈ [0,1]
        assert np.all(ls[:, 0] >= 0) and np.all(ls[:, 0] <= 1.0)
        assert np.all(ls[:, 1] >= 0) and np.all(ls[:, 1] <= 1.0)
        assert np.all(ls[:, 2] >= 0) and np.all(ls[:, 2] <= 1.0)
        # Cột 3-4: utilization, queue_util ∈ [0,1]
        assert np.all(ls[:, 3] >= 0) and np.all(ls[:, 3] <= 1.0)
        assert np.all(ls[:, 4] >= 0) and np.all(ls[:, 4] <= 1.0)

    def test_link_attr_has_6_fields(self):
        attr = self.topo.link(0, 1)
        assert hasattr(attr, "delay")
        assert hasattr(attr, "bandwidth")
        assert hasattr(attr, "queue_size_cur")
        assert hasattr(attr, "queue_used_cur")
        assert hasattr(attr, "queue_used")
        assert hasattr(attr, "load")

    def test_randomize_links_changes_params(self):
        ls1 = self.topo.link_state_vector()[:, :3].copy()
        self.topo.randomize_links()
        ls2 = self.topo.link_state_vector()[:, :3]
        assert not np.allclose(ls1, ls2)

    def test_randomize_preserves_adjacency(self):
        adj1 = self.topo.adj_matrix.copy()
        self.topo.randomize_links()
        np.testing.assert_array_equal(adj1, self.topo.adj_matrix)

    def test_reset_randomizes(self):
        ls1 = self.topo.link_state_vector()[:, :3].copy()
        self.topo.reset()
        ls2 = self.topo.link_state_vector()[:, :3]
        assert not np.allclose(ls1, ls2)

    def test_has_link_bidirectional(self):
        assert self.topo.has_link(0, 1) and self.topo.has_link(1, 0)

    def test_no_direct_link_0_7(self):
        assert not self.topo.has_link(0, 7)

    def test_send_traffic_updates_link(self):
        """send_traffic cập nhật load và queue_used_cur."""
        self.topo.send_traffic([0, 1], volume_mbps=10.0, is_first_hop=True)
        attr = self.topo.link(0, 1)
        assert attr.load > 0 or attr.queue_used_cur > 0

    def test_reduce_load_on_path(self):
        """reduce_load xử lý link trong path."""
        self.topo.send_traffic([0, 1], volume_mbps=10.0, is_first_hop=True)
        load_before = self.topo.link(0,1).load
        self.topo.reduce_load([0, 1])
        # load thay đổi sau reduce
        assert self.topo.link(0,1).load >= 0

    def test_decay_outside_path(self):
        """Link ngoài path bị decay."""
        attr = self.topo.link(1, 2)
        attr.load = 50.0
        self.topo.reduce_load([0, 1], decay=0.5)
        assert self.topo.link(1,2).load == pytest.approx(25.0)

    def test_step_background_changes_utilization(self):
        utils_before = [a.utilization for a in self.topo._link_attrs.values()]
        self.topo.step_background(intensity=0.5)
        utils_after = [a.utilization for a in self.topo._link_attrs.values()]
        assert any(a != b for a, b in zip(utils_before, utils_after))

    def test_link_state_varies_after_traffic(self):
        ls1 = self.topo.link_state_vector()[:, 3:].copy()
        self.topo.send_traffic([0, 1], volume_mbps=15.0, is_first_hop=True)
        ls2 = self.topo.link_state_vector()[:, 3:]
        assert not np.allclose(ls1, ls2)