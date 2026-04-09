"""
tests/Q_Learning/test_network.py
pytest tests/Q_Learning/test_network.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from network.Q_Learning.topology import NetworkTopology
from network.Q_Learning.traffic_generator import TrafficGenerator
from network.Q_Learning.metrics import NetworkMetrics
from network.Q_Learning.link import Link
from network.Q_Learning.node import Node


# ── Topology ──────────────────────────────────────────────────────────

class TestTopology:
    def setup_method(self):
        self.topo = NetworkTopology()

    def test_node_count(self):
        assert self.topo.graph.number_of_nodes() == 8

    def test_edge_count(self):
        # 13 undirected edges → 26 directed
        assert self.topo.num_edges() == 26

    def test_connected(self):
        import networkx as nx
        assert nx.is_weakly_connected(self.topo.graph)

    def test_adj_matrix_shape(self):
        adj = self.topo.adj_matrix
        assert adj.shape == (8, 8)
        assert adj.dtype == np.float32

    def test_link_state_vector_shape(self):
        ls = self.topo.link_state_vector()
        assert ls.shape == (26, 4)
        assert ls.dtype == np.float32

    def test_link_state_values_in_range(self):
        ls = self.topo.link_state_vector()
        # Utilization và queue_util hợp lệ [0,1]
        assert np.all(ls[:, 2] >= 0) and np.all(ls[:, 2] <= 1)
        assert np.all(ls[:, 3] >= 0) and np.all(ls[:, 3] <= 1)

    def test_has_link(self):
        assert self.topo.has_link(0, 1)
        assert self.topo.has_link(1, 0)   # bidirectional
        assert not self.topo.has_link(0, 7)  # không có link trực tiếp

    def test_neighbors(self):
        nbrs = self.topo.neighbors(0)
        assert 1 in nbrs and 2 in nbrs
        assert len(nbrs) >= 2

    def test_reset_zeroes_load(self):
        self.topo.send_traffic([0, 1, 3], volume_mbps=20.0)
        self.topo.reset()
        for attr in self.topo._link_attrs.values():
            assert attr.load == 0.0
            assert attr.queue_used == 0

    def test_send_traffic_updates_load(self):
        path = [0, 1, 3]
        self.topo.send_traffic(path, volume_mbps=10.0)
        assert self.topo.link(0, 1).load > 0
        assert self.topo.link(1, 3).load > 0

    def test_send_traffic_returns_delay(self):
        path = self.topo.shortest_path(0, 7)
        result = self.topo.send_traffic(path, volume_mbps=5.0)
        assert result["total_delay"] > 0
        assert result["hops"] == len(path) - 1

    def test_utilization_clamps_to_one(self):
        self.topo.send_traffic([0, 1], volume_mbps=9999.0)
        assert self.topo.link(0, 1).utilization <= 1.0

    def test_shortest_path_0_to_7(self):
        path = self.topo.shortest_path(0, 7)
        assert path[0] == 0 and path[-1] == 7
        assert len(path) >= 2

    def test_summary_keys(self):
        s = self.topo.summary()
        for key in ("avg_utilization", "max_utilization",
                    "avg_queue_util", "congested_links"):
            assert key in s


# ── Traffic generator ─────────────────────────────────────────────────

class TestTrafficGenerator:
    def test_no_self_loop(self):
        gen = TrafficGenerator(seed=0)
        for _ in range(100):
            src, dst, vol = gen.generate()
            assert src != dst

    def test_nodes_in_range(self):
        gen = TrafficGenerator(seed=1)
        for _ in range(100):
            src, dst, vol = gen.generate()
            assert 0 <= src < 8 and 0 <= dst < 8

    def test_volume_positive(self):
        gen = TrafficGenerator(seed=2)
        for _ in range(50):
            _, _, vol = gen.generate()
            assert vol >= 1.0

    def test_reproducible_with_seed(self):
        g1 = TrafficGenerator(seed=42)
        g2 = TrafficGenerator(seed=42)
        for _ in range(20):
            assert g1.generate() == g2.generate()


# ── Link dataclass ────────────────────────────────────────────────────

class TestLink:
    def test_utilization(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50, load=50.0)
        assert abs(lk.utilization - 0.5) < 1e-6

    def test_utilization_clamp(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50, load=999.0)
        assert lk.utilization == 1.0

    def test_is_congested(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50, load=85.0)
        assert lk.is_congested
        lk.load = 50.0
        assert not lk.is_congested

    def test_effective_delay_increases_with_load(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50)
        lk.load = 10.0
        d_low = lk.effective_delay
        lk.load = 90.0
        d_high = lk.effective_delay
        assert d_high > d_low

    def test_reset(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50,
                  load=50.0, queue_used=20)
        lk.reset()
        assert lk.load == 0.0 and lk.queue_used == 0


# ── Node ──────────────────────────────────────────────────────────────

class TestNode:
    def test_routing_table(self):
        nd = Node(node_id=3)
        nd.set_route(dst=7, next_hop=5)
        assert nd.get_next_hop(7) == 5
        assert nd.get_next_hop(0) == -1   # no route

    def test_enqueue_success(self):
        nd = Node(node_id=0, queue_capacity=50)
        assert nd.enqueue(30) is True
        assert nd.queue_util == pytest.approx(0.6)

    def test_enqueue_drop_when_full(self):
        nd = Node(node_id=0, queue_capacity=10)
        nd.enqueue(8)
        assert nd.enqueue(5) is False  # drop

    def test_reset_clears_everything(self):
        nd = Node(node_id=1)
        nd.set_route(7, 5)
        nd.enqueue(20)
        nd.reset()
        assert nd.get_next_hop(7) == -1
        assert nd._queue == 0


# ── NetworkMetrics ────────────────────────────────────────────────────

class TestNetworkMetrics:
    def setup_method(self):
        self.topo = NetworkTopology()
        self.mc   = NetworkMetrics(self.topo)

    def test_path_delay_positive(self):
        path = self.topo.shortest_path(0, 7)
        assert self.mc.path_delay(path) > 0

    def test_path_delay_single_hop(self):
        delay = self.mc.path_delay([0, 1])
        link_delay = self.topo.link(0, 1).delay
        assert abs(delay - link_delay) < 1e-6

    def test_drop_prob_zero_at_idle(self):
        path = self.topo.shortest_path(0, 7)
        assert self.mc.path_drop_prob(path) == 0.0

    def test_throughput_equals_volume_at_idle(self):
        path = self.topo.shortest_path(0, 7)
        assert abs(self.mc.path_throughput(path, 10.0) - 10.0) < 1e-6

    def test_throughput_decreases_under_congestion(self):
        path = self.topo.shortest_path(0, 7)
        # Saturate a link on the path
        for lnk in [self.topo.link(path[i], path[i+1])
                    for i in range(len(path)-1)]:
            lnk.load = lnk.bandwidth * 0.95
            lnk.queue_used = int(lnk.queue_size * 0.95)
        tp_loaded = self.mc.path_throughput(path, 10.0)
        assert tp_loaded <= 10.0