"""
tests/DQN/test_network.py
pytest tests/DQN/test_network.py -v
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from network.DQN.topology import NetworkTopology
from network.DQN.traffic_generator import TrafficGenerator
from network.DQN.metrics import NetworkMetrics
from network.DQN.link import Link
from network.DQN.node import Node


@pytest.fixture
def topo():
    return NetworkTopology(seed=0)


class TestTopology:
    def test_8_nodes(self, topo):
        assert topo.graph.number_of_nodes() == 8

    def test_26_directed_edges(self, topo):
        assert topo.num_edges() == 26

    def test_link_state_5_features(self, topo):
        ls = topo.link_state_vector()
        assert ls.shape == (26, 5), f"got {ls.shape}"

    def test_5_features_in_range(self, topo):
        ls = topo.link_state_vector()
        assert np.all(ls >= 0)
        assert np.all(ls[:, 3] <= 1.0)   # utilization
        assert np.all(ls[:, 4] <= 1.0)   # queue_util

    def test_has_link_bidirectional(self, topo):
        assert topo.has_link(0, 1) and topo.has_link(1, 0)

    def test_no_link_0_7(self, topo):
        assert not topo.has_link(0, 7)

    def test_randomize_links_changes_params(self, topo):
        """randomize_links thay đổi delay/bw/queue_size."""
        ls1 = topo.link_state_vector()[:, :3].copy()
        topo.randomize_links()
        ls2 = topo.link_state_vector()[:, :3].copy()
        assert not np.allclose(ls1, ls2)

    def test_randomize_preserves_adjacency(self, topo):
        adj1 = topo.adj_matrix.copy()
        topo.randomize_links()
        np.testing.assert_array_equal(adj1, topo.adj_matrix)

    def test_reset_calls_randomize(self, topo):
        """reset() sinh lại tham số link."""
        ls1 = topo.link_state_vector()[:, :3].copy()
        topo.reset()
        ls2 = topo.link_state_vector()[:, :3].copy()
        assert not np.allclose(ls1, ls2)

    def test_send_traffic_updates_load(self, topo):
        topo.send_traffic([0, 1, 3], 10.0)
        assert topo.link(0, 1).load > 0
        assert topo.link(1, 3).load > 0

    def test_utilization_clamp(self, topo):
        topo.send_traffic([0, 1], 9999.0)
        assert topo.link(0, 1).utilization <= 1.0

    def test_decay_load_reduces_utilization(self, topo):
        for a in topo._link_attrs.values():
            a.load = a.bandwidth * 0.9
        u_before = np.mean([a.utilization for a in topo._link_attrs.values()])
        topo.decay_load()
        u_after = np.mean([a.utilization for a in topo._link_attrs.values()])
        assert u_after < u_before

    def test_step_background_changes_utilization(self, topo):
        utils_before = [a.utilization for a in topo._link_attrs.values()]
        topo.step_background(intensity=0.5)
        utils_after  = [a.utilization for a in topo._link_attrs.values()]
        assert any(a != b for a, b in zip(utils_before, utils_after))

    def test_shortest_path_0_to_7(self, topo):
        path = topo.shortest_path(0, 7)
        assert path[0] == 0 and path[-1] == 7

    def test_summary_keys(self, topo):
        s = topo.summary()
        for k in ("avg_utilization", "max_utilization",
                  "avg_queue_util", "congested_links"):
            assert k in s


class TestTrafficGenerator:
    def test_no_self_loop(self):
        gen = TrafficGenerator(seed=0)
        for _ in range(100):
            s, d, v = gen.generate()
            assert s != d

    def test_nodes_in_range(self):
        gen = TrafficGenerator(seed=1)
        for _ in range(50):
            s, d, _ = gen.generate()
            assert 0 <= s < 8 and 0 <= d < 8

    def test_volume_positive(self):
        gen = TrafficGenerator(seed=2)
        for _ in range(50):
            _, _, v = gen.generate()
            assert v >= 1.0


class TestLink:
    def test_utilization(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50, load=50.0)
        assert abs(lk.utilization - 0.5) < 1e-6

    def test_congested(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50, load=85.0)
        assert lk.is_congested
        lk.load = 50.0
        assert not lk.is_congested

    def test_reset(self):
        lk = Link(0, 1, delay=2, bandwidth=100, queue_size=50,
                  load=50.0, queue_used=20)
        lk.reset()
        assert lk.load == 0.0 and lk.queue_used == 0


class TestNode:
    def test_routing_table(self):
        nd = Node(node_id=3)
        nd.set_route(7, 5)
        assert nd.get_next_hop(7) == 5
        assert nd.get_next_hop(0) == -1

    def test_enqueue_and_drop(self):
        nd = Node(node_id=0, queue_capacity=10)
        assert nd.enqueue(8) is True
        assert nd.enqueue(5) is False

    def test_reset(self):
        nd = Node(node_id=1)
        nd.set_route(7, 5); nd.enqueue(5)
        nd.reset()
        assert nd.get_next_hop(7) == -1 and nd._queue == 0


class TestMetrics:
    def test_path_delay_positive(self, topo):
        mc   = NetworkMetrics(topo)
        path = topo.shortest_path(0, 7)
        assert mc.path_delay(path) > 0

    def test_drop_prob_zero_at_idle(self, topo):
        mc   = NetworkMetrics(topo)
        path = topo.shortest_path(0, 7)
        assert mc.path_drop_prob(path) == 0.0