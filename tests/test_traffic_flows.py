"""Tests for `aperta.traffic_flows`.

Run with:
    cd src && python -m unittest aperta.tests.test_traffic_flows

Two public entry points:
    - `get` — naive edge-betweenness-based traffic flow with optional cutoff
      OR a `nested_node_sample` dict for nested betweenness.
    - `nested_node_sample` — sample destinations for sampled origins, weighted
      by cost-derived scores, integrating cell + zone + region tiers.
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODNodePairs
from aperta.traffic_flows import get, nested_node_sample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _toy_graph() -> nx.MultiDiGraph:
    """Tiny multigraph: A → B → C → D plus a shortcut A → C, all edges
    `length = cost = 1`."""
    g = nx.MultiDiGraph()
    for u, v in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'C')]:
        g.add_edge(u, v, length=1.0, cost=1.0)
    return g


def _toy_tiered_inputs():
    """4 cells in 2 zones, no region tier.

    Layout:
        c1, c2 ∈ Z1;  c3, c4 ∈ Z2.

    Cell-tier dests (self-pair always included at cost 0):
        c1: [c1, c2]   c2: [c1, c2]   c3: [c3, c4]   c4: [c3, c4]
    Zone-tier dests (Z1 ↔ Z2):
        Z1: [Z2]   Z2: [Z1]
    Costs: 0 for self, 100 for in-zone other, 500 zone-tier.
    Weights: 1 per cell, 10 per zone.
    """
    pairs = TieredODNodePairs(
        cells_to_cells={
            'c1': np.array(['c1', 'c2']),
            'c2': np.array(['c1', 'c2']),
            'c3': np.array(['c3', 'c4']),
            'c4': np.array(['c3', 'c4']),
        },
        zones_to_zones={
            'Z1': np.array(['Z2']),
            'Z2': np.array(['Z1']),
        },
    )
    weights = TieredODNodePairs(
        cells_to_cells={k: np.ones(len(v)) for k, v in pairs.cells_to_cells.items()},
        zones_to_zones={k: np.array([10.0]) for k in pairs.zones_to_zones},
    )
    costs = TieredODNodePairs(
        cells_to_cells={
            'c1': np.array([0.0, 100.0]),
            'c2': np.array([100.0, 0.0]),
            'c3': np.array([0.0, 100.0]),
            'c4': np.array([100.0, 0.0]),
        },
        zones_to_zones={
            'Z1': np.array([500.0]),
            'Z2': np.array([500.0]),
        },
    )
    cell_to_zone_node = {'c1': 'Z1', 'c2': 'Z1', 'c3': 'Z2', 'c4': 'Z2'}
    orig_weights = np.array([1.0, 1.0, 1.0, 1.0])
    return pairs, weights, costs, cell_to_zone_node, orig_weights


def _inverse_distance(x: np.ndarray) -> np.ndarray:
    """Vectorised inverse-distance score; `+ 1` avoids div-by-zero at self-pairs."""
    return 1.0 / (x + 1.0)


# ---------------------------------------------------------------------------
# `get` — naive betweenness-based traffic flow estimation
# ---------------------------------------------------------------------------

class GetTrafficFlowsTestCase(unittest.TestCase):

    def test_returns_series(self):
        g = _toy_graph()
        flows = get(g, routing_edge_weight='cost', expected_km_driven=100.0)
        self.assertIsInstance(flows, pd.Series)
        # Each edge is keyed by (u, v, k) (MultiDiGraph).
        for key in flows.index:
            self.assertEqual(len(key), 3)

    def test_normalisation_matches_expected_km(self):
        """`sum(flow × length) == expected_km_driven` by construction."""
        g = _toy_graph()
        flows = get(g, routing_edge_weight='cost', expected_km_driven=100.0)
        lengths = nx.get_edge_attributes(g, 'length')
        total_vkt = sum(v * lengths[k] for k, v in flows.items())
        self.assertAlmostEqual(total_vkt, 100.0)

    def test_normalisation_with_cutoff(self):
        """Same normalisation holds when a cutoff truncates the betweenness."""
        g = _toy_graph()
        flows = get(g, routing_edge_weight='cost',
                    expected_km_driven=50.0, cutoff=1)
        lengths = nx.get_edge_attributes(g, 'length')
        total_vkt = sum(v * lengths[k] for k, v in flows.items())
        self.assertAlmostEqual(total_vkt, 50.0)
        self.assertGreater(flows.sum(), 0)

    def test_nested_node_sample_path(self):
        """With `nested_node_sample` given, only edges on the sampled paths
        get positive flow. A→D goes via A-C-D (the shortcut)."""
        g = _toy_graph()
        sample = {'A': ['D']}
        flows = get(g, routing_edge_weight='cost',
                    expected_km_driven=10.0, nested_node_sample=sample)
        lengths = nx.get_edge_attributes(g, 'length')
        total_vkt = sum(v * lengths[k] for k, v in flows.items())
        self.assertAlmostEqual(total_vkt, 10.0)
        # The shortcut A-C and the segment C-D carry the trip; A-B and B-C are unused.
        self.assertGreater(flows.get(('A', 'C', 0), 0), 0)
        self.assertGreater(flows.get(('C', 'D', 0), 0), 0)
        self.assertEqual(flows.get(('A', 'B', 0), 0), 0)
        self.assertEqual(flows.get(('B', 'C', 0), 0), 0)


# ---------------------------------------------------------------------------
# `nested_node_sample` — weighted-sampled origin / destination pairs
# ---------------------------------------------------------------------------

class NestedNodeSampleTestCase(unittest.TestCase):

    def setUp(self):
        (self.pairs, self.weights, self.costs,
         self.c2z, self.orig_weights) = _toy_tiered_inputs()

    def test_returns_dict_keyed_by_origin_cells(self):
        rs = np.random.RandomState(42)
        out = nested_node_sample(
            self.pairs, self.weights, self.costs, self.c2z,
            self.orig_weights, _inverse_distance,
            n_orig=4, n_dest=10, random_state=rs,
        )
        self.assertIsInstance(out, dict)
        for k in out.keys():
            self.assertIn(k, ('c1', 'c2', 'c3', 'c4'))

    def test_n_dest_per_origin(self):
        rs = np.random.RandomState(42)
        out = nested_node_sample(
            self.pairs, self.weights, self.costs, self.c2z,
            self.orig_weights, _inverse_distance,
            n_orig=4, n_dest=15, random_state=rs,
        )
        for dests in out.values():
            self.assertEqual(len(dests), 15)

    def test_origin_weight_concentration(self):
        """All origin-weight on c2 → c2 is the only sampled origin (and
        appears regardless of n_orig because dupes are de-duped)."""
        rs = np.random.RandomState(42)
        weights_c2_only = np.array([0.0, 1.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs, self.weights, self.costs, self.c2z,
            weights_c2_only, _inverse_distance,
            n_orig=4, n_dest=10, random_state=rs,
        )
        self.assertEqual(list(out.keys()), ['c2'])

    def test_dest_score_concentration_on_self_pair(self):
        """With inverse-distance scoring, the cost-0 self-pair dominates
        the dest score for an origin like c1: ~97 % of samples should be c1
        (cell self, score 1.0 vs c2's 1/101 ≈ 0.0099 and Z2's 10/501 ≈ 0.02).

        Origin weights are concentrated on c1 so the sampled origin set is
        deterministic regardless of `random_state`.
        """
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs, self.weights, self.costs, self.c2z,
            c1_only, _inverse_distance,
            n_orig=1, n_dest=1000, random_state=rs,
        )
        c1_dests = out['c1']
        self.assertGreater((c1_dests == 'c1').sum() / len(c1_dests), 0.9)

    def test_zone_tier_dests_appear_when_weighted_up(self):
        """If zone-tier dest weights are boosted enough, zone-tier dests
        (here Z2 reached from c1's zone Z1) start appearing in the output."""
        big_zone_weights = TieredODNodePairs(
            cells_to_cells=self.weights.cells_to_cells,
            zones_to_zones={k: np.array([10_000.0])
                            for k in self.pairs.zones_to_zones},
        )
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs, big_zone_weights, self.costs, self.c2z,
            c1_only, _inverse_distance,
            n_orig=1, n_dest=1000, random_state=rs,
        )
        self.assertIn('Z2', set(out['c1'].tolist()))

    def test_reproducible_with_random_state(self):
        rs1 = np.random.RandomState(42)
        rs2 = np.random.RandomState(42)
        out1 = nested_node_sample(
            self.pairs, self.weights, self.costs, self.c2z,
            self.orig_weights, _inverse_distance,
            n_orig=4, n_dest=10, random_state=rs1,
        )
        out2 = nested_node_sample(
            self.pairs, self.weights, self.costs, self.c2z,
            self.orig_weights, _inverse_distance,
            n_orig=4, n_dest=10, random_state=rs2,
        )
        self.assertSetEqual(set(out1.keys()), set(out2.keys()))
        for k in out1:
            np.testing.assert_array_equal(out1[k], out2[k])

    def test_mask_filters_cell_tier_dests(self):
        """A cell-tier mask removes specific destinations from the pool —
        masked-out dests should never appear in the sampled output."""
        mask = TieredODNodePairs(
            cells_to_cells={
                # For c1 origin: drop c1 (self) — only c2 remains at cell tier.
                'c1': np.array([False, True]),
                'c2': np.array([True, True]),
                'c3': np.array([True, True]),
                'c4': np.array([True, True]),
            },
            zones_to_zones={k: np.array([True])
                            for k in self.pairs.zones_to_zones},
        )
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs, self.weights, self.costs, self.c2z,
            c1_only, _inverse_distance,
            n_orig=1, n_dest=200, random_state=rs,
            mask=mask,
        )
        self.assertNotIn('c1', set(out['c1'].tolist()))

    def test_mask_filters_zone_tier_dests(self):
        """A zone-tier mask removes zone-tier dests for the affected zones."""
        mask = TieredODNodePairs(
            cells_to_cells={k: np.ones(len(v), dtype=bool)
                            for k, v in self.pairs.cells_to_cells.items()},
            # Z1's outgoing zone-tier dest (Z2) is masked out.
            zones_to_zones={'Z1': np.array([False]), 'Z2': np.array([True])},
        )
        big_zone_weights = TieredODNodePairs(
            cells_to_cells=self.weights.cells_to_cells,
            zones_to_zones={k: np.array([10_000.0])
                            for k in self.pairs.zones_to_zones},
        )
        rs = np.random.RandomState(42)
        # Sample only c1 and c3 (so we deterministically check both branches).
        c1_and_c3 = np.array([1.0, 0.0, 1.0, 0.0])
        out = nested_node_sample(
            self.pairs, big_zone_weights, self.costs, self.c2z,
            c1_and_c3, _inverse_distance,
            n_orig=20, n_dest=200, random_state=rs,
            mask=mask,
        )
        # c1 (in Z1) had Z2 as zone-tier dest — should now be absent.
        self.assertNotIn('Z2', set(out['c1'].tolist()))
        # c3 (in Z2) still has Z1 as zone-tier dest — should still appear.
        self.assertIn('Z1', set(out['c3'].tolist()))


if __name__ == '__main__':
    unittest.main()
