"""Tests for `aperta.utility` — utility-based travel costs and accessibility.

Run with:
    cd src && python -m unittest tests.test_utility
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODNodePairs, TieredODPairs
from aperta.routing import tiered_path_costs
from aperta.utility import (
    RouteFeature,
    Utility,
    add_endpoint_utility,
    route_utility,
)


def _graph_and_pairs():
    """Toy graph:
        a -[w=1, attr=10]- b
        b -[w=2, attr=20]- c
        a -[w=10, attr=100]- c  (long way; shortest a→c is via b)
    """
    g = nx.Graph()
    g.add_node('a', x=0.0, y=0.0)
    g.add_node('b', x=1.0, y=0.0)
    g.add_node('c', x=2.0, y=0.0)
    g.add_edge('a', 'b', w=1.0, attr=10.0)
    g.add_edge('b', 'c', w=2.0, attr=20.0)
    g.add_edge('a', 'c', w=10.0, attr=100.0)
    pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['a', 'b', 'c'])})
    return g, pairs


class RouteUtilityTestCase(unittest.TestCase):
    """`route_utility` returns β_cost·cost + Σ β_route·aggregated_route_feature."""

    def test_cost_only(self):
        """With only a cost coefficient and no route features, U = β_cost · cost."""
        g, pairs = _graph_and_pairs()
        u = Utility(cost_coefficient=-2.0)
        out = route_utility(pairs, g, cost_weight='w', utility=u)
        # Costs: a→a = 0, a→b = 1, a→c via b = 3.
        # Utilities: -2*0 = 0, -2*1 = -2, -2*3 = -6.
        np.testing.assert_array_almost_equal(out.cells_to_cells['a'],
                                             np.array([0.0, -2.0, -6.0]))

    def test_cost_and_route_feature(self):
        """Utility combines cost and aggregated route feature linearly."""
        g, pairs = _graph_and_pairs()
        u = Utility(
            cost_coefficient=-2.0,
            route_features=[
                RouteFeature('attr_total', 'attr', coefficient=-0.1, aggregator='sum'),
            ],
        )
        out = route_utility(pairs, g, cost_weight='w', utility=u)
        # a→a: cost 0, attr_total 0 → -0 - 0 = 0
        # a→b: cost 1, attr_total 10 → -2 - 1 = -3
        # a→c via b: cost 3, attr_total 30 → -6 - 3 = -9
        np.testing.assert_array_almost_equal(out.cells_to_cells['a'],
                                             np.array([0.0, -3.0, -9.0]))

    def test_no_features_uses_tiered_path_costs(self):
        """When there are no route features, the cost component matches
        `tiered_path_costs` exactly (modulo inf → NaN convention for utility)."""
        g, pairs = _graph_and_pairs()
        u = Utility(cost_coefficient=1.0)
        u_out = route_utility(pairs, g, cost_weight='w', utility=u)
        costs = tiered_path_costs(pairs, g, weight='w')
        # With cost_coefficient=1.0, U = cost (where finite).
        np.testing.assert_array_almost_equal(u_out.cells_to_cells['a'],
                                             costs.cells_to_cells['a'])

    def test_unreachable_dest_is_nan(self):
        """Unreachable destinations have NaN utility (signed-quantity convention)."""
        g = nx.Graph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('x', x=10.0, y=10.0)
        # No edge between them.
        pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['a', 'x'])})
        u = Utility(cost_coefficient=-1.0)
        out = route_utility(pairs, g, cost_weight='w', utility=u)
        self.assertEqual(out.cells_to_cells['a'][0], 0.0)  # self-pair, finite
        self.assertTrue(np.isnan(out.cells_to_cells['a'][1]))  # unreachable → NaN

    def test_multiple_route_features(self):
        """Multiple route features add their contributions."""
        g, pairs = _graph_and_pairs()
        u = Utility(
            cost_coefficient=-1.0,
            route_features=[
                RouteFeature('attr_sum', 'attr', coefficient=-0.1, aggregator='sum'),
                RouteFeature('attr_mean', 'attr', coefficient=-0.01, aggregator='mean'),
            ],
        )
        out = route_utility(pairs, g, cost_weight='w', utility=u)
        # a→c: cost 3, attr_sum 30, attr_mean 15.
        # U = -1*3 - 0.1*30 - 0.01*15 = -3 - 3 - 0.15 = -6.15
        self.assertAlmostEqual(out.cells_to_cells['a'][2], -6.15)

    def test_self_pair_with_mean_aggregator_is_finite(self):
        """For a self-pair (cost = 0, path of 0 edges), the 'mean' aggregator
        on the route feature returns NaN — the empty-path is undefined.
        route_utility must handle this: treat the route-feature contribution
        as 0 (no route, no feature contribution), not propagate NaN.

        Regression test: this bug previously caused the cell containing a
        destination to be EXCLUDED from its own gravity / logsum sum, giving
        the cell-with-supermarket a lower accessibility than its neighbours.
        """
        g, pairs = _graph_and_pairs()
        u = Utility(
            cost_coefficient=-2.0,
            route_features=[
                # 'mean' returns NaN for an empty (0-edge) path.
                RouteFeature('attr_mean', 'attr', coefficient=-0.1, aggregator='mean'),
            ],
        )
        out = route_utility(pairs, g, cost_weight='w', utility=u)
        # Self-pair a→a: cost 0, no edges. Route utility should be
        # -2*0 + (-0.1)*0_contribution = 0.0 — finite, not NaN.
        self.assertFalse(np.isnan(out.cells_to_cells['a'][0]))
        self.assertEqual(out.cells_to_cells['a'][0], 0.0)
        # a→b: cost 1, 1 edge with attr 10, mean = 10.
        # Utility = -2*1 + (-0.1)*10 = -2 - 1 = -3.
        self.assertAlmostEqual(out.cells_to_cells['a'][1], -3.0)
        # Sanity: unreachable destinations (cost = inf) STILL get NaN
        # utility — the fix only changes finite-cost zero-edge cases.

    def test_self_pair_unreachable_distinction(self):
        """Self-pair (cost = 0, NaN aggregation) → finite utility.
        Unreachable (cost = inf, NaN aggregation) → NaN utility."""
        g = nx.Graph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('b', x=1.0, y=0.0)
        g.add_edge('a', 'b', w=1.0, attr=10.0)
        # Add disconnected node 'x' to test unreachable case.
        g.add_node('x', x=10.0, y=10.0)
        pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['a', 'b', 'x'])})
        u = Utility(
            cost_coefficient=-1.0,
            route_features=[
                RouteFeature('attr_mean', 'attr', coefficient=-0.1, aggregator='mean'),
            ],
        )
        out = route_utility(pairs, g, cost_weight='w', utility=u)
        # Self-pair: finite (0).
        self.assertEqual(out.cells_to_cells['a'][0], 0.0)
        # Reachable destination: finite (-1*1 - 0.1*10 = -2).
        self.assertAlmostEqual(out.cells_to_cells['a'][1], -2.0)
        # Unreachable destination: NaN (preserved — utility undefined when no path exists).
        self.assertTrue(np.isnan(out.cells_to_cells['a'][2]))

class AddEndpointUtilityTestCase(unittest.TestCase):
    """`add_endpoint_utility` adds constant + origin + destination components."""

    def _setup(self):
        """Toy two-tier setup with origin and destination features."""
        g = nx.Graph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('b', x=1.0, y=0.0)
        g.add_node('c', x=2.0, y=0.0)
        g.add_edge('a', 'b', w=1.0)
        g.add_edge('b', 'c', w=2.0)
        pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['a', 'b', 'c'])})
        cells = pd.DataFrame(
            {'node_id': ['a', 'b', 'c'],
             'pop_density': [100.0, 200.0, 300.0],
             'jobs': [10.0, 20.0, 30.0]},
            index=pd.Index(['cell_a', 'cell_b', 'cell_c'], name='cell_id'),
        )
        return g, pairs, cells

    def test_constant_only(self):
        """With only a constant, U_full = U_route + constant for every OD."""
        g, pairs, cells = self._setup()
        u = Utility(cost_coefficient=-1.0, constant=5.0)
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        full = add_endpoint_utility(r_u, pairs, u, cells=cells)
        np.testing.assert_array_almost_equal(
            full.cells_to_cells['a'], r_u.cells_to_cells['a'] + 5.0)

    def test_destination_feature(self):
        """Destination feature lookup adds β · feature(j) per OD pair."""
        g, pairs, cells = self._setup()
        u = Utility(
            cost_coefficient=-1.0,
            destination_features={'jobs': 0.5},
        )
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        full = add_endpoint_utility(r_u, pairs, u, cells=cells)
        # Destinations are nodes a, b, c with jobs 10, 20, 30.
        # Full utility = route_utility + 0.5 * jobs at dest.
        # a→a: 0  + 0.5*10 = 5
        # a→b: -1 + 0.5*20 = 9
        # a→c: -3 + 0.5*30 = 12
        np.testing.assert_array_almost_equal(
            full.cells_to_cells['a'], np.array([5.0, 9.0, 12.0]))

    def test_origin_feature(self):
        """Origin feature lookup adds β · feature(i) once per origin (broadcast across dests)."""
        g, pairs, cells = self._setup()
        u = Utility(
            cost_coefficient=-1.0,
            origin_features={'pop_density': 0.01},
        )
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        full = add_endpoint_utility(r_u, pairs, u, cells=cells)
        # Origin a has pop_density 100. Contribution: 0.01 * 100 = 1.0 (added to every OD from a).
        # Costs: 0, 1, 3 → utility = 0, -1, -3.
        # Full = utility + 1.0 = 1, 0, -2.
        np.testing.assert_array_almost_equal(
            full.cells_to_cells['a'], np.array([1.0, 0.0, -2.0]))

    def test_combined_origin_dest_constant(self):
        """All components stack additively."""
        g, pairs, cells = self._setup()
        u = Utility(
            constant=2.0,
            cost_coefficient=-1.0,
            origin_features={'pop_density': 0.01},   # +1.0 at origin a
            destination_features={'jobs': 0.5},      # +5, +10, +15 at dests
        )
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        full = add_endpoint_utility(r_u, pairs, u, cells=cells)
        # a→a: 0  + 2 + 1 + 5  = 8
        # a→b: -1 + 2 + 1 + 10 = 12
        # a→c: -3 + 2 + 1 + 15 = 15
        np.testing.assert_array_almost_equal(
            full.cells_to_cells['a'], np.array([8.0, 12.0, 15.0]))

    def test_missing_origin_feature_raises(self):
        g, pairs, cells = self._setup()
        u = Utility(origin_features={'nonexistent': 1.0})
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        with self.assertRaisesRegex(ValueError, "Origin feature"):
            add_endpoint_utility(r_u, pairs, u, cells=cells)

    def test_missing_dest_feature_raises(self):
        g, pairs, cells = self._setup()
        u = Utility(destination_features={'nonexistent': 1.0})
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        with self.assertRaisesRegex(ValueError, "Destination feature"):
            add_endpoint_utility(r_u, pairs, u, cells=cells)

    def test_two_cells_one_node_origin_features_averaged(self):
        """When multiple cells share a network node, origin features are averaged."""
        g = nx.Graph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('b', x=1.0, y=0.0)
        g.add_edge('a', 'b', w=1.0)
        pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['a', 'b'])})
        cells = pd.DataFrame(
            {'node_id': ['a', 'a', 'b'],   # two cells on node 'a'
             'density': [100.0, 200.0, 50.0]},
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
        )
        u = Utility(cost_coefficient=0.0,
                    origin_features={'density': 1.0})
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        full = add_endpoint_utility(r_u, pairs, u, cells=cells)
        # Origin 'a' has two cells with density 100 and 200; mean = 150.
        # Origin contribution: 1.0 * 150 = 150 (added to every OD from 'a').
        # Both OD pairs from a get +150 over the route utility (which is 0
        # since cost_coefficient=0).
        np.testing.assert_array_almost_equal(
            full.cells_to_cells['a'], np.array([150.0, 150.0]))


class UtilityCompositionTestCase(unittest.TestCase):
    """End-to-end: route_utility + add_endpoint_utility produces the expected
    full utility, including via gravity for accessibility output."""

    def test_full_pipeline_matches_hand_calculation(self):
        """Verify the documented formula holds end-to-end on a hand-checkable example."""
        g = nx.Graph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('b', x=1.0, y=0.0)
        g.add_edge('a', 'b', w=2.0, attr=10.0)
        pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['b'])})
        cells = pd.DataFrame(
            {'node_id': ['a', 'b'],
             'pop': [100.0, 200.0]},
            index=pd.Index(['c_a', 'c_b'], name='cell_id'),
        )
        u = Utility(
            constant=1.0,
            cost_coefficient=-0.5,
            route_features=[RouteFeature('attr', 'attr', coefficient=-0.01, aggregator='sum')],
            origin_features={'pop': 0.001},
            destination_features={'pop': 0.002},
        )
        r_u = route_utility(pairs, g, cost_weight='w', utility=u)
        full = add_endpoint_utility(r_u, pairs, u, cells=cells)
        # a→b: cost=2, attr_sum=10, orig pop=100, dest pop=200.
        # U_route = -0.5 * 2 - 0.01 * 10 = -1 - 0.1 = -1.1
        # U_full = -1.1 + 1 + 0.001 * 100 + 0.002 * 200 = -1.1 + 1 + 0.1 + 0.4 = 0.4
        self.assertAlmostEqual(full.cells_to_cells['a'][0], 0.4)


if __name__ == '__main__':
    unittest.main()
