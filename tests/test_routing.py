"""Tests for routing-layer helpers operating on cost ODMs.

Run with:
    cd src && python -m unittest tests.test_routing
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODNodePairs, TieredODPairs
from aperta.routing import (
    PathAggregation,
    set_min_intrazonal_cost,
    tiered_path_aggregate,
    tiered_path_costs,
)


class SetMinIntrazonalCostTestCase(unittest.TestCase):
    """`set_min_intrazonal_cost` floors every cell-tier cost entry at `min_cost`.

    The floor applies uniformly (not just to self-pairs) to keep cost geometry
    consistent: if intrazonal travel is floored at X, then a longer trip should
    not be allowed to be cheaper than X.
    """

    def _costs(self):
        """Two origins, with the self-pair at cost 0 and other small/large costs."""
        return TieredODNodePairs(
            cells_to_cells={'a': np.array([0., 5., 200.]),
                            'b': np.array([0., 300.])},
            zones_to_zones={'Z': np.array([1500.])},
        )

    def test_scalar_floors_all_entries(self):
        """Floor applies uniformly: every entry below `min_cost` becomes `min_cost`."""
        costs = self._costs()
        out = set_min_intrazonal_cost(costs, min_cost=10.0)
        # 'a': self-pair 0 → 10; close pair 5 → 10; far pair 200 unchanged.
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([10., 10., 200.]))
        # 'b': self-pair 0 → 10; far pair 300 unchanged.
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([10., 300.]))

    def test_entries_above_floor_unchanged(self):
        """Costs already above the floor pass through unchanged."""
        costs = self._costs()
        out = set_min_intrazonal_cost(costs, min_cost=1.0)
        # All non-self-pair entries are >= 1, so only self-pairs (cost 0) get floored.
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([1., 5., 200.]))
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([1., 300.]))

    def test_other_tiers_unchanged(self):
        """zones_to_zones and zones_to_regions pass through untouched."""
        costs = self._costs()
        out = set_min_intrazonal_cost(costs, min_cost=10.0)
        self.assertIs(out.zones_to_zones, costs.zones_to_zones)
        self.assertIsNone(out.zones_to_regions)

    def test_dict_per_origin(self):
        """Per-origin floors apply independently."""
        costs = self._costs()
        out = set_min_intrazonal_cost(costs, min_cost={'a': 10.0, 'b': 50.0})
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([10., 10., 200.]))
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([50., 300.]))

    def test_dict_missing_origin_passes_through(self):
        """Origins absent from the dict get no floor applied."""
        costs = self._costs()
        out = set_min_intrazonal_cost(costs, min_cost={'a': 10.0})  # no 'b'
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([10., 10., 200.]))
        # 'b' unchanged — its self-pair 0 is preserved.
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([0., 300.]))

    def test_series_per_origin(self):
        costs = self._costs()
        s = pd.Series({'a': 10.0, 'b': 50.0})
        out = set_min_intrazonal_cost(costs, min_cost=s)
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([10., 10., 200.]))
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([50., 300.]))

    def test_non_finite_entries_passed_through(self):
        """`inf` / `nan` entries are not floored — they retain their semantics."""
        costs = TieredODNodePairs(
            cells_to_cells={'a': np.array([0., np.inf, np.nan, 200.])},
        )
        out = set_min_intrazonal_cost(costs, min_cost=10.0)
        # 0 → 10; inf and nan preserved; 200 unchanged.
        self.assertEqual(out.cells_to_cells['a'][0], 10.0)
        self.assertTrue(np.isinf(out.cells_to_cells['a'][1]))
        self.assertTrue(np.isnan(out.cells_to_cells['a'][2]))
        self.assertEqual(out.cells_to_cells['a'][3], 200.0)

    def test_returns_copy_not_in_place(self):
        """Original cost arrays must not be mutated."""
        costs = self._costs()
        original_a = costs.cells_to_cells['a'].copy()
        _ = set_min_intrazonal_cost(costs, min_cost=10.0)
        np.testing.assert_array_equal(costs.cells_to_cells['a'], original_a)

    def test_works_with_gravity_to_avoid_inf(self):
        """End-to-end: a small floor lets power-law gravity sum cleanly across all entries."""
        from aperta.accessibility import gravity, power_decay
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([0., 1., 2.])})
        weights = TieredODNodePairs(cells_to_cells={'a': np.array([10., 1., 1.])})
        # Without the floor, c=0 would give 1/0 → inf; the defensive drop in
        # gravity would silently lose the self-pair weight.
        fixed_costs = set_min_intrazonal_cost(costs, min_cost=0.5)
        df = gravity(fixed_costs, {'w': weights}, {'a': None}, power_decay('inv', 1.0))
        # 10 / 0.5 + 1 / 1 + 1 / 2 = 20 + 1 + 0.5 = 21.5
        self.assertAlmostEqual(df.loc['a', ('inv', 'w')], 21.5)


class TieredPathAggregateTestCase(unittest.TestCase):
    """`tiered_path_aggregate` routes shortest paths and aggregates per-edge
    attributes along each realised path. Tested against a small hand-checkable
    graph.
    """

    def _graph(self) -> nx.Graph:
        """Toy graph:
            a -[w=1, attr=10]- b
            b -[w=2, attr=20]- c
            a -[w=10, attr=100]- c   (the long way; shortest a→c is via b)
        """
        g = nx.Graph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('b', x=1.0, y=0.0)
        g.add_node('c', x=2.0, y=0.0)
        g.add_edge('a', 'b', w=1.0, attr=10.0)
        g.add_edge('b', 'c', w=2.0, attr=20.0)
        g.add_edge('a', 'c', w=10.0, attr=100.0)
        return g

    def _pairs(self):
        """One origin 'a' with dests [b, c], cells-only tier."""
        return TieredODNodePairs(
            cells_to_cells={'a': np.array(['a', 'b', 'c'])},
        )

    def test_sum_aggregator_basic(self):
        """Sum of `attr` along the realised shortest path."""
        pairs = self._pairs()
        graph = self._graph()
        agg = [PathAggregation('attr_total', 'attr', 'sum')]
        costs, aggs = tiered_path_aggregate(pairs, graph, weight='w',
                                            aggregations=agg)
        # Self-pair a→a: cost 0, sum over 0 edges = 0.
        self.assertEqual(costs.cells_to_cells['a'][0], 0.0)
        self.assertEqual(aggs['attr_total'].cells_to_cells['a'][0], 0.0)
        # a→b: 1 edge (w=1, attr=10). Cost 1, attr_total 10.
        self.assertEqual(costs.cells_to_cells['a'][1], 1.0)
        self.assertEqual(aggs['attr_total'].cells_to_cells['a'][1], 10.0)
        # a→c via b: 2 edges (a→b: attr=10; b→c: attr=20). Cost 3, attr_total 30.
        self.assertEqual(costs.cells_to_cells['a'][2], 3.0)
        self.assertEqual(aggs['attr_total'].cells_to_cells['a'][2], 30.0)

    def test_cost_matches_tiered_path_costs(self):
        """The cost component of tiered_path_aggregate must match tiered_path_costs."""
        pairs = self._pairs()
        graph = self._graph()
        agg = [PathAggregation('attr', 'attr', 'sum')]
        costs_agg, _ = tiered_path_aggregate(pairs, graph, weight='w',
                                             aggregations=agg)
        costs_only = tiered_path_costs(pairs, graph, weight='w')
        np.testing.assert_array_almost_equal(
            costs_agg.cells_to_cells['a'], costs_only.cells_to_cells['a'])

    def test_mean_aggregator(self):
        """Mean is the arithmetic average of edge attributes along the path."""
        pairs = self._pairs()
        graph = self._graph()
        agg = [PathAggregation('attr_mean', 'attr', 'mean')]
        _, aggs = tiered_path_aggregate(pairs, graph, weight='w',
                                        aggregations=agg)
        # a→a: NaN (no edges to average).
        self.assertTrue(np.isnan(aggs['attr_mean'].cells_to_cells['a'][0]))
        # a→b: single edge of attr=10 → mean = 10.
        self.assertEqual(aggs['attr_mean'].cells_to_cells['a'][1], 10.0)
        # a→c via b: edges attr=10, 20 → mean = 15.
        self.assertEqual(aggs['attr_mean'].cells_to_cells['a'][2], 15.0)

    def test_min_max_aggregators(self):
        """Min/max along the realised path."""
        pairs = self._pairs()
        graph = self._graph()
        agg = [
            PathAggregation('attr_min', 'attr', 'min'),
            PathAggregation('attr_max', 'attr', 'max'),
        ]
        _, aggs = tiered_path_aggregate(pairs, graph, weight='w',
                                        aggregations=agg)
        # a→c via b: edges attr=10, 20.
        self.assertEqual(aggs['attr_min'].cells_to_cells['a'][2], 10.0)
        self.assertEqual(aggs['attr_max'].cells_to_cells['a'][2], 20.0)

    def test_callable_aggregator(self):
        """A custom callable can replace the named aggregator."""
        pairs = self._pairs()
        graph = self._graph()
        # Custom: squared-sum, e.g. for a "squared distance" interpretation.
        agg = [PathAggregation('sq_sum', 'attr',
                                aggregator=lambda arr: float((arr ** 2).sum()))]
        _, aggs = tiered_path_aggregate(pairs, graph, weight='w',
                                        aggregations=agg)
        # a→c via b: 10² + 20² = 500.
        self.assertEqual(aggs['sq_sum'].cells_to_cells['a'][2], 500.0)

    def test_callable_attribute_extractor(self):
        """`attribute` as a callable taking (u, v, data)."""
        pairs = self._pairs()
        graph = self._graph()
        # Custom attribute: 1.0 per edge — counts edges in the path.
        agg = [PathAggregation('edge_count', lambda u, v, d: 1.0, 'sum')]
        _, aggs = tiered_path_aggregate(pairs, graph, weight='w',
                                        aggregations=agg)
        self.assertEqual(aggs['edge_count'].cells_to_cells['a'][0], 0.0)  # self
        self.assertEqual(aggs['edge_count'].cells_to_cells['a'][1], 1.0)  # 1 edge
        self.assertEqual(aggs['edge_count'].cells_to_cells['a'][2], 2.0)  # 2 edges

    def test_multiple_aggregations_one_call(self):
        """Multiple aggregations share the per-origin routing pass."""
        pairs = self._pairs()
        graph = self._graph()
        agg = [
            PathAggregation('total', 'attr', 'sum'),
            PathAggregation('avg', 'attr', 'mean'),
            PathAggregation('worst', 'attr', 'max'),
        ]
        _, aggs = tiered_path_aggregate(pairs, graph, weight='w',
                                        aggregations=agg)
        # All three should be filled for a→c.
        self.assertEqual(aggs['total'].cells_to_cells['a'][2], 30.0)
        self.assertEqual(aggs['avg'].cells_to_cells['a'][2], 15.0)
        self.assertEqual(aggs['worst'].cells_to_cells['a'][2], 20.0)

    def test_unreachable_destination_is_nan(self):
        """Aggregation for an unreachable dest is NaN; cost is inf."""
        # Graph with two disconnected components; 'a' can't reach 'x'.
        g = nx.Graph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('x', x=10.0, y=10.0)
        # No edges between a and x.
        pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['a', 'x'])})
        agg = [PathAggregation('attr_total', 'attr', 'sum')]
        costs, aggs = tiered_path_aggregate(pairs, g, weight='w',
                                            aggregations=agg)
        # Self-pair: cost 0, sum 0.
        self.assertEqual(costs.cells_to_cells['a'][0], 0.0)
        # Unreachable: cost inf, aggregation NaN.
        self.assertTrue(np.isinf(costs.cells_to_cells['a'][1]))
        self.assertTrue(np.isnan(aggs['attr_total'].cells_to_cells['a'][1]))

    def test_multidigraph_picks_min_weight_edge(self):
        """For OSMnx-style MultiDiGraphs, the min-`weight` parallel edge wins."""
        g = nx.MultiDiGraph()
        g.add_node('a', x=0.0, y=0.0)
        g.add_node('b', x=1.0, y=0.0)
        # Two edges a→b: one fast (low weight, high attr), one slow (high weight, low attr).
        g.add_edge('a', 'b', w=1.0, attr=99.0)
        g.add_edge('a', 'b', w=5.0, attr=10.0)
        # Reverse edges (walking is bidirectional).
        g.add_edge('b', 'a', w=1.0, attr=99.0)
        g.add_edge('b', 'a', w=5.0, attr=10.0)
        pairs = TieredODNodePairs(cells_to_cells={'a': np.array(['b'])})
        agg = [PathAggregation('attr_total', 'attr', 'sum')]
        costs, aggs = tiered_path_aggregate(pairs, g, weight='w',
                                            aggregations=agg)
        # Router picks w=1 edge → cost 1, attr 99.
        self.assertEqual(costs.cells_to_cells['a'][0], 1.0)
        self.assertEqual(aggs['attr_total'].cells_to_cells['a'][0], 99.0)

    def test_mask_skips_destinations(self):
        """Mask=False destinations get inf cost and NaN aggregations."""
        pairs = self._pairs()
        graph = self._graph()
        # Mask out the 'c' destination.
        mask = TieredODNodePairs(cells_to_cells={'a': np.array([True, True, False])})
        agg = [PathAggregation('attr_total', 'attr', 'sum')]
        costs, aggs = tiered_path_aggregate(pairs, graph, weight='w',
                                            aggregations=agg, mask=mask)
        self.assertEqual(costs.cells_to_cells['a'][1], 1.0)  # b — routed
        self.assertTrue(np.isinf(costs.cells_to_cells['a'][2]))  # c — masked out
        self.assertTrue(np.isnan(aggs['attr_total'].cells_to_cells['a'][2]))

    def test_empty_aggregations_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            tiered_path_aggregate(self._pairs(), self._graph(),
                                  weight='w', aggregations=[])

    def test_duplicate_aggregation_names_raises(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            tiered_path_aggregate(
                self._pairs(), self._graph(), weight='w',
                aggregations=[
                    PathAggregation('x', 'attr', 'sum'),
                    PathAggregation('x', 'attr', 'mean'),
                ])

    def test_unknown_aggregator_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown aggregator"):
            tiered_path_aggregate(
                self._pairs(), self._graph(), weight='w',
                aggregations=[PathAggregation('x', 'attr', 'nope')])


if __name__ == '__main__':
    unittest.main()
