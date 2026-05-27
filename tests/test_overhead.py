"""Tests for `aperta.overhead` — first/last-mile overhead helpers.

Run with:
    cd src && python -m unittest tests.test_overhead
"""
import unittest

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from aperta.od_pairs import TieredODNodePairs, TieredODPairs
from aperta.overhead import (
    add_node_overheads,
    aggregate_dest_overhead_per_group_euclidean,
    aggregate_dest_overhead_per_group_routed,
    aggregate_dest_overhead_per_node,
)


class AggregateDestOverheadPerNodeTestCase(unittest.TestCase):
    """`aggregate_dest_overhead_per_node` — mean of per-cell overheads across
    cells sharing each network node.
    """

    def test_basic_mean(self):
        cells = pd.DataFrame(
            {'node_id': ['a', 'a', 'b'],
             'overhead_s': [10.0, 30.0, 50.0]},
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
        )
        out = aggregate_dest_overhead_per_node(cells, 'overhead_s')
        # Node a: mean(10, 30) = 20.
        # Node b: mean(50) = 50.
        self.assertEqual(out.loc['a'], 20.0)
        self.assertEqual(out.loc['b'], 50.0)

    def test_single_cell_per_node(self):
        cells = pd.DataFrame(
            {'node_id': ['a', 'b', 'c'],
             'overhead_s': [10.0, 20.0, 30.0]},
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
        )
        out = aggregate_dest_overhead_per_node(cells, 'overhead_s')
        # 1-cell-per-node: mean = the value itself.
        self.assertEqual(out.loc['a'], 10.0)
        self.assertEqual(out.loc['b'], 20.0)
        self.assertEqual(out.loc['c'], 30.0)

    def test_weighted_mean(self):
        cells = pd.DataFrame(
            {'node_id': ['a', 'a', 'b'],
             'overhead_s': [10.0, 30.0, 50.0],
             'pop': [1.0, 9.0, 1.0]},
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
        )
        out = aggregate_dest_overhead_per_node(
            cells, 'overhead_s', weight_column='pop')
        # Node a: (10*1 + 30*9) / (1 + 9) = (10 + 270) / 10 = 28.
        # Node b: (50*1) / 1 = 50.
        self.assertEqual(out.loc['a'], 28.0)
        self.assertEqual(out.loc['b'], 50.0)

    def test_cells_with_no_node_dropped(self):
        cells = pd.DataFrame(
            {'node_id': ['a', None, 'b'],
             'overhead_s': [10.0, 99.0, 20.0]},
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
        )
        out = aggregate_dest_overhead_per_node(cells, 'overhead_s')
        self.assertEqual(out.loc['a'], 10.0)  # 99-overhead cell ignored
        self.assertEqual(out.loc['b'], 20.0)
        self.assertEqual(len(out), 2)


class AggregateDestOverheadPerGroupRoutedTestCase(unittest.TestCase):
    """`aggregate_dest_overhead_per_group_routed` — per-zone/region destination
    overhead via routing. Reserved for transit-style use cases.
    """

    def _graph(self) -> nx.Graph:
        """Toy graph:
            n1 -[w=10]- n2 -[w=20]- n3
        """
        g = nx.Graph()
        g.add_node('n1', x=0.0, y=0.0)
        g.add_node('n2', x=1.0, y=0.0)
        g.add_node('n3', x=2.0, y=0.0)
        g.add_edge('n1', 'n2', w=10.0)
        g.add_edge('n2', 'n3', w=20.0)
        return g

    def _cells_in_zones(self) -> pd.DataFrame:
        """Three cells: c1, c2 in zone Z; c3 in zone W."""
        return pd.DataFrame(
            {'node_id': ['n1', 'n3', 'n3'],
             'zone_id': ['Z', 'Z', 'W'],
             'first_mile': [5.0, 7.0, 3.0]},
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
        )

    def _zones(self) -> pd.DataFrame:
        """Zone Z has representative node n2; zone W has n3."""
        return pd.DataFrame(
            {'node_id': ['n2', 'n3']},
            index=pd.Index(['Z', 'W'], name='zone_id'),
        )

    def test_basic_routed_average(self):
        """For each zone, mean of route(g_node, c_node) across constituent cells."""
        out = aggregate_dest_overhead_per_group_routed(
            self._cells_in_zones(), self._zones(), self._graph(), weight='w',
            group_id_column='zone_id',
        )
        # Zone Z (g=n2): cells c1 (n1), c2 (n3). Distances: n2→n1=10, n2→n3=20.
        # Mean = 15.
        self.assertEqual(out.loc['Z'], 15.0)
        # Zone W (g=n3): cell c3 (n3). Distance n3→n3 = 0. Mean = 0.
        self.assertEqual(out.loc['W'], 0.0)

    def test_with_cell_overhead(self):
        """The cell first-mile is added to the routed distance before averaging."""
        out = aggregate_dest_overhead_per_group_routed(
            self._cells_in_zones(), self._zones(), self._graph(), weight='w',
            group_id_column='zone_id', cell_overhead_column='first_mile',
        )
        # Zone Z: c1 = 10 + 5 = 15; c2 = 20 + 7 = 27. Mean = 21.
        # Zone W: c3 = 0 + 3 = 3.
        self.assertEqual(out.loc['Z'], 21.0)
        self.assertEqual(out.loc['W'], 3.0)

    def test_weighted_routed_average(self):
        """Weighted average of routed distances + first-mile."""
        cells = self._cells_in_zones().assign(pop=[10.0, 30.0, 1.0])
        out = aggregate_dest_overhead_per_group_routed(
            cells, self._zones(), self._graph(), weight='w',
            group_id_column='zone_id', cell_overhead_column='first_mile',
            weight_column='pop',
        )
        # Zone Z: per-cell (route + first_mile): c1=15 (w=10), c2=27 (w=30).
        # Weighted mean = (15*10 + 27*30) / (10 + 30) = (150 + 810) / 40 = 24.
        self.assertEqual(out.loc['Z'], 24.0)

    def test_group_with_no_cells_is_nan(self):
        """A group in target_groups with no matching cells gets NaN."""
        zones = self._zones().reindex(['Z', 'W', 'Empty'])
        zones.loc['Empty', 'node_id'] = 'n1'
        out = aggregate_dest_overhead_per_group_routed(
            self._cells_in_zones(), zones, self._graph(), weight='w',
            group_id_column='zone_id',
        )
        self.assertNotIn('Empty', out.index)  # skipped — no cells in group

    def test_same_function_works_for_regions(self):
        """The function is tier-agnostic — pass group_id_column='region_id'
        and a regions DataFrame to use it for regions."""
        cells = pd.DataFrame(
            {'node_id': ['n1', 'n3'],
             'region_id': ['R', 'R']},
            index=pd.Index(['c1', 'c2'], name='cell_id'),
        )
        regions = pd.DataFrame(
            {'node_id': ['n2']},
            index=pd.Index(['R'], name='region_id'),
        )
        out = aggregate_dest_overhead_per_group_routed(
            cells, regions, self._graph(), weight='w',
            group_id_column='region_id',
        )
        # Region R (g=n2): n2→n1=10, n2→n3=20. Mean = 15.
        self.assertEqual(out.loc['R'], 15.0)


class AggregateDestOverheadPerGroupEuclideanTestCase(unittest.TestCase):
    """`aggregate_dest_overhead_per_group_euclidean` — Euclidean-distance-based
    per-group destination overhead. For road-network destinations where users
    don't actually pass through a specific representative node.
    """

    def _cells_and_zones(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Three cells (each a 100m square) in two zones.
            zone Z covers x ∈ [0, 200], centroid at (100, 50)
                cell c1 at (50, 50)  → distance to Z centroid = 50
                cell c2 at (150, 50) → distance to Z centroid = 50
            zone W covers x ∈ [200, 300], centroid at (250, 50)
                cell c3 at (250, 50) → distance to W centroid = 0
        """
        cells = gpd.GeoDataFrame(
            {'zone_id': ['Z', 'Z', 'W'],
             'first_mile': [10.0, 20.0, 5.0]},
            geometry=[box(0, 0, 100, 100),
                      box(100, 0, 200, 100),
                      box(200, 0, 300, 100)],
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
            crs='EPSG:2056',
        )
        zones = gpd.GeoDataFrame(
            geometry=[box(0, 0, 200, 100),
                      box(200, 0, 300, 100)],
            index=pd.Index(['Z', 'W'], name='zone_id'),
            crs='EPSG:2056',
        )
        return cells, zones

    def test_basic_euclidean_average(self):
        """For each zone, mean Euclidean distance from cell centroids to zone
        centroid, divided by speed."""
        cells, zones = self._cells_and_zones()
        out = aggregate_dest_overhead_per_group_euclidean(
            cells, zones, speed=1.0, group_id_column='zone_id',
        )
        # Z: cells at (50,50), (150,50); centroid at (100,50). Distances: 50, 50.
        # Mean = 50 (with speed=1, time == distance).
        self.assertAlmostEqual(out.loc['Z'], 50.0)
        # W: cell at (250,50); centroid at (250,50). Distance = 0.
        self.assertAlmostEqual(out.loc['W'], 0.0)

    def test_speed_scaling(self):
        """Doubling the speed halves the overhead time."""
        cells, zones = self._cells_and_zones()
        out_slow = aggregate_dest_overhead_per_group_euclidean(
            cells, zones, speed=1.0, group_id_column='zone_id',
        )
        out_fast = aggregate_dest_overhead_per_group_euclidean(
            cells, zones, speed=2.0, group_id_column='zone_id',
        )
        self.assertAlmostEqual(out_fast.loc['Z'], out_slow.loc['Z'] / 2.0)
        self.assertAlmostEqual(out_fast.loc['W'], out_slow.loc['W'] / 2.0)

    def test_with_cell_overhead(self):
        """`cell_overhead_column` is added to the Euclidean time."""
        cells, zones = self._cells_and_zones()
        out = aggregate_dest_overhead_per_group_euclidean(
            cells, zones, speed=1.0,
            group_id_column='zone_id', cell_overhead_column='first_mile',
        )
        # Z: c1 = 50 + 10 = 60; c2 = 50 + 20 = 70. Mean = 65.
        # W: c3 = 0 + 5 = 5.
        self.assertAlmostEqual(out.loc['Z'], 65.0)
        self.assertAlmostEqual(out.loc['W'], 5.0)

    def test_weighted_mean(self):
        """Weight column scales the per-cell contribution to the average."""
        cells, zones = self._cells_and_zones()
        cells = cells.assign(pop=[10.0, 30.0, 1.0])
        out = aggregate_dest_overhead_per_group_euclidean(
            cells, zones, speed=1.0,
            group_id_column='zone_id', cell_overhead_column='first_mile',
            weight_column='pop',
        )
        # Z: per-cell (euclid + first_mile): c1=60 (w=10), c2=70 (w=30).
        # Weighted mean = (60*10 + 70*30) / (10 + 30) = (600 + 2100) / 40 = 67.5.
        self.assertAlmostEqual(out.loc['Z'], 67.5)

    def test_empty_group_is_nan(self):
        """A group with no cells gets NaN (reindex fills missing)."""
        cells, zones = self._cells_and_zones()
        zones = zones.reindex(['Z', 'W', 'Empty'])
        # 'Empty' zone has no cells assigned to it.
        out = aggregate_dest_overhead_per_group_euclidean(
            cells, zones, speed=1.0, group_id_column='zone_id',
        )
        self.assertTrue(pd.isna(out.loc['Empty']))

    def test_zero_or_negative_speed_raises(self):
        cells, zones = self._cells_and_zones()
        with self.assertRaisesRegex(ValueError, "speed"):
            aggregate_dest_overhead_per_group_euclidean(
                cells, zones, speed=0.0, group_id_column='zone_id',
            )
        with self.assertRaisesRegex(ValueError, "speed"):
            aggregate_dest_overhead_per_group_euclidean(
                cells, zones, speed=-1.0, group_id_column='zone_id',
            )

    def test_same_function_works_for_regions(self):
        """Tier-agnostic — works for regions by passing group_id_column='region_id'."""
        cells = gpd.GeoDataFrame(
            {'region_id': ['R', 'R']},
            geometry=[Point(0, 0), Point(100, 0)],
            index=pd.Index(['c1', 'c2'], name='cell_id'),
            crs='EPSG:2056',
        )
        regions = gpd.GeoDataFrame(
            geometry=[Point(50, 0)],
            index=pd.Index(['R'], name='region_id'),
            crs='EPSG:2056',
        )
        out = aggregate_dest_overhead_per_group_euclidean(
            cells, regions, speed=1.0, group_id_column='region_id',
        )
        # R: c1→R = 50, c2→R = 50. Mean = 50.
        self.assertAlmostEqual(out.loc['R'], 50.0)


class AddNodeOverheadsTestCase(unittest.TestCase):
    """`add_node_overheads` — adds per-node overheads to a TieredODPairs of costs."""

    def _setup(self):
        pairs = TieredODNodePairs(
            cells_to_cells={'a': np.array(['a', 'b', 'c']),
                            'b': np.array(['a', 'b'])},
            cells_to_zones={'a': np.array(['ZB']),
                            'b': np.array(['ZA'])},
            zones_to_zones={'ZA': np.array(['ZB']),
                            'ZB': np.array(['ZA'])},
        )
        costs = TieredODNodePairs(
            cells_to_cells={'a': np.array([0.0, 10.0, 20.0]),
                            'b': np.array([10.0, 0.0])},
            cells_to_zones={'a': np.array([300.0]),
                            'b': np.array([300.0])},
            zones_to_zones={'ZA': np.array([100.0]),
                            'ZB': np.array([100.0])},
        )
        return pairs, costs
    def test_origin_only(self):
        """Origin overhead is added to every OD pair from that origin, all tiers."""
        pairs, costs = self._setup()
        origin = pd.Series({'a': 3.0, 'b': 5.0, 'ZA': 100.0, 'ZB': 200.0})
        out = add_node_overheads(costs, pairs, origin=origin)
        # cells_to_cells: a gets +3 added to every entry; b gets +5.
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([3.0, 13.0, 23.0]))
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([15.0, 5.0]))
        # cells_to_zones: cell-node origins → a +3, b +5.
        np.testing.assert_array_equal(out.cells_to_zones['a'], np.array([303.0]))
        np.testing.assert_array_equal(out.cells_to_zones['b'], np.array([305.0]))
        # zones_to_zones: zone-node origins → ZA +100, ZB +200.
        np.testing.assert_array_equal(out.zones_to_zones['ZA'], np.array([200.0]))
        np.testing.assert_array_equal(out.zones_to_zones['ZB'], np.array([300.0]))
    def test_dest_cell_only(self):
        """Destination cell-tier overhead is added per cell-tier destination."""
        pairs, costs = self._setup()
        dest_cell = pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0})
        out = add_node_overheads(costs, pairs, dest_cell=dest_cell)
        # a's dests: [a, b, c] → adds [1, 2, 3]. Costs: [0, 10, 20] → [1, 12, 23].
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([1.0, 12.0, 23.0]))
        # b's dests: [a, b] → adds [1, 2]. Costs: [10, 0] → [11, 2].
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([11.0, 2.0]))
        # Other tiers unchanged (no dest_zone given).
        np.testing.assert_array_equal(out.cells_to_zones['a'], np.array([300.0]))
        np.testing.assert_array_equal(out.zones_to_zones['ZA'], np.array([100.0]))
    def test_per_tier_dest_overheads(self):
        """dest_cell and dest_zone apply independently per tier; dest_zone
        applies to BOTH cells_to_zones and zones_to_zones (both have zone dests)."""
        pairs, costs = self._setup()
        out = add_node_overheads(
            costs, pairs,
            dest_cell={'a': 1.0, 'b': 2.0, 'c': 3.0},
            dest_zone={'ZA': 50.0, 'ZB': 60.0},
        )
        # cells: a [0,10,20] + dest [1,2,3] = [1, 12, 23].
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([1.0, 12.0, 23.0]))
        # c2z: a's dest is ZB → +60. cost 300 → 360.
        np.testing.assert_array_equal(out.cells_to_zones['a'], np.array([360.0]))
        # c2z: b's dest is ZA → +50. cost 300 → 350.
        np.testing.assert_array_equal(out.cells_to_zones['b'], np.array([350.0]))
        # z2z: ZA's dest is ZB → +60. cost 100 → 160.
        np.testing.assert_array_equal(out.zones_to_zones['ZA'], np.array([160.0]))
    def test_origin_and_destination_combine(self):
        """Origin and destination overheads add independently."""
        pairs, costs = self._setup()
        out = add_node_overheads(
            costs, pairs,
            origin={'a': 3.0, 'b': 5.0},
            dest_cell={'a': 1.0, 'b': 2.0, 'c': 3.0},
        )
        # a: cost [0,10,20] + origin 3 + dest [1,2,3] = [4, 15, 26].
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([4.0, 15.0, 26.0]))
    def test_missing_keys_get_zero(self):
        """Nodes absent from a lookup contribute 0 overhead."""
        pairs, costs = self._setup()
        out = add_node_overheads(
            costs, pairs,
            origin={'a': 3.0},  # 'b' missing → 0
            dest_cell={'b': 2.0},  # 'a' and 'c' missing → 0
        )
        # a: cost [0,10,20] + origin 3 + dest [0, 2, 0] = [3, 15, 23].
        np.testing.assert_array_equal(out.cells_to_cells['a'], np.array([3.0, 15.0, 23.0]))
        # b: cost [10,0] + origin 0 + dest [0, 2] = [10, 2].
        np.testing.assert_array_equal(out.cells_to_cells['b'], np.array([10.0, 2.0]))
    def test_all_none_is_no_op(self):
        """No overheads provided → costs returned unchanged (but as a new TieredODPairs)."""
        pairs, costs = self._setup()
        out = add_node_overheads(costs, pairs)
        np.testing.assert_array_equal(
            out.cells_to_cells['a'], costs.cells_to_cells['a'])
        np.testing.assert_array_equal(
            out.zones_to_zones['ZA'], costs.zones_to_zones['ZA'])
        # Must be a copy, not the same object.
        self.assertIsNot(out.cells_to_cells['a'], costs.cells_to_cells['a'])
    def test_dict_or_series_accepted(self):
        """Both dict and Series work for any kwarg."""
        pairs, costs = self._setup()
        a = add_node_overheads(costs, pairs, dest_cell={'a': 1.0, 'b': 2.0, 'c': 3.0})
        b = add_node_overheads(costs, pairs,
                                dest_cell=pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}))
        np.testing.assert_array_equal(a.cells_to_cells['a'], b.cells_to_cells['a'])
    def test_input_costs_not_mutated(self):
        """Input cost TieredODPairs is not modified by add_node_overheads."""
        pairs, costs = self._setup()
        original_a = costs.cells_to_cells['a'].copy()
        _ = add_node_overheads(costs, pairs, origin={'a': 10.0}, dest_cell={'b': 5.0})
        np.testing.assert_array_equal(costs.cells_to_cells['a'], original_a)


if __name__ == '__main__':
    unittest.main()
