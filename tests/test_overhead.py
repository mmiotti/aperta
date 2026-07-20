"""Tests for `aperta.overhead` — first/last-mile overhead helpers.

Run with:
    python -m unittest tests.test_overhead
"""

import unittest

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from aperta.od_pairs import TieredODGeoPairs
from aperta.overhead import (
    add_geo_overheads,
    aggregate_dest_overhead_per_group,
    aggregate_dest_overhead_per_node,
)


class AggregateDestOverheadPerNodeTestCase(unittest.TestCase):
    """`aggregate_dest_overhead_per_node` — mean of per-cell overheads across
    cells sharing each network node.
    """

    def test_basic_mean(self):
        cells = pd.DataFrame(
            {"node_id": ["a", "a", "b"], "overhead_s": [10.0, 30.0, 50.0]},
            index=pd.Index(["c1", "c2", "c3"], name="cell_id"),
        )
        out = aggregate_dest_overhead_per_node(cells, "overhead_s")
        # Node a: mean(10, 30) = 20.
        # Node b: mean(50) = 50.
        self.assertEqual(out.loc["a"], 20.0)
        self.assertEqual(out.loc["b"], 50.0)

    def test_single_cell_per_node(self):
        cells = pd.DataFrame(
            {"node_id": ["a", "b", "c"], "overhead_s": [10.0, 20.0, 30.0]},
            index=pd.Index(["c1", "c2", "c3"], name="cell_id"),
        )
        out = aggregate_dest_overhead_per_node(cells, "overhead_s")
        # 1-cell-per-node: mean = the value itself.
        self.assertEqual(out.loc["a"], 10.0)
        self.assertEqual(out.loc["b"], 20.0)
        self.assertEqual(out.loc["c"], 30.0)

    def test_weighted_mean(self):
        cells = pd.DataFrame(
            {"node_id": ["a", "a", "b"], "overhead_s": [10.0, 30.0, 50.0], "pop": [1.0, 9.0, 1.0]},
            index=pd.Index(["c1", "c2", "c3"], name="cell_id"),
        )
        out = aggregate_dest_overhead_per_node(cells, "overhead_s", weight_column="pop")
        # Node a: (10*1 + 30*9) / (1 + 9) = (10 + 270) / 10 = 28.
        # Node b: (50*1) / 1 = 50.
        self.assertEqual(out.loc["a"], 28.0)
        self.assertEqual(out.loc["b"], 50.0)

    def test_cells_with_no_node_dropped(self):
        cells = pd.DataFrame(
            {"node_id": ["a", None, "b"], "overhead_s": [10.0, 99.0, 20.0]},
            index=pd.Index(["c1", "c2", "c3"], name="cell_id"),
        )
        out = aggregate_dest_overhead_per_node(cells, "overhead_s")
        self.assertEqual(out.loc["a"], 10.0)  # 99-overhead cell ignored
        self.assertEqual(out.loc["b"], 20.0)
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
        g.add_node("n1", x=0.0, y=0.0)
        g.add_node("n2", x=1.0, y=0.0)
        g.add_node("n3", x=2.0, y=0.0)
        g.add_edge("n1", "n2", w=10.0)
        g.add_edge("n2", "n3", w=20.0)
        return g

    def _cells_in_zones(self) -> pd.DataFrame:
        """Three cells: c1, c2 in zone Z; c3 in zone W."""
        return pd.DataFrame(
            {
                "node_id": ["n1", "n3", "n3"],
                "zone_id": ["Z", "Z", "W"],
                "first_mile": [5.0, 7.0, 3.0],
            },
            index=pd.Index(["c1", "c2", "c3"], name="cell_id"),
        )

    def _zones(self) -> pd.DataFrame:
        """Zone Z has representative node n2; zone W has n3."""
        return pd.DataFrame(
            {"node_id": ["n2", "n3"]},
            index=pd.Index(["Z", "W"], name="zone_id"),
        )

    def test_basic_routed_average(self):
        """For each zone, mean of route(g_node, c_node) across constituent cells."""
        out = aggregate_dest_overhead_per_group(
            self._cells_in_zones(),
            self._zones(),
            distance="routed",
            graph=self._graph(),
            weight="w",
            group_id_column="zone_id",
        )
        # Zone Z (g=n2): cells c1 (n1), c2 (n3). Distances: n2→n1=10, n2→n3=20.
        # Mean = 15.
        self.assertEqual(out.loc["Z"], 15.0)
        # Zone W (g=n3): cell c3 (n3). Distance n3→n3 = 0. Mean = 0.
        self.assertEqual(out.loc["W"], 0.0)

    def test_with_cell_overhead(self):
        """The cell first-mile is added to the routed distance before averaging."""
        out = aggregate_dest_overhead_per_group(
            self._cells_in_zones(),
            self._zones(),
            distance="routed",
            graph=self._graph(),
            weight="w",
            group_id_column="zone_id",
            cell_overhead_column="first_mile",
        )
        # Zone Z: c1 = 10 + 5 = 15; c2 = 20 + 7 = 27. Mean = 21.
        # Zone W: c3 = 0 + 3 = 3.
        self.assertEqual(out.loc["Z"], 21.0)
        self.assertEqual(out.loc["W"], 3.0)

    def test_weighted_routed_average(self):
        """Weighted average of routed distances + first-mile."""
        cells = self._cells_in_zones().assign(pop=[10.0, 30.0, 1.0])
        out = aggregate_dest_overhead_per_group(
            cells,
            self._zones(),
            distance="routed",
            graph=self._graph(),
            weight="w",
            group_id_column="zone_id",
            cell_overhead_column="first_mile",
            weight_column="pop",
        )
        # Zone Z: per-cell (route + first_mile): c1=15 (w=10), c2=27 (w=30).
        # Weighted mean = (15*10 + 27*30) / (10 + 30) = (150 + 810) / 40 = 24.
        self.assertEqual(out.loc["Z"], 24.0)

    def test_group_with_no_cells_is_nan(self):
        """A group in target_groups with no matching cells gets NaN."""
        zones = self._zones().reindex(["Z", "W", "Empty"])
        zones.loc["Empty", "node_id"] = "n1"
        out = aggregate_dest_overhead_per_group(
            self._cells_in_zones(),
            zones,
            distance="routed",
            graph=self._graph(),
            weight="w",
            group_id_column="zone_id",
        )
        self.assertTrue(pd.isna(out.loc["Empty"]))

    def test_tier_agnostic_for_any_grouping(self):
        """The function works for any grouping, not just zone-tier — pass
        whichever `group_id_column` and group-DataFrame the caller has."""
        cells = pd.DataFrame(
            {"node_id": ["n1", "n3"], "group_id": ["G", "G"]},
            index=pd.Index(["c1", "c2"], name="cell_id"),
        )
        groups = pd.DataFrame(
            {"node_id": ["n2"]},
            index=pd.Index(["G"], name="group_id"),
        )
        out = aggregate_dest_overhead_per_group(
            cells,
            groups,
            distance="routed",
            graph=self._graph(),
            weight="w",
            group_id_column="group_id",
        )
        # Group G (g=n2): n2→n1=10, n2→n3=20. Mean = 15.
        self.assertEqual(out.loc["G"], 15.0)


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
            {"zone_id": ["Z", "Z", "W"], "first_mile": [10.0, 20.0, 5.0]},
            geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(200, 0, 300, 100)],
            index=pd.Index(["c1", "c2", "c3"], name="cell_id"),
            crs="EPSG:2056",
        )
        zones = gpd.GeoDataFrame(
            geometry=[box(0, 0, 200, 100), box(200, 0, 300, 100)],
            index=pd.Index(["Z", "W"], name="zone_id"),
            crs="EPSG:2056",
        )
        return cells, zones

    def test_basic_euclidean_average(self):
        """For each zone, mean Euclidean distance from cell centroids to zone
        centroid, divided by speed."""
        cells, zones = self._cells_and_zones()
        out = aggregate_dest_overhead_per_group(
            cells,
            zones,
            distance="euclidean",
            speed=1.0,
            group_id_column="zone_id",
        )
        # Z: cells at (50,50), (150,50); centroid at (100,50). Distances: 50, 50.
        # Mean = 50 (with speed=1, time == distance).
        self.assertAlmostEqual(out.loc["Z"], 50.0)
        # W: cell at (250,50); centroid at (250,50). Distance = 0.
        self.assertAlmostEqual(out.loc["W"], 0.0)

    def test_speed_scaling(self):
        """Doubling the speed halves the overhead time."""
        cells, zones = self._cells_and_zones()
        out_slow = aggregate_dest_overhead_per_group(
            cells,
            zones,
            distance="euclidean",
            speed=1.0,
            group_id_column="zone_id",
        )
        out_fast = aggregate_dest_overhead_per_group(
            cells,
            zones,
            distance="euclidean",
            speed=2.0,
            group_id_column="zone_id",
        )
        self.assertAlmostEqual(out_fast.loc["Z"], out_slow.loc["Z"] / 2.0)
        self.assertAlmostEqual(out_fast.loc["W"], out_slow.loc["W"] / 2.0)

    def test_with_cell_overhead(self):
        """`cell_overhead_column` is added to the Euclidean time."""
        cells, zones = self._cells_and_zones()
        out = aggregate_dest_overhead_per_group(
            cells,
            zones,
            distance="euclidean",
            speed=1.0,
            group_id_column="zone_id",
            cell_overhead_column="first_mile",
        )
        # Z: c1 = 50 + 10 = 60; c2 = 50 + 20 = 70. Mean = 65.
        # W: c3 = 0 + 5 = 5.
        self.assertAlmostEqual(out.loc["Z"], 65.0)
        self.assertAlmostEqual(out.loc["W"], 5.0)

    def test_weighted_mean(self):
        """Weight column scales the per-cell contribution to the average."""
        cells, zones = self._cells_and_zones()
        cells = cells.assign(pop=[10.0, 30.0, 1.0])
        out = aggregate_dest_overhead_per_group(
            cells,
            zones,
            distance="euclidean",
            speed=1.0,
            group_id_column="zone_id",
            cell_overhead_column="first_mile",
            weight_column="pop",
        )
        # Z: per-cell (euclid + first_mile): c1=60 (w=10), c2=70 (w=30).
        # Weighted mean = (60*10 + 70*30) / (10 + 30) = (600 + 2100) / 40 = 67.5.
        self.assertAlmostEqual(out.loc["Z"], 67.5)

    def test_empty_group_is_nan(self):
        """A group with no cells gets NaN (reindex fills missing)."""
        cells, zones = self._cells_and_zones()
        zones = zones.reindex(["Z", "W", "Empty"])
        # 'Empty' zone has no cells assigned to it.
        out = aggregate_dest_overhead_per_group(
            cells,
            zones,
            distance="euclidean",
            speed=1.0,
            group_id_column="zone_id",
        )
        self.assertTrue(pd.isna(out.loc["Empty"]))

    def test_zero_or_negative_speed_raises(self):
        cells, zones = self._cells_and_zones()
        with self.assertRaisesRegex(ValueError, "speed"):
            aggregate_dest_overhead_per_group(
                cells,
                zones,
                distance="euclidean",
                speed=0.0,
                group_id_column="zone_id",
            )
        with self.assertRaisesRegex(ValueError, "speed"):
            aggregate_dest_overhead_per_group(
                cells,
                zones,
                distance="euclidean",
                speed=-1.0,
                group_id_column="zone_id",
            )

    def test_tier_agnostic_for_any_grouping(self):
        """Tier-agnostic — works for any grouping by passing the matching
        `group_id_column` and group-DataFrame."""
        cells = gpd.GeoDataFrame(
            {"group_id": ["G", "G"]},
            geometry=[Point(0, 0), Point(100, 0)],
            index=pd.Index(["c1", "c2"], name="cell_id"),
            crs="EPSG:2056",
        )
        groups = gpd.GeoDataFrame(
            geometry=[Point(50, 0)],
            index=pd.Index(["G"], name="group_id"),
            crs="EPSG:2056",
        )
        out = aggregate_dest_overhead_per_group(
            cells,
            groups,
            distance="euclidean",
            speed=1.0,
            group_id_column="group_id",
        )
        # G: c1→G = 50, c2→G = 50. Mean = 50.
        self.assertAlmostEqual(out.loc["G"], 50.0)


class AddGeoOverheadsTestCase(unittest.TestCase):
    """`add_geo_overheads` applies per-tier overheads to a geo-keyed cost ODM
    and, crucially, AUTO-DERIVES zone-tier overheads from cell-tier overheads
    + `cell_to_zone` when the costs include c2z or z2z. Without auto-derivation,
    z2z OD pairs would silently carry zero overhead while c2c pairs carry 2×
    cell overhead, producing visible origin-zone outlines in accessibility maps.
    """

    def _setup(self):
        """Two cells per zone, two zones. Costs at zero everywhere so the
        per-tier overhead additions are directly readable in the result.
        """
        # Geo IDs: cells = c1..c4, zones = Z1 (holds c1, c2), Z2 (holds c3, c4).
        # cells_to_cells: from c1, dests = [c1, c2]   (route costs all 0)
        # cells_to_zones: from c1, dests = [Z2]      (route cost 0)
        # zones_to_zones: from Z1, dests = [Z2]      (route cost 0)
        pairs = TieredODGeoPairs(
            cells_to_cells={"c1": np.array(["c1", "c2"], dtype=object)},
            cells_to_zones={"c1": np.array(["Z2"], dtype=object)},
            zones_to_zones={"Z1": np.array(["Z2"], dtype=object)},
        )
        costs = TieredODGeoPairs(
            cells_to_cells={"c1": np.array([0.0, 0.0], dtype=np.float32)},
            cells_to_zones={"c1": np.array([0.0], dtype=np.float32)},
            zones_to_zones={"Z1": np.array([0.0], dtype=np.float32)},
        )
        overhead_per_cell = pd.Series({"c1": 10.0, "c2": 20.0, "c3": 30.0, "c4": 40.0})
        cell_to_zone = pd.Series({"c1": "Z1", "c2": "Z1", "c3": "Z2", "c4": "Z2"})
        return pairs, costs, overhead_per_cell, cell_to_zone

    def test_zone_overheads_auto_derived_from_cell_overheads(self):
        """When `origin_zone` / `dest_zone` are absent but `origin_cell` /
        `dest_cell` + `cell_to_zone` are given, zone overheads are computed
        as the mean of cell overheads per zone."""
        pairs, costs, oh_per_cell, c2z = self._setup()
        result = add_geo_overheads(
            costs,
            pairs,
            origin_cell=oh_per_cell,
            dest_cell=oh_per_cell,
            cell_to_zone=c2z,
        )
        # c2c: origin c1 (=10) + dest c1 (=10), origin c1 (=10) + dest c2 (=20).
        np.testing.assert_array_equal(result.cells_to_cells["c1"], [20.0, 30.0])
        # c2z: origin c1 (=10) + dest Z2 (=mean of c3, c4 = 35).
        np.testing.assert_array_equal(result.cells_to_zones["c1"], [45.0])
        # z2z: origin Z1 (=mean of c1, c2 = 15) + dest Z2 (=35).
        np.testing.assert_array_equal(result.zones_to_zones["Z1"], [50.0])

    def test_missing_cell_to_zone_raises_when_zone_tier_needed(self):
        """If origin_cell is given but no cell_to_zone and z2z tier present,
        raise rather than silently apply zero overhead to z2z."""
        pairs, costs, oh_per_cell, _ = self._setup()
        with self.assertRaisesRegex(ValueError, "cell_to_zone"):
            add_geo_overheads(
                costs,
                pairs,
                origin_cell=oh_per_cell,
                dest_cell=oh_per_cell,
                # cell_to_zone deliberately omitted
            )

    def test_explicit_zone_overhead_overrides_auto_derivation(self):
        """An explicit `origin_zone` / `dest_zone` skips auto-derivation
        (cell_to_zone is not consulted for those sides)."""
        pairs, costs, oh_per_cell, c2z = self._setup()
        explicit_origin_zone = pd.Series({"Z1": 999.0})
        explicit_dest_zone = pd.Series({"Z2": 888.0})
        result = add_geo_overheads(
            costs,
            pairs,
            origin_cell=oh_per_cell,
            dest_cell=oh_per_cell,
            origin_zone=explicit_origin_zone,
            dest_zone=explicit_dest_zone,
            cell_to_zone=c2z,  # not consulted for origin or dest zones
        )
        # z2z: explicit origin (999) + explicit dest (888) = 1887
        np.testing.assert_array_equal(result.zones_to_zones["Z1"], [1887.0])

    def test_single_tier_cells_only_no_cell_to_zone_required(self):
        """If costs has only cells_to_cells (no c2z or z2z), `cell_to_zone`
        is NOT required — single-tier usage stays simple."""
        pairs = TieredODGeoPairs(
            cells_to_cells={"c1": np.array(["c1", "c2"], dtype=object)},
        )
        costs = TieredODGeoPairs(
            cells_to_cells={"c1": np.array([5.0, 7.0], dtype=np.float32)},
        )
        oh_per_cell = pd.Series({"c1": 10.0, "c2": 20.0})
        # No cell_to_zone needed; no zone tiers exist, so nothing to derive.
        result = add_geo_overheads(
            costs,
            pairs,
            origin_cell=oh_per_cell,
            dest_cell=oh_per_cell,
        )
        # c2c only: 5 + (origin 10 + dest 10) = 25, 7 + (origin 10 + dest 20) = 37
        np.testing.assert_array_equal(result.cells_to_cells["c1"], [25.0, 37.0])

    def test_zone_with_no_constituent_cells_raises(self):
        """A zone referenced in costs but absent from cell_to_zone's image
        raises a clear error rather than silently producing zero overhead."""
        pairs, costs, oh_per_cell, _ = self._setup()
        # cell_to_zone maps c1, c2 to Z1 — c3, c4 to a third zone Z3, not Z2.
        # Z2 is referenced as a destination in pairs but has no constituent cells.
        bad_c2z = pd.Series({"c1": "Z1", "c2": "Z1", "c3": "Z3", "c4": "Z3"})
        with self.assertRaisesRegex(ValueError, "no constituent cells"):
            add_geo_overheads(
                costs,
                pairs,
                origin_cell=oh_per_cell,
                dest_cell=oh_per_cell,
                cell_to_zone=bad_c2z,
            )

    def test_cell_missing_from_cell_to_zone_raises(self):
        """A cell in cell_overhead but missing from cell_to_zone raises."""
        pairs, costs, oh_per_cell, _ = self._setup()
        partial_c2z = pd.Series({"c1": "Z1", "c2": "Z1", "c3": "Z2"})  # c4 missing
        with self.assertRaisesRegex(ValueError, "no zone assignment"):
            add_geo_overheads(
                costs,
                pairs,
                origin_cell=oh_per_cell,
                dest_cell=oh_per_cell,
                cell_to_zone=partial_c2z,
            )

    def test_zone_aggregator_callable(self):
        """The `zone_aggregator` parameter accepts any pandas-compatible
        aggregator. Median, max, etc. all work."""
        pairs, costs, oh_per_cell, c2z = self._setup()
        result = add_geo_overheads(
            costs,
            pairs,
            origin_cell=oh_per_cell,
            dest_cell=oh_per_cell,
            cell_to_zone=c2z,
            zone_aggregator="max",
        )
        # Z1 max overhead = max(c1=10, c2=20) = 20.
        # Z2 max overhead = max(c3=30, c4=40) = 40.
        # z2z: 20 (Z1 origin) + 40 (Z2 dest) = 60.
        np.testing.assert_array_equal(result.zones_to_zones["Z1"], [60.0])

    def test_no_overhead_at_all_passes_through(self):
        """If neither cell nor zone overheads are given, costs pass through
        unchanged. cell_to_zone is then never required."""
        pairs, costs, _, _ = self._setup()
        result = add_geo_overheads(costs, pairs)
        np.testing.assert_array_equal(result.cells_to_cells["c1"], [0.0, 0.0])
        np.testing.assert_array_equal(result.cells_to_zones["c1"], [0.0])
        np.testing.assert_array_equal(result.zones_to_zones["Z1"], [0.0])

    def test_input_costs_not_mutated(self):
        """Input cost TieredODGeoPairs is not modified by add_geo_overheads."""
        pairs, costs, oh_per_cell, c2z = self._setup()
        before = costs.cells_to_cells["c1"].copy()
        _ = add_geo_overheads(
            costs,
            pairs,
            origin_cell=oh_per_cell,
            dest_cell=oh_per_cell,
            cell_to_zone=c2z,
        )
        np.testing.assert_array_equal(costs.cells_to_cells["c1"], before)


if __name__ == "__main__":
    unittest.main()
