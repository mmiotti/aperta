"""Tests for `aperta.od_pairs.get_pairs` and friends (the tiered OD API).

Run with:
    python -m unittest tests.test_od_pairs

The contract:
    `get_pairs` returns a `TieredODPairs` dataclass with up to three OD dicts —
    `cells_to_cells`, `cells_to_zones`, `zones_to_zones` — at increasing
    levels of aggregation. Tier assignment is per-ZONE-pair (uses zone-centroid
    distance for clean mutual exclusion); the middle tier preserves per-cell
    origin precision.

Synthetic-world fixture: a 3×3 cell grid grouped into 3 zones:

    cells (unit spacing, integer coords):
        C6 C7 C8        zones: Z0 = {C0..C3}
        C3 C4 C5               Z1 = {C4, C5}
        C0 C1 C2               Z2 = {C6, C7, C8}

    Cell Ci is mapped to network node Ni at the same coords.
    Zone Zk → representative network node ZNk at the zone centroid.

Zone *polygon centroids* (what tier classification uses):
    Z0 → (1.0, 0.0)    Z1 → (1.0, 1.0)    Z2 → (1.0, 2.0)
Zone-pair distances: d(Z0, Z1) = 1.0, d(Z1, Z2) = 1.0, d(Z0, Z2) = 2.0.

(The network nodes ZN0/ZN1/ZN2 are placed elsewhere — at (1, 0.5), (1, 1),
(1, 2) — and only matter for tests that exercise `get_euclidian_dists` etc.)

Populations: each cell `Ci` has `population = (i + 1) * 100` (total = 4500).
Zones are aggregated through the cell→zone mapping so the conservation
invariant is testable.
"""

import logging
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from aperta.od_pairs import (
    TieredODGeoPairs,
    TieredODNodePairs,
    TieredODPairs,
    aggregate_across_modes,
    dest_values,
    get_euclidian_dists,
    get_pairs,
)

# Per-cell population values; distinct so summation bugs are detectable.
# Total = 100 + 200 + ... + 900 = 4500.
_CELL_POPULATIONS = [100, 200, 300, 400, 500, 600, 700, 800, 900]


logging.getLogger().setLevel(logging.ERROR)


# -------------------- fixture --------------------


def _square(x: float, y: float) -> Polygon:
    return Polygon([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)])


def _build_world() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cell_coords = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    node_specs = [(f"N{i}", float(x), float(y)) for i, (x, y) in enumerate(cell_coords)] + [
        ("ZN0", 1.0, 0.5),
        ("ZN1", 1.0, 1.0),
        ("ZN2", 1.0, 2.0),
    ]
    nodes = gpd.GeoDataFrame(
        [{"node_id": nid, "geometry": Point(x, y)} for nid, x, y in node_specs],
        crs="EPSG:2056",
    ).set_index("node_id")

    cell_rows = []
    for i, (x, y) in enumerate(cell_coords):
        zone_id = "Z0" if i <= 3 else ("Z1" if i <= 5 else "Z2")
        cell_rows.append(
            {
                "cell_id": f"C{i}",
                "node_id": f"N{i}",
                "zone_id": zone_id,
                "population": _CELL_POPULATIONS[i],
                "geometry": _square(x, y),
            }
        )
    cells = gpd.GeoDataFrame(cell_rows, crs="EPSG:2056").set_index("cell_id")

    zones = gpd.GeoDataFrame(
        [
            {
                "zone_id": "Z0",
                "node_id": "ZN0",
                "geometry": Polygon([(-0.5, -0.5), (2.5, -0.5), (2.5, 0.5), (-0.5, 0.5)]),
            },
            {
                "zone_id": "Z1",
                "node_id": "ZN1",
                "geometry": Polygon([(-0.5, 0.5), (2.5, 0.5), (2.5, 1.5), (-0.5, 1.5)]),
            },
            {
                "zone_id": "Z2",
                "node_id": "ZN2",
                "geometry": Polygon([(-0.5, 1.5), (2.5, 1.5), (2.5, 2.5), (-0.5, 2.5)]),
            },
        ],
        crs="EPSG:2056",
    ).set_index("zone_id")
    zones["population"] = cells.groupby("zone_id")["population"].sum()
    return nodes, cells, zones


# -------------------- helpers --------------------


def _is_symmetric(d: dict) -> tuple[bool, tuple | None]:
    """`d` is symmetric iff `b ∈ d[a]` implies `a ∈ d[b]` for every (a, b)."""
    for a, dests in d.items():
        for b in dests:
            if a not in d.get(b, []):
                return False, (a, b)
    return True, None


# -------------------- tests --------------------


class CellsOnlyTestCase(unittest.TestCase):
    """`get_pairs` with no zones: per-cell distance filter, single tier."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def test_returns_only_cells_tier(self):
        pairs = get_pairs(self.cells, r_cells=0.5, node_column="node_id")
        self.assertIsInstance(pairs, TieredODPairs)
        self.assertIsNone(pairs.cells_to_zones)
        self.assertIsNone(pairs.zones_to_zones)
        for cn in self.cells["node_id"]:
            self.assertIn(cn, pairs.cells_to_cells[cn].tolist())

    def test_per_cell_distance(self):
        pairs = get_pairs(self.cells, r_cells=1.01, node_column="node_id")
        self.assertCountEqual(pairs.cells_to_cells["N0"].tolist(), ["N0", "N1", "N3"])
        self.assertCountEqual(pairs.cells_to_cells["N4"].tolist(), ["N1", "N3", "N4", "N5", "N7"])

    def test_symmetric(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column="node_id")
        ok, bad = _is_symmetric(pairs.cells_to_cells)
        self.assertTrue(ok, f"asymmetric pair: {bad}")


class TieredAssignmentTestCase(unittest.TestCase):
    """Per-zone-pair tier classification by zone-centroid distance.

    With our toy world: d(Z0, Z1)=0.5, d(Z1, Z2)=1.0, d(Z0, Z2)=1.5.
    """

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def test_same_zone_always_cell_tier(self):
        # r_cells=0.0: only the same-zone carve-out fires.
        pairs = get_pairs(
            self.cells, r_cells=0.0, node_column="node_id", zones=self.zones, r_zones=0.0
        )
        for orig in ["N0", "N1", "N2", "N3"]:
            self.assertCountEqual(pairs.cells_to_cells[orig].tolist(), ["N0", "N1", "N2", "N3"])

    def test_cells_to_zones_when_pair_in_middle_band(self):
        # r_cells=0, r_medium=3.0, r_zones=3.0 → all non-same-zone pairs
        # (d ∈ {1, 1, 2}) land in the middle tier; far tier empty.
        pairs = get_pairs(
            self.cells,
            r_cells=0.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=3.0,
            r_medium=3.0,
        )
        assert pairs.cells_to_zones is not None
        # N0 (cell C0 in Z0) → should reach ZN1 and ZN2 at middle tier.
        self.assertCountEqual(pairs.cells_to_zones["N0"].tolist(), ["ZN1", "ZN2"])
        self.assertIsNone(pairs.zones_to_zones)

    def test_zones_to_zones_when_pair_in_far_band(self):
        # r_cells=0, r_medium=0 → no middle tier; r_zones=3.0 → all
        # non-same-zone pairs land in the far tier.
        pairs = get_pairs(
            self.cells,
            r_cells=0.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=3.0,
            r_medium=0.0,
        )
        assert pairs.zones_to_zones is not None
        # ZN0 (Z0) → reaches ZN1 and ZN2 at far tier.
        self.assertCountEqual(pairs.zones_to_zones["ZN0"].tolist(), ["ZN1", "ZN2"])
        self.assertIsNone(pairs.cells_to_zones)

    def test_far_tier_dropped_when_outside_r_zones(self):
        # r_zones=0 → no zone-pair is in the outer annulus.
        pairs = get_pairs(
            self.cells,
            r_cells=0.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=0.0,
            r_medium=0.0,
        )
        self.assertIsNone(pairs.zones_to_zones)
        self.assertIsNone(pairs.cells_to_zones)


class SymmetryTestCase(unittest.TestCase):
    """Tiered output is symmetric within node-keyed tiers (cells_to_cells and
    zones_to_zones). The middle tier mixes cell-node and zone-node keys, so
    symmetry doesn't apply per-key — only its overall coverage is symmetric.
    """

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def test_node_keyed_tiers_symmetric(self):
        # r_cells=1.5 catches d(Z0,Z1)=d(Z1,Z2)=1.0; r_medium=1.5 (=r_cells,
        # drops middle tier); r_zones=2.5 catches d(Z0,Z2)=2.0 in far tier.
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            r_medium=1.5,
        )
        ok_c, bad_c = _is_symmetric(pairs.cells_to_cells)
        self.assertTrue(ok_c, f"cells_to_cells asymmetric at {bad_c}")
        assert pairs.zones_to_zones is not None
        ok_z, bad_z = _is_symmetric(pairs.zones_to_zones)
        self.assertTrue(ok_z, f"zones_to_zones asymmetric at {bad_z}")


class CustomZoneCentroidsTestCase(unittest.TestCase):
    """User-supplied `zones_centroids` override `zones.geometry.centroid`."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def test_custom_centroids_change_cell_tier(self):
        # Move Z1's centroid far out so (Z0, Z1) is no longer cell-tier.
        custom = gpd.GeoSeries(
            [Point(1.0, 0.0), Point(100.0, 100.0), Point(1.0, 2.0)],
            index=self.zones.index,
            crs=self.zones.crs,
        )
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            zones_centroids=custom,
        )
        # N0 should now see only Z0 cells (no cross to Z1 at cell tier).
        self.assertSetEqual(set(pairs.cells_to_cells["N0"].tolist()), {"N0", "N1", "N2", "N3"})


class SharedNodeDedupTestCase(unittest.TestCase):
    """Multiple cells sharing a node: dedup, and populations sum in dest_values."""

    def test_shared_dest_node(self):
        cells = gpd.GeoDataFrame(
            {
                "node_id": ["N0", "N0", "N1"],
                "zone_id": ["Z0", "Z0", "Z1"],
                "population": [10, 20, 30],
                "geometry": [Point(0, 0), Point(0.1, 0), Point(1, 0)],
            },
            index=pd.Index(["C0", "C1", "C2"], name="cell_id"),
            crs="EPSG:2056",
        )
        zones = gpd.GeoDataFrame(
            {
                "node_id": ["ZN0", "ZN1"],
                "population": [30, 30],
                "geometry": [Point(0.05, 0), Point(1, 0)],
            },
            index=pd.Index(["Z0", "Z1"], name="zone_id"),
            crs="EPSG:2056",
        )
        pairs = get_pairs(cells, r_cells=2.0, node_column="node_id", zones=zones, r_zones=2.0)
        self.assertCountEqual(pairs.cells_to_cells["N0"].tolist(), ["N0", "N1"])
        vals = dest_values("population", pairs, cells, "node_id", zones=zones)
        n0_dests = pairs.cells_to_cells["N0"]
        n0_vals = dict(zip(n0_dests.tolist(), vals.cells_to_cells["N0"].tolist()))
        self.assertEqual(n0_vals["N0"], 30)  # 10 + 20
        self.assertEqual(n0_vals["N1"], 30)


class DestValuesTestCase(unittest.TestCase):
    """`dest_values` looks up the per-tier value matching each tier's destinations."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()
        # Override zone centroids to get three DISTINCT zone-pair distances
        # (1.0, 3.0, 4.0), so all three tiers can populate from one call.
        cls.custom_centroids = gpd.GeoSeries(
            [Point(1.0, 0.0), Point(1.0, 1.0), Point(1.0, 4.0)],
            index=cls.zones.index,
            crs=cls.zones.crs,
        )

    def test_returns_three_dicts_same_shape(self):
        # r_cells=1.5 catches d(Z0,Z1)=1.0; r_medium=3.5 catches d(Z1,Z2)=3.0;
        # r_zones=4.5 catches d(Z0,Z2)=4.0. All three tiers populated.
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=4.5,
            r_medium=3.5,
            zones_centroids=self.custom_centroids,
        )
        vals = dest_values("population", pairs, self.cells, "node_id", zones=self.zones)
        for ot_pairs, ot_vals in [
            (pairs.cells_to_cells, vals.cells_to_cells),
            (pairs.cells_to_zones, vals.cells_to_zones),
            (pairs.zones_to_zones, vals.zones_to_zones),
        ]:
            if ot_pairs is None:
                self.assertIsNone(ot_vals)
                continue
            assert ot_vals is not None
            self.assertSetEqual(set(ot_pairs.keys()), set(ot_vals.keys()))
            for k in ot_pairs:
                self.assertEqual(len(ot_pairs[k]), len(ot_vals[k]))

    def test_raises_without_required_frame(self):
        pairs = get_pairs(
            self.cells, r_cells=1.5, node_column="node_id", zones=self.zones, r_zones=2.5
        )
        with self.assertRaisesRegex(ValueError, "zones"):
            dest_values("population", pairs, self.cells, "node_id")

    def test_raises_on_missing_column(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column="node_id")
        with self.assertRaisesRegex(ValueError, "unknown_col"):
            dest_values("unknown_col", pairs, self.cells, "node_id")

    def test_cells_to_zones_looks_up_zones_column(self):
        # r_cells=0, r_medium=r_zones=3.0 → all non-same-zone pairs (d ∈ {1,1,2})
        # in middle tier.
        pairs = get_pairs(
            self.cells,
            r_cells=0.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=3.0,
            r_medium=3.0,
        )
        vals = dest_values("population", pairs, self.cells, "node_id", zones=self.zones)
        assert vals.cells_to_zones is not None
        # N0 (in Z0) → ZN1 and ZN2 at middle tier. Z1 pop = 500+600=1100,
        # Z2 pop = 700+800+900=2400.
        d_to_v = dict(zip(pairs.cells_to_zones["N0"].tolist(), vals.cells_to_zones["N0"].tolist()))
        self.assertEqual(d_to_v["ZN1"], 1100)
        self.assertEqual(d_to_v["ZN2"], 2400)


class ConservationTestCase(unittest.TestCase):
    """Population invariant: for every origin cell, sum across all tiers = total."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def _check_conservation(self, pairs):
        vals = dest_values("population", pairs, self.cells, "node_id", zones=self.zones)
        total = sum(_CELL_POPULATIONS)
        cell_to_zone = self.cells.set_index("node_id")["zone_id"]
        zone_to_node = self.zones["node_id"]
        for origin in self.cells["node_id"]:
            tier_sum = vals.cells_to_cells.get(origin, np.array([])).sum()
            zone_origin = zone_to_node[cell_to_zone[origin]]
            if vals.cells_to_zones is not None:
                tier_sum += vals.cells_to_zones.get(origin, np.array([])).sum()
            if vals.zones_to_zones is not None:
                tier_sum += vals.zones_to_zones.get(zone_origin, np.array([])).sum()
            self.assertEqual(tier_sum, total, f"origin {origin}: {tier_sum} != {total}")

    def test_conservation_three_tier_full_coverage(self):
        # r_cells=1.0 cell-tier covers (Z0,Z1) (d=0.5); r_medium=1.2 middle
        # tier covers (Z1,Z2) (d=1.0); r_zones=100 far tier covers (Z0,Z2)
        # (d=1.5). All zone pairs slot into exactly one tier.
        pairs = get_pairs(
            self.cells,
            r_cells=1.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=100.0,
            r_medium=1.2,
        )
        self._check_conservation(pairs)

    def test_conservation_two_tier_no_far(self):
        # r_medium=r_zones → middle tier captures everything, far tier empty.
        pairs = get_pairs(
            self.cells,
            r_cells=1.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=10.0,
            r_medium=10.0,
        )
        self.assertIsNone(pairs.zones_to_zones)
        self._check_conservation(pairs)

    def test_conservation_no_middle_tier(self):
        # r_medium=r_cells → middle tier empty; far tier captures everything.
        pairs = get_pairs(
            self.cells,
            r_cells=0.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=10.0,
            r_medium=0.5,
        )
        self.assertIsNone(pairs.cells_to_zones)
        self._check_conservation(pairs)


class GetEuclidianDistsTestCase(unittest.TestCase):
    """`get_euclidian_dists` returns a TieredODPairs of float distance arrays."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def test_dists_pair_position_wise_with_pairs(self):
        pairs = get_pairs(
            self.cells, r_cells=1.5, node_column="node_id", zones=self.zones, r_zones=2.5
        )
        dists = get_euclidian_dists(self.nodes, pairs)
        for origin, dest_ids in pairs.cells_to_cells.items():
            self.assertEqual(len(dest_ids), len(dists.cells_to_cells[origin]))
            for i, d in enumerate(dest_ids):
                expected = np.hypot(
                    self.nodes.loc[origin].geometry.x - self.nodes.loc[d].geometry.x,
                    self.nodes.loc[origin].geometry.y - self.nodes.loc[d].geometry.y,
                )
                self.assertAlmostEqual(dists.cells_to_cells[origin][i], expected, places=6)

    def test_dists_dtype_param(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column="node_id")
        dists = get_euclidian_dists(self.nodes, pairs, dtype=np.float32)
        first = next(iter(dists.cells_to_cells.values()))
        self.assertEqual(first.dtype, np.float32)

    def test_dists_three_tiers(self):
        # Override centroids for 3 distinct zone-pair distances: 1, 3, 4.
        custom = gpd.GeoSeries(
            [Point(1.0, 0.0), Point(1.0, 1.0), Point(1.0, 4.0)],
            index=self.zones.index,
            crs=self.zones.crs,
        )
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=4.5,
            r_medium=3.5,
            zones_centroids=custom,
        )
        dists = get_euclidian_dists(self.nodes, pairs)
        assert dists.cells_to_zones is not None
        assert dists.zones_to_zones is not None
        # ZN0 at (1, 0.5), ZN2 at (1, 2.0) → d = 1.5 (uses *network node*
        # coords, not zone polygon centroids).
        self.assertAlmostEqual(dists.zones_to_zones["ZN0"][0], 1.5, places=6)


class ValidationTestCase(unittest.TestCase):
    """Input contract enforcement."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def test_missing_node_column_raises(self):
        bad = self.cells.drop(columns="node_id")
        with self.assertRaisesRegex(ValueError, "missing required column 'node_id'"):
            get_pairs(bad, r_cells=1.0, node_column="node_id")

    def test_zones_without_r_zones_raises(self):
        with self.assertRaisesRegex(ValueError, "r_zones"):
            get_pairs(self.cells, r_cells=1.0, node_column="node_id", zones=self.zones)

    def test_cells_without_zone_id_raises(self):
        bad = self.cells.drop(columns="zone_id")
        with self.assertRaisesRegex(ValueError, "'zone_id'"):
            get_pairs(bad, r_cells=1.0, node_column="node_id", zones=self.zones, r_zones=1.0)


class DescribeTestCase(unittest.TestCase):
    """`TieredODPairs.describe` prints a per-tier summary for both ID and value tables."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    def test_describe_id_pairs_cells_only(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column="node_id")
        out = pairs.describe()
        self.assertIn("cells_to_cells", out)
        self.assertIn("zones_to_zones: None", out)

    def test_describe_id_pairs_all_tiers(self):
        pairs = get_pairs(
            self.cells, r_cells=1.5, node_column="node_id", zones=self.zones, r_zones=2.5
        )
        out = pairs.describe()
        self.assertIn("cells_to_cells", out)
        # cells_to_zones is populated here (auto r_medium = min(15, 2.5) = 2.5
        # → middle tier captures non-same-zone within [1.5, 2.5)).
        self.assertIn("cells_to_zones", out)

    def test_describe_distance_values_shows_stats(self):
        pairs = get_pairs(
            self.cells, r_cells=1.5, node_column="node_id", zones=self.zones, r_zones=2.5
        )
        dists = get_euclidian_dists(self.nodes, pairs)
        out = dists.describe()
        # Numeric tier should produce a stats line with 'mean' / 'median'.
        self.assertIn("mean", out)
        self.assertIn("median", out)


class GetPairsMaskFiltersTestCase(unittest.TestCase):
    """`orig_cells` / `dest_cells` / `dest_zones` masks restrict origins and
    destinations to subsets of the full universe.
    """

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones = _build_world()

    # ----- baseline: no mask = current behaviour -----
    def test_no_masks_matches_baseline(self):
        """All mask kwargs unspecified = identical to the no-filter call."""
        pairs_a = get_pairs(
            self.cells, r_cells=1.5, node_column="node_id", zones=self.zones, r_zones=2.5
        )
        pairs_b = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            orig_cells=None,
            dest_cells=None,
            dest_zones=None,
        )
        assert pairs_a.cells_to_cells is not None
        assert pairs_b.cells_to_cells is not None
        self.assertSetEqual(set(pairs_a.cells_to_cells.keys()), set(pairs_b.cells_to_cells.keys()))
        for k in pairs_a.cells_to_cells:
            self.assertCountEqual(
                pairs_a.cells_to_cells[k].tolist(), pairs_b.cells_to_cells[k].tolist()
            )

    # ----- orig_cells: filter origins -----
    def test_orig_cells_filters_origins(self):
        """orig_cells=mask drops cells where False from being origins entirely."""
        # Mark only C0 and C4 as origins (rest are excluded).
        orig = self.cells.index.isin(["C0", "C4"])
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            orig_cells=orig,
        )
        # Only N0 and N4 (cells C0 and C4) should appear as cells_to_cells origins.
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {"N0", "N4"})

    def test_orig_cells_excludes_zone_tier_origin_when_no_origins_in_zone(self):
        """A zone with no origin cells in it should not appear as zone-tier origin."""
        # No cells in Z2 (cells C6-C8) are origins.
        orig = ~self.cells["zone_id"].isin(["Z2"]).to_numpy()
        # r_cells=r_medium=1.5 → middle tier empty; all non-same-zone → far.
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=10.0,
            r_medium=1.5,
            orig_cells=orig,
        )
        assert pairs.zones_to_zones is not None
        # ZN2 is Z2's zone node — it should NOT appear as a zones_to_zones origin.
        self.assertNotIn("ZN2", pairs.zones_to_zones.keys())

    # ----- dest_cells: filter cell-tier destinations -----
    def test_dest_cells_filters_destinations(self):
        """dest_cells=mask drops cells where False from being cell-tier destinations."""
        # Only C0, C1, C2 are valid destinations.
        dest = self.cells.index.isin(["C0", "C1", "C2"])
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            dest_cells=dest,
        )
        # Every origin's cell-tier destinations should be a subset of {N0, N1, N2}.
        allowed = {"N0", "N1", "N2"}
        for orig, dests in pairs.cells_to_cells.items():
            self.assertTrue(
                set(dests.tolist()).issubset(allowed),
                f"Origin {orig!r} has dests {dests.tolist()} outside allowed set.",
            )

    def test_dest_cells_does_not_affect_zone_dest_tiers(self):
        """dest_cells filters cell-tier dests only; zone-dest tiers are independent."""
        dest = self.cells.index.isin(["C0", "C1"])
        pairs_filt = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            dest_cells=dest,
        )
        pairs_full = get_pairs(
            self.cells, r_cells=1.5, node_column="node_id", zones=self.zones, r_zones=2.5
        )
        # The zone-dest tiers (cells_to_zones, zones_to_zones) should match.
        for tier_name in ("cells_to_zones", "zones_to_zones"):
            full_tier = getattr(pairs_full, tier_name)
            filt_tier = getattr(pairs_filt, tier_name)
            if full_tier is None:
                self.assertIsNone(filt_tier)
                continue
            assert filt_tier is not None
            self.assertSetEqual(set(filt_tier.keys()), set(full_tier.keys()))
            for k in filt_tier:
                self.assertCountEqual(filt_tier[k].tolist(), full_tier[k].tolist())

    # ----- dest_zones: filter zone-tier destinations -----
    def test_dest_zones_filters_far_tier(self):
        """dest_zones drops filtered zones from far-tier destinations."""
        # r_cells=r_medium=1.5 → middle empty; r_zones=3.0 catches d=2.0
        # (Z0↔Z2). dest_z=[Z2] means only ZN2 can be a far-tier dest.
        dest_z = self.zones.index.isin(["Z2"])
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=3.0,
            r_medium=1.5,
            dest_zones=dest_z,
        )
        assert pairs.zones_to_zones is not None
        for orig, dests in pairs.zones_to_zones.items():
            self.assertTrue(
                all(d == "ZN2" for d in dests.tolist()),
                f"Origin {orig!r} has unexpected dests {dests.tolist()}.",
            )

    def test_dest_zones_filters_middle_tier(self):
        """dest_zones drops filtered zones from middle-tier destinations."""
        # r_cells=0, r_medium=r_zones=3.0 → all non-same-zone in middle tier.
        # dest_z=[Z1] → cells_to_zones dests must be ZN1 only.
        dest_z = self.zones.index.isin(["Z1"])
        pairs = get_pairs(
            self.cells,
            r_cells=0.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=3.0,
            r_medium=3.0,
            dest_zones=dest_z,
        )
        assert pairs.cells_to_zones is not None
        for orig, dests in pairs.cells_to_zones.items():
            self.assertTrue(
                all(d == "ZN1" for d in dests.tolist()),
                f"Origin {orig!r} has unexpected dests {dests.tolist()}.",
            )

    # ----- combined filters -----
    def test_combined_orig_and_dest_filters(self):
        """orig + dest filters compose: only filtered origins route to filtered dests."""
        orig = self.cells.index.isin(["C0", "C4"])
        dest = self.cells.index.isin(["C0", "C1", "C4"])
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            orig_cells=orig,
            dest_cells=dest,
        )
        # Only N0 and N4 appear as origins.
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {"N0", "N4"})
        # Each origin's dests are in {N0, N1, N4} (cells C0, C1, C4).
        allowed = {"N0", "N1", "N4"}
        for orig_node, dests in pairs.cells_to_cells.items():
            self.assertTrue(set(dests.tolist()).issubset(allowed))

    # ----- cells-only (no zones) variant -----
    def test_masks_work_in_cells_only_mode(self):
        """orig_cells and dest_cells filters work when zones aren't provided."""
        orig = self.cells.index.isin(["C0", "C1"])
        dest = self.cells.index.isin(["C0", "C3"])
        pairs = get_pairs(
            self.cells, r_cells=1.5, node_column="node_id", orig_cells=orig, dest_cells=dest
        )
        self.assertIsNone(pairs.zones_to_zones)
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {"N0", "N1"})
        allowed = {"N0", "N3"}
        for orig_node, dests in pairs.cells_to_cells.items():
            self.assertTrue(set(dests.tolist()).issubset(allowed))

    # ----- validation -----
    def test_wrong_length_mask_raises(self):
        wrong = np.array([True] * (len(self.cells) + 1))
        with self.assertRaisesRegex(ValueError, "length"):
            get_pairs(
                self.cells,
                r_cells=1.5,
                node_column="node_id",
                zones=self.zones,
                r_zones=2.5,
                orig_cells=wrong,
            )

    def test_non_boolean_mask_raises(self):
        not_bool = np.arange(len(self.cells))
        with self.assertRaisesRegex(ValueError, "boolean"):
            get_pairs(
                self.cells,
                r_cells=1.5,
                node_column="node_id",
                zones=self.zones,
                r_zones=2.5,
                dest_cells=not_bool,
            )

    def test_series_mask_accepted(self):
        """pd.Series boolean masks work the same as numpy arrays."""
        orig = self.cells["zone_id"] == "Z0"  # pd.Series of booleans
        pairs = get_pairs(
            self.cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            orig_cells=orig,
        )
        # Origins limited to N0-N3 (Z0's cells).
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {"N0", "N1", "N2", "N3"})

    def test_node_column_with_string_extension_dtype(self):
        """Regression: pandas `StringDtype` `node_column` shouldn't break
        the orig_cells filter. `s.unique()` returns a `pd.StringArray`
        whose `.dtype` is a pandas ExtensionDtype that `np.array(...,
        dtype=...)` downstream can't interpret. `cells_in_zone` coerces
        through `np.asarray` to defend against this."""
        cells = self.cells.copy()
        cells["node_id"] = cells["node_id"].astype("string")
        orig = self.cells["zone_id"] == "Z0"
        pairs = get_pairs(
            cells,
            r_cells=1.5,
            node_column="node_id",
            zones=self.zones,
            r_zones=2.5,
            orig_cells=orig,
        )
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {"N0", "N1", "N2", "N3"})


class AggregateAcrossModesTestCase(unittest.TestCase):
    """`aggregate_across_modes` combines per-mode geo-keyed (pairs, costs) into
    a combined (union_pairs, combined_costs)."""

    def _aligned_geo_odms(self):
        """Two modes with the SAME dest sets per origin (no union needed) —
        used for the basic aggregator semantics tests."""
        walk_pairs = TieredODGeoPairs(
            cells_to_cells={"A": np.array(["X", "Y", "Z"])},
            zones_to_zones={"ZA": np.array(["ZX", "ZY"])},
        )
        walk_costs = TieredODGeoPairs(
            cells_to_cells={"A": np.array([100.0, 200.0, 300.0])},
            zones_to_zones={"ZA": np.array([1500.0, 5000.0])},
        )
        car_pairs = TieredODGeoPairs(
            cells_to_cells={"A": np.array(["X", "Y", "Z"])},
            zones_to_zones={"ZA": np.array(["ZX", "ZY"])},
        )
        car_costs = TieredODGeoPairs(
            cells_to_cells={"A": np.array([50.0, 250.0, 100.0])},
            zones_to_zones={"ZA": np.array([800.0, 4000.0])},
        )
        return {"walk": (walk_pairs, walk_costs), "car": (car_pairs, car_costs)}

    def test_returns_geo_pairs(self):
        union_pairs, combined = aggregate_across_modes(self._aligned_geo_odms(), aggregator="min")
        self.assertIsInstance(union_pairs, TieredODGeoPairs)
        self.assertIsInstance(combined, TieredODGeoPairs)

    def test_min_per_pair(self):
        """`min` aggregator picks the cheapest mode per OD pair."""
        union_pairs, combined = aggregate_across_modes(self._aligned_geo_odms(), aggregator="min")
        # dest order in union is sorted: X, Y, Z.
        self.assertEqual(list(union_pairs.cells_to_cells["A"]), ["X", "Y", "Z"])
        np.testing.assert_array_equal(combined.cells_to_cells["A"], np.array([50.0, 200.0, 100.0]))
        self.assertEqual(list(union_pairs.zones_to_zones["ZA"]), ["ZX", "ZY"])
        np.testing.assert_array_equal(combined.zones_to_zones["ZA"], np.array([800.0, 4000.0]))

    def test_logsum_scale_1_canonical(self):
        """`logsum` with scale=1 returns -ln Σ exp(-cost). Hand-check one entry."""
        _, combined = aggregate_across_modes(
            self._aligned_geo_odms(), aggregator="logsum", scale=1.0
        )
        # First cell-tier OD (dest X): walk=100, car=50 → ≈ 50.
        self.assertAlmostEqual(combined.cells_to_cells["A"][0], 50.0, places=5)

    def test_logsum_two_equal_costs(self):
        """Two modes at identical cost: logsum = cost - ln(2)."""
        a_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D"])})
        a_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([10.0])})
        b_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D"])})
        b_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([10.0])})
        _, combined = aggregate_across_modes(
            {"a": (a_pairs, a_costs), "b": (b_pairs, b_costs)}, aggregator="logsum", scale=1.0
        )
        # -ln(2 · exp(-10)) = 10 - ln(2)
        self.assertAlmostEqual(combined.cells_to_cells["O"][0], 10.0 - np.log(2))

    def test_logsum_with_scale(self):
        a_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D"])})
        a_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([20.0])})
        b_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D"])})
        b_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([20.0])})
        scale = 5.0
        _, combined = aggregate_across_modes(
            {"a": (a_pairs, a_costs), "b": (b_pairs, b_costs)}, aggregator="logsum", scale=scale
        )
        # -5 · ln(2 · exp(-4)) = 20 - 5·ln(2)
        self.assertAlmostEqual(combined.cells_to_cells["O"][0], 20.0 - scale * np.log(2))

    def test_custom_callable_aggregator(self):
        """Callable aggregator: takes (n_modes, n_dests) and returns (n_dests,)."""

        def weighted_mean(stacked: np.ndarray) -> np.ndarray:
            return (1 / 3) * stacked[0] + (2 / 3) * stacked[1]

        _, combined = aggregate_across_modes(self._aligned_geo_odms(), aggregator=weighted_mean)
        expected_first = (1 / 3) * 100.0 + (2 / 3) * 50.0
        self.assertAlmostEqual(combined.cells_to_cells["A"][0], expected_first)

    # ---------- Union + inf-fill semantics ----------

    def test_disjoint_dests_per_origin_union_with_inf_fill(self):
        """When modes have different dest sets, the result has the union; the
        mode that lacks a given dest contributes inf to that position."""
        walk_pairs = TieredODGeoPairs(cells_to_cells={"A": np.array(["X", "Y"])})
        walk_costs = TieredODGeoPairs(cells_to_cells={"A": np.array([10.0, 20.0])})
        car_pairs = TieredODGeoPairs(cells_to_cells={"A": np.array(["Y", "Z"])})
        car_costs = TieredODGeoPairs(cells_to_cells={"A": np.array([5.0, 7.0])})
        union_pairs, combined = aggregate_across_modes(
            {"walk": (walk_pairs, walk_costs), "car": (car_pairs, car_costs)}, aggregator="min"
        )
        # Union dests = {X, Y, Z}, sorted.
        self.assertEqual(list(union_pairs.cells_to_cells["A"]), ["X", "Y", "Z"])
        # X: walk=10, car=inf → min=10. Y: walk=20, car=5 → min=5. Z: walk=inf, car=7 → 7.
        np.testing.assert_array_equal(combined.cells_to_cells["A"], np.array([10.0, 5.0, 7.0]))

    def test_disjoint_origins_union(self):
        """Modes with different origin sets → union of origins appears in result."""
        walk_pairs = TieredODGeoPairs(cells_to_cells={"A": np.array(["D"])})
        walk_costs = TieredODGeoPairs(cells_to_cells={"A": np.array([10.0])})
        car_pairs = TieredODGeoPairs(cells_to_cells={"B": np.array(["D"])})
        car_costs = TieredODGeoPairs(cells_to_cells={"B": np.array([5.0])})
        union_pairs, combined = aggregate_across_modes(
            {"walk": (walk_pairs, walk_costs), "car": (car_pairs, car_costs)}, aggregator="min"
        )
        self.assertSetEqual(set(union_pairs.cells_to_cells.keys()), {"A", "B"})
        # A only routed by walk → A's combined = walk cost.
        np.testing.assert_array_equal(combined.cells_to_cells["A"], np.array([10.0]))
        # B only routed by car → B's combined = car cost.
        np.testing.assert_array_equal(combined.cells_to_cells["B"], np.array([5.0]))

    def test_min_with_inf_treated_as_unreachable(self):
        """`inf` is finite-worst; if any mode is reachable, min picks that one."""
        a_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D1", "D2"])})
        a_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([np.inf, 100.0])})
        b_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D1", "D2"])})
        b_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([50.0, np.inf])})
        _, combined = aggregate_across_modes(
            {"a": (a_pairs, a_costs), "b": (b_pairs, b_costs)}, aggregator="min"
        )
        np.testing.assert_array_equal(combined.cells_to_cells["O"], np.array([50.0, 100.0]))

    def test_logsum_inf_contributes_nothing(self):
        """exp(-inf) = 0, so an unreachable mode does not affect the logsum."""
        a_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D1", "D2"])})
        a_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([10.0, np.inf])})
        b_pairs = TieredODGeoPairs(cells_to_cells={"O": np.array(["D1", "D2"])})
        b_costs = TieredODGeoPairs(cells_to_cells={"O": np.array([np.inf, 10.0])})
        _, combined = aggregate_across_modes(
            {"a": (a_pairs, a_costs), "b": (b_pairs, b_costs)}, aggregator="logsum", scale=1.0
        )
        np.testing.assert_array_almost_equal(combined.cells_to_cells["O"], np.array([10.0, 10.0]))

    def test_single_mode_passthrough(self):
        """One mode: combined ODM equals that mode's ODM (no-op)."""
        only_pairs = TieredODGeoPairs(cells_to_cells={"A": np.array(["X", "Y", "Z"])})
        only_costs = TieredODGeoPairs(cells_to_cells={"A": np.array([1.0, 2.0, 3.0])})
        union_pairs, combined = aggregate_across_modes(
            {"only": (only_pairs, only_costs)}, aggregator="min"
        )
        self.assertEqual(list(union_pairs.cells_to_cells["A"]), ["X", "Y", "Z"])
        np.testing.assert_array_equal(combined.cells_to_cells["A"], np.array([1.0, 2.0, 3.0]))

    def test_missing_tier_stays_none(self):
        """Tiers that are None in all inputs stay None in the output."""
        a_pairs = TieredODGeoPairs(cells_to_cells={"A": np.array(["X"])})
        a_costs = TieredODGeoPairs(cells_to_cells={"A": np.array([1.0])})
        b_pairs = TieredODGeoPairs(cells_to_cells={"A": np.array(["X"])})
        b_costs = TieredODGeoPairs(cells_to_cells={"A": np.array([2.0])})
        union_pairs, combined = aggregate_across_modes(
            {"walk": (a_pairs, a_costs), "car": (b_pairs, b_costs)}, aggregator="min"
        )
        self.assertIsNone(union_pairs.zones_to_zones)
        self.assertIsNone(combined.zones_to_zones)

    def test_inconsistent_tier_structure_raises(self):
        """If one mode has a tier and another doesn't, raise."""
        a_pairs = TieredODGeoPairs(cells_to_cells={"A": np.array(["X"])})
        a_costs = TieredODGeoPairs(cells_to_cells={"A": np.array([1.0])})
        b_pairs = TieredODGeoPairs(
            cells_to_cells={"A": np.array(["X"])}, zones_to_zones={"ZA": np.array(["ZX"])}
        )
        b_costs = TieredODGeoPairs(
            cells_to_cells={"A": np.array([2.0])}, zones_to_zones={"ZA": np.array([5.0])}
        )
        with self.assertRaisesRegex(ValueError, "tier"):
            aggregate_across_modes(
                {"walk": (a_pairs, a_costs), "car": (b_pairs, b_costs)}, aggregator="min"
            )

    def test_empty_odms_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            aggregate_across_modes({}, aggregator="min")

    def test_unknown_aggregator_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown aggregator"):
            aggregate_across_modes(self._aligned_geo_odms(), aggregator="nope")

    def test_node_keyed_input_raises_type_error(self):
        """Cross-modal aggregation must run in geo-unit space, not node space."""
        walk_node = TieredODNodePairs(cells_to_cells={"A": np.array(["X"])})
        walk_node_costs = TieredODNodePairs(cells_to_cells={"A": np.array([1.0])})
        with self.assertRaisesRegex(TypeError, "TieredODGeoPairs"):
            aggregate_across_modes({"walk": (walk_node, walk_node_costs)}, aggregator="min")


class CellsToZonesMiddleTierTestCase(unittest.TestCase):
    """Phase B: `cells_to_zones` middle tier (cell origin, zone dest, for
    zone-pair distances in [r_cells, r_medium)). Validates the new three-tier
    design with cells_to_cells + cells_to_zones + zones_to_zones.

    Toy world: 6 cells in 3 zones on a line, with zone centroids at x = 0.5,
    5.5, 12.5. Zone-pair distances: d(ZA, ZB) = 5, d(ZB, ZC) = 7,
    d(ZA, ZC) = 12.
    """

    @classmethod
    def setUpClass(cls):
        cell_data = [
            ("a0", 0.0, 0.0, "ZA"),
            ("a1", 1.0, 0.0, "ZA"),
            ("b0", 5.0, 0.0, "ZB"),
            ("b1", 6.0, 0.0, "ZB"),
            ("c0", 12.0, 0.0, "ZC"),
            ("c1", 13.0, 0.0, "ZC"),
        ]
        cls.cells = gpd.GeoDataFrame(
            {
                "cell_id": [r[0] for r in cell_data],
                "node_id": [r[0] for r in cell_data],
                "zone_id": [r[3] for r in cell_data],
            },
            geometry=[Point(r[1], r[2]) for r in cell_data],
        ).set_index("cell_id")
        zone_data = [("ZA", 0.5), ("ZB", 5.5), ("ZC", 12.5)]
        cls.zones = gpd.GeoDataFrame(
            {"zone_id": [r[0] for r in zone_data], "node_id": [r[0] for r in zone_data]},
            geometry=[Point(r[1], 0.0) for r in zone_data],
        ).set_index("zone_id")

    def test_tier_classification_by_zone_pair_distance(self):
        """All three tiers populate correctly based on zone-pair distance.

        With r_cells=2, r_medium=8, r_zones=15:
          - ZA-ZA, ZB-ZB, ZC-ZC (d=0) → cells_to_cells (same-zone)
          - ZA-ZB (5), ZB-ZC (7) in [2, 8) → cells_to_zones
          - ZA-ZC (12) in [8, 15) → zones_to_zones
        """
        pairs = get_pairs(
            self.cells,
            r_cells=2.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=15.0,
            r_medium=8.0,
        )
        # cells_to_cells: each cell has only its same-zone partner + self.
        self.assertEqual(set(pairs.cells_to_cells["a0"]), {"a0", "a1"})
        self.assertEqual(set(pairs.cells_to_cells["b0"]), {"b0", "b1"})
        self.assertEqual(set(pairs.cells_to_cells["c0"]), {"c0", "c1"})
        # cells_to_zones: a0/a1 → ZB; b0/b1 → ZA, ZC; c0/c1 → ZB.
        assert pairs.cells_to_zones is not None
        self.assertEqual(set(pairs.cells_to_zones["a0"]), {"ZB"})
        self.assertEqual(set(pairs.cells_to_zones["a1"]), {"ZB"})
        self.assertEqual(set(pairs.cells_to_zones["b0"]), {"ZA", "ZC"})
        self.assertEqual(set(pairs.cells_to_zones["c0"]), {"ZB"})
        # zones_to_zones: ZA ↔ ZC only.
        assert pairs.zones_to_zones is not None
        self.assertEqual(set(pairs.zones_to_zones["ZA"]), {"ZC"})
        self.assertEqual(set(pairs.zones_to_zones["ZC"]), {"ZA"})
        self.assertNotIn("ZB", pairs.zones_to_zones)  # ZB has no far-tier dests

    def test_r_medium_auto_inference(self):
        """Default r_medium = min(r_cells * 10, r_zones)."""
        # r_cells=1, r_zones=100 → auto r_medium = min(10, 100) = 10
        # d(ZA, ZB) = 5 < 10 → cells_to_zones
        # d(ZB, ZC) = 7 < 10 → cells_to_zones
        # d(ZA, ZC) = 12 >= 10 → zones_to_zones
        pairs = get_pairs(
            self.cells, r_cells=1.0, node_column="node_id", zones=self.zones, r_zones=100.0
        )
        assert pairs.cells_to_zones is not None
        assert pairs.zones_to_zones is not None
        self.assertEqual(set(pairs.cells_to_zones["a0"]), {"ZB"})
        self.assertEqual(set(pairs.zones_to_zones["ZA"]), {"ZC"})

    def test_r_medium_equals_r_zones_drops_far_tier(self):
        """When r_medium == r_zones, the far tier is empty (and absent)."""
        pairs = get_pairs(
            self.cells,
            r_cells=1.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=15.0,
            r_medium=15.0,
        )
        # All non-same-zone pairs are cells_to_zones; zones_to_zones is None.
        self.assertIsNone(pairs.zones_to_zones)
        assert pairs.cells_to_zones is not None
        self.assertEqual(set(pairs.cells_to_zones["a0"]), {"ZB", "ZC"})

    def test_r_medium_equals_r_cells_drops_middle_tier(self):
        """When r_medium == r_cells, the middle tier is empty (and absent)."""
        pairs = get_pairs(
            self.cells,
            r_cells=2.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=15.0,
            r_medium=2.0,
        )
        self.assertIsNone(pairs.cells_to_zones)
        # All non-same-zone pairs go to zones_to_zones.
        assert pairs.zones_to_zones is not None
        self.assertEqual(set(pairs.zones_to_zones["ZA"]), {"ZB", "ZC"})

    def test_invalid_r_medium_raises(self):
        """`r_medium` must satisfy r_cells ≤ r_medium ≤ r_zones."""
        with self.assertRaisesRegex(ValueError, "r_medium"):
            get_pairs(
                self.cells,
                r_cells=2.0,
                node_column="node_id",
                zones=self.zones,
                r_zones=10.0,
                r_medium=20.0,
            )
        with self.assertRaisesRegex(ValueError, "r_medium"):
            get_pairs(
                self.cells,
                r_cells=5.0,
                node_column="node_id",
                zones=self.zones,
                r_zones=10.0,
                r_medium=2.0,
            )

    def test_mutual_exclusion_no_double_counting(self):
        """No (origin, dest) pair appears in more than one tier."""
        pairs = get_pairs(
            self.cells,
            r_cells=2.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=15.0,
            r_medium=8.0,
        )
        # For each cell origin, count: same-cell dests across cells_to_cells
        # and cells_to_zones (translated to zones).
        assert pairs.cells_to_zones is not None
        for cell_origin in pairs.cells_to_cells:
            cell_dests = set(pairs.cells_to_cells[cell_origin].tolist())
            cell_dest_zones = {self.cells.loc[c, "zone_id"] for c in cell_dests}
            c2z_dest_zones = (
                set(pairs.cells_to_zones[cell_origin].tolist())
                if cell_origin in pairs.cells_to_zones
                else set()
            )
            # Cell-tier dest zones and c2z dest zones must not overlap.
            self.assertFalse(
                cell_dest_zones & c2z_dest_zones,
                f"Origin {cell_origin}: overlap between cell-tier dest zones "
                f"{cell_dest_zones} and cells_to_zones {c2z_dest_zones}.",
            )

    def test_cells_in_same_zone_share_c2z_dest_set(self):
        """Cells in the same zone get the same cells_to_zones dest zones
        (since tier classification is zone-pair-based, all cells in zone Z
        see the same medium-tier dest zones)."""
        pairs = get_pairs(
            self.cells,
            r_cells=2.0,
            node_column="node_id",
            zones=self.zones,
            r_zones=15.0,
            r_medium=8.0,
        )
        assert pairs.cells_to_zones is not None
        # a0 and a1 (both in ZA) have identical dest zone sets.
        self.assertEqual(
            set(pairs.cells_to_zones["a0"]),
            set(pairs.cells_to_zones["a1"]),
        )


if __name__ == "__main__":
    unittest.main()
