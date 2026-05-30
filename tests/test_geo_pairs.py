"""Tests for the geo-keyed branch of the tiered OD API:

- `od_pairs.reindex_by_geo_unit` — node-keyed → geo-unit-keyed conversion.
- `od_pairs.dest_values_geo`     — destination value lookup on geo-keyed pairs.
- `overhead.add_geo_overheads`   — generic origin/dest overhead application.
- `overhead.add_origin_cell_overhead` — convenience wrapper for per-cell origin
  overhead baking at all tiers.

Run with:
    python -m unittest tests.test_geo_pairs
"""

import unittest
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from aperta.od_pairs import (
    TieredODGeoPairs,
    TieredODNodePairs,
    dest_values_geo,
    reindex_by_geo_unit,
)
from aperta.overhead import add_geo_overheads, add_origin_cell_overhead

# ---------------------------------------------------------------------------
# Fixture: a small node-keyed (pairs, odm) on which reindex semantics are
# straightforward to verify by hand.
#
# Network nodes: N0, N1, N2.
# Cells: C0 → N0, C1 → N0, C2 → N1, C3 → N2.   (two cells share N0)
# Zones: Z0 (contains C0, C1, C2) → ZN0;  Z1 (contains C3) → ZN1.
# ---------------------------------------------------------------------------


def _build_fixture():
    cells = pd.DataFrame(
        {
            "node_id": ["N0", "N0", "N1", "N2"],
            "zone_id": ["Z0", "Z0", "Z0", "Z1"],
            "population": [10.0, 20.0, 5.0, 7.0],
            "walk_overhead_s": [30.0, 60.0, 45.0, 90.0],
        },
        index=pd.Index(["C0", "C1", "C2", "C3"], name="cell_id"),
    )
    zones = pd.DataFrame(
        {
            "node_id": ["ZN0", "ZN1"],
            "population": [35.0, 7.0],
        },
        index=pd.Index(["Z0", "Z1"], name="zone_id"),
    )
    return cells, zones


def _node_keyed_pairs_and_costs():
    """A small node-keyed (pairs, odm) — all three tiers populated.

    cells_to_cells (cell-node → cell-node):
        N0 → [N0, N1]    N1 → [N1, N2]    N2 → [N2]
    cells_to_zones (cell-node → zone-node):
        N0 → [ZN1]       N1 → [ZN1]
    zones_to_zones (zone-node → zone-node):
        ZN0 → [ZN1]      ZN1 → [ZN0]
    """
    cells_pairs = {
        "N0": np.array(["N0", "N1"]),
        "N1": np.array(["N1", "N2"]),
        "N2": np.array(["N2"]),
    }
    cells_costs = {
        "N0": np.array([0.0, 100.0]),
        "N1": np.array([0.0, 50.0]),
        "N2": np.array([0.0]),
    }
    c2z_pairs = {
        "N0": np.array(["ZN1"]),
        "N1": np.array(["ZN1"]),
    }
    c2z_costs = {
        "N0": np.array([200.0]),
        "N1": np.array([180.0]),
    }
    zones_pairs = {
        "ZN0": np.array(["ZN1"]),
        "ZN1": np.array(["ZN0"]),
    }
    zones_costs = {
        "ZN0": np.array([300.0]),
        "ZN1": np.array([350.0]),
    }
    pairs = TieredODNodePairs(
        cells_to_cells=cells_pairs,
        cells_to_zones=c2z_pairs,
        zones_to_zones=zones_pairs,
    )
    odm = TieredODNodePairs(
        cells_to_cells=cells_costs,
        cells_to_zones=c2z_costs,
        zones_to_zones=zones_costs,
    )
    return pairs, odm


class ReindexByGeoUnitTestCase(unittest.TestCase):
    """Conversion semantics: node-keyed → geo-keyed with fan-out."""

    def setUp(self):
        self.cells, self.zones = _build_fixture()
        self.pairs_n, self.odm_n = _node_keyed_pairs_and_costs()

    def test_returns_geo_pairs_subclasses(self):
        new_pairs, new_odm = reindex_by_geo_unit(
            self.pairs_n,
            self.odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )
        self.assertIsInstance(new_pairs, TieredODGeoPairs)
        self.assertIsInstance(new_odm, TieredODGeoPairs)

    def test_cells_to_cells_fan_out_and_sort(self):
        """Cells sharing the same dest node fan out; dest arrays sorted by cell_id."""
        new_pairs, new_odm = reindex_by_geo_unit(
            self.pairs_n,
            self.odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )
        # Origin C0 (at N0): original dests N0 -> {C0, C1}, N1 -> {C2}
        # → geo-dests: [C0, C1, C2] sorted, with costs [0, 0, 100].
        self.assertEqual(list(new_pairs.cells_to_cells["C0"]), ["C0", "C1", "C2"])
        np.testing.assert_array_equal(new_odm.cells_to_cells["C0"], np.array([0.0, 0.0, 100.0]))
        # Origin C1 also at N0 — same outgoing as C0.
        self.assertEqual(list(new_pairs.cells_to_cells["C1"]), ["C0", "C1", "C2"])
        np.testing.assert_array_equal(new_odm.cells_to_cells["C1"], np.array([0.0, 0.0, 100.0]))

    def test_cells_to_cells_full_coverage(self):
        """Every cell in `cells` appears as an origin key in the geo-keyed result."""
        new_pairs, _ = reindex_by_geo_unit(
            self.pairs_n,
            self.odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )
        self.assertEqual(set(new_pairs.cells_to_cells.keys()), {"C0", "C1", "C2", "C3"})

    def test_cells_to_zones_geo_keyed(self):
        """cells_to_zones origins fan out to cells at the origin node; dests
        translate from zone-node to zone-id."""
        new_pairs, new_odm = reindex_by_geo_unit(
            self.pairs_n,
            self.odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )
        # C0 and C1 share node N0 → both get N0's cells_to_zones entry [ZN1 → Z1].
        assert new_pairs.cells_to_zones is not None
        self.assertEqual(list(new_pairs.cells_to_zones["C0"]), ["Z1"])
        self.assertEqual(list(new_pairs.cells_to_zones["C1"]), ["Z1"])
        assert new_odm.cells_to_zones is not None
        np.testing.assert_array_equal(new_odm.cells_to_zones["C0"], np.array([200.0]))
        # C2 at N1 → [Z1] cost 180.
        self.assertEqual(list(new_pairs.cells_to_zones["C2"]), ["Z1"])
        np.testing.assert_array_equal(new_odm.cells_to_zones["C2"], np.array([180.0]))
        # C3 at N2 has no cells_to_zones origin in the input → absent.
        self.assertNotIn("C3", new_pairs.cells_to_zones)

    def test_zones_to_zones_geo_keyed(self):
        new_pairs, new_odm = reindex_by_geo_unit(
            self.pairs_n,
            self.odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )
        # Z0 (at ZN0): dest ZN1 → Z1. Cost 300.
        self.assertEqual(list(new_pairs.zones_to_zones["Z0"]), ["Z1"])
        np.testing.assert_array_equal(new_odm.zones_to_zones["Z0"], np.array([300.0]))
        # Z1 (at ZN1): dest ZN0 → Z0. Cost 350.
        self.assertEqual(list(new_pairs.zones_to_zones["Z1"]), ["Z0"])
        np.testing.assert_array_equal(new_odm.zones_to_zones["Z1"], np.array([350.0]))

    def test_odm_none_returns_pairs_only(self):
        new_pairs, new_odm = reindex_by_geo_unit(
            self.pairs_n,
            None,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )
        self.assertIsInstance(new_pairs, TieredODGeoPairs)
        self.assertIsNone(new_odm)
        self.assertEqual(set(new_pairs.cells_to_cells.keys()), {"C0", "C1", "C2", "C3"})

    def test_missing_zones_when_zone_tier_present_raises(self):
        with self.assertRaisesRegex(ValueError, "zones.*required"):
            reindex_by_geo_unit(
                self.pairs_n,
                self.odm_n,
                self.cells,
                cell_node_column="node_id",
                # no zones / zone_node_column
            )

    def test_cells_with_nan_node_skipped(self):
        cells = self.cells.copy()
        cells.loc["C3", "node_id"] = np.nan
        new_pairs, _ = reindex_by_geo_unit(
            self.pairs_n,
            self.odm_n,
            cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )
        self.assertNotIn("C3", new_pairs.cells_to_cells)


class DestValuesGeoTestCase(unittest.TestCase):
    """`dest_values_geo` — destination value lookup on geo-keyed pairs."""

    def setUp(self):
        self.cells, self.zones = _build_fixture()
        pairs_n, odm_n = _node_keyed_pairs_and_costs()
        self.pairs_geo, _ = reindex_by_geo_unit(
            pairs_n,
            odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )

    def test_cells_to_cells_values_per_cell_no_summing(self):
        """Unlike node-keyed `dest_values` (which sums values across cells at
        a node), `dest_values_geo` returns the per-cell value directly."""
        v = dest_values_geo("population", self.pairs_geo, self.cells, zones=self.zones)
        # Origin C0: dests [C0, C1, C2] → populations [10, 20, 5].
        np.testing.assert_array_equal(v.cells_to_cells["C0"], np.array([10.0, 20.0, 5.0]))

    def test_cells_to_zones_per_zone(self):
        v = dest_values_geo("population", self.pairs_geo, self.cells, zones=self.zones)
        # C0 → Z1 with population 7.
        assert v.cells_to_zones is not None
        np.testing.assert_array_equal(v.cells_to_zones["C0"], np.array([7.0]))

    def test_zones_to_zones_per_zone(self):
        v = dest_values_geo("population", self.pairs_geo, self.cells, zones=self.zones)
        # Z0 → Z1 with population 7.
        np.testing.assert_array_equal(v.zones_to_zones["Z0"], np.array([7.0]))

    def test_missing_column_raises(self):
        with self.assertRaisesRegex(ValueError, "missing column"):
            dest_values_geo("nonexistent", self.pairs_geo, self.cells, zones=self.zones)


class AddGeoOverheadsTestCase(unittest.TestCase):
    """`add_geo_overheads` — four independent overhead lookups, generic."""

    def setUp(self):
        self.cells, self.zones = _build_fixture()
        pairs_n, odm_n = _node_keyed_pairs_and_costs()
        self.pairs, self.costs = reindex_by_geo_unit(
            pairs_n,
            odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )

    def test_origin_cell_affects_cells_to_cells_and_cells_to_zones(self):
        out = add_geo_overheads(
            self.costs, self.pairs, origin_cell=pd.Series({"C0": 10.0, "C1": 20.0})
        )
        # C0 (cell-tier): every outgoing cost +10.
        np.testing.assert_array_equal(
            out.cells_to_cells["C0"], self.costs.cells_to_cells["C0"] + 10.0
        )
        # C0 (middle-tier): every outgoing cost +10 as well.
        np.testing.assert_array_equal(
            out.cells_to_zones["C0"], self.costs.cells_to_zones["C0"] + 10.0
        )
        # C2 not in lookup → unchanged.
        np.testing.assert_array_equal(out.cells_to_cells["C2"], self.costs.cells_to_cells["C2"])
        # Zone tier untouched.
        np.testing.assert_array_equal(out.zones_to_zones["Z0"], self.costs.zones_to_zones["Z0"])

    def test_origin_zone_only_affects_zones_to_zones(self):
        out = add_geo_overheads(self.costs, self.pairs, origin_zone=pd.Series({"Z0": 50.0}))
        # Z0 zone-tier: +50.
        np.testing.assert_array_equal(
            out.zones_to_zones["Z0"], self.costs.zones_to_zones["Z0"] + 50.0
        )
        # Z1 not in lookup → unchanged.
        np.testing.assert_array_equal(out.zones_to_zones["Z1"], self.costs.zones_to_zones["Z1"])
        # Cell and middle tiers untouched.
        np.testing.assert_array_equal(out.cells_to_cells["C0"], self.costs.cells_to_cells["C0"])
        np.testing.assert_array_equal(out.cells_to_zones["C0"], self.costs.cells_to_zones["C0"])

    def test_dest_cell_adds_per_dest(self):
        out = add_geo_overheads(
            self.costs, self.pairs, dest_cell=pd.Series({"C0": 1.0, "C1": 2.0, "C2": 3.0})
        )
        # C0 → dests [C0, C1, C2]: + [1, 2, 3].
        np.testing.assert_array_equal(
            out.cells_to_cells["C0"], self.costs.cells_to_cells["C0"] + np.array([1.0, 2.0, 3.0])
        )

    def test_dest_zone_adds_per_dest_zone_at_both_tiers(self):
        out = add_geo_overheads(self.costs, self.pairs, dest_zone=pd.Series({"Z1": 25.0}))
        # Z0 → dest Z1 at far tier: +25.
        np.testing.assert_array_equal(
            out.zones_to_zones["Z0"], self.costs.zones_to_zones["Z0"] + 25.0
        )
        # C0 → dest Z1 at middle tier: +25.
        np.testing.assert_array_equal(
            out.cells_to_zones["C0"], self.costs.cells_to_zones["C0"] + 25.0
        )

    def test_returns_geo_subclass_not_mutating_input(self):
        out = add_geo_overheads(self.costs, self.pairs, origin_cell=pd.Series({"C0": 1.0}))
        self.assertIsInstance(out, TieredODGeoPairs)
        # Input unchanged.
        np.testing.assert_array_equal(self.costs.cells_to_cells["C0"], np.array([0.0, 0.0, 100.0]))


class AddOriginCellOverheadTestCase(unittest.TestCase):
    """`add_origin_cell_overhead` — per-cell at cells_to_cells and
    cells_to_zones tiers, per-zone-mean at the zones_to_zones tier."""

    def setUp(self):
        self.cells, self.zones = _build_fixture()
        pairs_n, odm_n = _node_keyed_pairs_and_costs()
        self.pairs, self.costs = reindex_by_geo_unit(
            pairs_n,
            odm_n,
            self.cells,
            cell_node_column="node_id",
            zones=self.zones,
            zone_node_column="node_id",
        )

    def test_per_cell_baked_at_cell_tier(self):
        out = add_origin_cell_overhead(self.costs, self.pairs, self.cells, "walk_overhead_s")
        # C0 has overhead 30 → +30 on every cell-tier outgoing cost.
        np.testing.assert_array_equal(
            out.cells_to_cells["C0"], self.costs.cells_to_cells["C0"] + 30.0
        )
        # C1 has overhead 60 → +60.
        np.testing.assert_array_equal(
            out.cells_to_cells["C1"], self.costs.cells_to_cells["C1"] + 60.0
        )

    def test_per_cell_baked_at_middle_tier(self):
        out = add_origin_cell_overhead(self.costs, self.pairs, self.cells, "walk_overhead_s")
        # C0 overhead 30 → +30 on cells_to_zones origin C0.
        np.testing.assert_array_equal(
            out.cells_to_zones["C0"], self.costs.cells_to_zones["C0"] + 30.0
        )

    def test_zone_mean_baked_at_zone_tier(self):
        out = add_origin_cell_overhead(self.costs, self.pairs, self.cells, "walk_overhead_s")
        # Z0 contains C0, C1, C2 with overheads 30, 60, 45 → mean 45.
        np.testing.assert_array_equal(
            out.zones_to_zones["Z0"], self.costs.zones_to_zones["Z0"] + 45.0
        )
        # Z1 contains only C3 with overhead 90 → mean 90.
        np.testing.assert_array_equal(
            out.zones_to_zones["Z1"], self.costs.zones_to_zones["Z1"] + 90.0
        )

    def test_works_when_zone_tier_absent(self):
        """No zone tier in costs → no zone_id_column requirement."""
        cells_only_costs = TieredODGeoPairs(
            cells_to_cells=dict(self.costs.cells_to_cells),
        )
        cells_only_pairs = TieredODGeoPairs(
            cells_to_cells=dict(self.pairs.cells_to_cells),
        )
        # Remove zone_id column to prove it's not required when no zone tier.
        cells_no_zone = self.cells.drop(columns="zone_id")
        out = add_origin_cell_overhead(
            cells_only_costs, cells_only_pairs, cells_no_zone, "walk_overhead_s"
        )
        np.testing.assert_array_equal(
            out.cells_to_cells["C0"], self.costs.cells_to_cells["C0"] + 30.0
        )


class ReindexPerCellTierFilterTestCase(unittest.TestCase):
    """Per-cell zone-pair tier filtering at reindex — eliminates the
    multi-zone-snap-node union artifact where a destination ends up in
    BOTH c2c (as individual cells) and c2z (as aggregated zone) for the
    same origin cell.

    Scenario: zones on a line at y=0 — ZA(x=0), ZB(x=900), ZC(x=1700),
    ZD(x=3000) — with r_cells=1000, r_medium=4000, r_zones=10000:

        zone | cell-tier  | c2z         | z2z
        ZA   | {ZA,ZB}    | {ZC,ZD}     | {}
        ZB   | {ZA,ZB,ZC} | {ZD}        | {}
        ZC   | {ZB,ZC}    | {ZA,ZD}     | {}
        ZD   | {ZD}       | {ZA,ZB,ZC}  | {}

    Cells: C_A1∈ZA and C_B1∈ZB both snap to `Nshared` (the multi-zone
    node — the source of the artifact). C_C1, C_C2 in ZC snap to N_C,
    N_C2 respectively; C_D1 in ZD snaps to N_D.

    Because cells in ZA and ZB share `Nshared`, the node-keyed pair-builder
    unions their tier-dest sets at that node. The asymmetry between ZA's
    cell-tier ({ZA,ZB}) and ZB's ({ZA,ZB,ZC}) means `Nshared`'s c2c at the
    node level pulls in ZC's cells. At reindex (without filter), every cell
    on `Nshared` inherits the union — so C_A1 ends up with ZC cells in c2c
    *and* ZN_C in c2z. That's the double-count bug; the per-cell filter
    eliminates it.
    """

    R_CELLS = 1000.0
    R_MEDIUM = 4000.0
    R_ZONES = 10000.0

    @staticmethod
    def _fixture():
        cells = pd.DataFrame(
            {
                "node_id": ["Nshared", "Nshared", "N_C", "N_C2", "N_D"],
                "zone_id": ["ZA", "ZB", "ZC", "ZC", "ZD"],
            },
            index=pd.Index(["C_A1", "C_B1", "C_C1", "C_C2", "C_D1"], name="cell_id"),
        )
        zones = gpd.GeoDataFrame(
            {"node_id": ["ZN_A", "ZN_B", "ZN_C", "ZN_D"]},
            geometry=[Point(0, 0), Point(900, 0), Point(1700, 0), Point(3000, 0)],
            index=pd.Index(["ZA", "ZB", "ZC", "ZD"], name="zone_id"),
            crs="EPSG:2056",
        )
        return cells, zones

    @staticmethod
    def _node_keyed_pairs():
        """Hand-built `TieredODNodePairs` matching what `get_pairs` would emit
        on the fixture above. `Nshared` entries embody the union artifact.
        """
        return TieredODNodePairs(
            cells_to_cells={
                # Union of ZA's & ZB's cell-tier dest nodes — includes ZC's nodes
                # because ZB's cell-tier reaches ZC, even though ZA's doesn't.
                "Nshared": np.array(["N_C", "N_C2", "Nshared"]),
                "N_C": np.array(["N_C", "N_C2", "Nshared"]),
                "N_C2": np.array(["N_C", "N_C2", "Nshared"]),
                "N_D": np.array(["N_D"]),
            },
            cells_to_zones={
                # Union of ZA's c2z ({ZN_C,ZN_D}) and ZB's c2z ({ZN_D}).
                "Nshared": np.array(["ZN_C", "ZN_D"]),
                "N_C": np.array(["ZN_A", "ZN_D"]),
                "N_C2": np.array(["ZN_A", "ZN_D"]),
                "N_D": np.array(["ZN_A", "ZN_B", "ZN_C"]),
            },
            zones_to_zones={},  # empty under these radii (no zone pair >= r_medium)
        )

    EXPECTED_CELL_TIER = {
        "ZA": frozenset({"ZA", "ZB"}),
        "ZB": frozenset({"ZA", "ZB", "ZC"}),
        "ZC": frozenset({"ZB", "ZC"}),
        "ZD": frozenset({"ZD"}),
    }
    EXPECTED_C2Z = {
        "ZA": frozenset({"ZC", "ZD"}),
        "ZB": frozenset({"ZD"}),
        "ZC": frozenset({"ZA", "ZD"}),
        "ZD": frozenset({"ZA", "ZB", "ZC"}),
    }

    def setUp(self):
        self.cells, self.zones = self._fixture()
        self.pairs = self._node_keyed_pairs()

    def _filtered(self):
        return reindex_by_geo_unit(
            self.pairs, None, self.cells,
            cell_node_column="node_id",
            zones=self.zones, zone_node_column="node_id",
            r_cells=self.R_CELLS, r_medium=self.R_MEDIUM, r_zones=self.R_ZONES,
        )[0]

    def _unfiltered(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return reindex_by_geo_unit(
                self.pairs, None, self.cells,
                cell_node_column="node_id",
                zones=self.zones, zone_node_column="node_id",
            )[0]

    def test_unfiltered_reindex_reproduces_bug(self):
        """Documents the bug: without the per-cell tier filter, ZC ends up
        in BOTH c2c (as ZC cells C_C1/C_C2) and c2z (as ZN_C) for C_A1.
        This test PASSES only as long as the unfiltered path stays buggy —
        flagging if someone ever changes the legacy behavior."""
        out = self._unfiltered()
        c2c_dest_zones_for_A1 = {
            self.cells.loc[c, "zone_id"] for c in out.cells_to_cells["C_A1"]
        }
        c2z_dest_zones_for_A1 = set(out.cells_to_zones["C_A1"])
        overlap = c2c_dest_zones_for_A1 & c2z_dest_zones_for_A1
        self.assertIn(
            "ZC", overlap,
            msg="expected the multi-zone-snap-node union artifact to place "
                "ZC in both c2c and c2z for C_A1 when filter is OFF",
        )

    def test_filter_eliminates_double_counting(self):
        out = self._filtered()
        for cell_id in self.cells.index:
            c2c_dest_zones = {
                self.cells.loc[c, "zone_id"]
                for c in out.cells_to_cells.get(cell_id, [])
            }
            c2z_dest_zones = set(out.cells_to_zones.get(cell_id, []))
            overlap = c2c_dest_zones & c2z_dest_zones
            self.assertEqual(
                overlap, set(),
                msg=f"{cell_id}: c2c parent zones {c2c_dest_zones} and c2z zones "
                    f"{c2z_dest_zones} must be disjoint (overlap={overlap})",
            )

    def test_filter_assigns_correct_tier_per_cell(self):
        """c2c dest cells' parent zones must lie in the origin cell's cell-tier
        zone set; c2z dest zones must lie in the origin cell's c2z zone set."""
        out = self._filtered()
        for cell_id in self.cells.index:
            origin_zone = self.cells.loc[cell_id, "zone_id"]
            allowed_cell = self.EXPECTED_CELL_TIER[origin_zone]
            allowed_c2z = self.EXPECTED_C2Z[origin_zone]
            c2c_dest_zones = {
                self.cells.loc[c, "zone_id"]
                for c in out.cells_to_cells.get(cell_id, [])
            }
            c2z_dest_zones = set(out.cells_to_zones.get(cell_id, []))
            self.assertTrue(
                c2c_dest_zones.issubset(allowed_cell),
                msg=f"{cell_id} ({origin_zone}): c2c dest zones {c2c_dest_zones} "
                    f"not within cell-tier {set(allowed_cell)}",
            )
            self.assertTrue(
                c2z_dest_zones.issubset(allowed_c2z),
                msg=f"{cell_id} ({origin_zone}): c2z dest zones {c2z_dest_zones} "
                    f"not within c2z {set(allowed_c2z)}",
            )

    def test_filter_is_lossless_for_tier_correct_dests(self):
        """Filter only drops tier-incorrect dests; every dest in the unfiltered
        output that *belongs* to the origin's tier (per the zone-pair classification)
        survives the filter."""
        unfiltered = self._unfiltered()
        filtered = self._filtered()
        for cell_id in self.cells.index:
            origin_zone = self.cells.loc[cell_id, "zone_id"]
            allowed_cell = self.EXPECTED_CELL_TIER[origin_zone]
            allowed_c2z = self.EXPECTED_C2Z[origin_zone]
            # c2c: every unfiltered dest whose parent zone is in allowed_cell
            # should be present in filtered output.
            uf_c2c_kept = {
                c for c in unfiltered.cells_to_cells.get(cell_id, [])
                if self.cells.loc[c, "zone_id"] in allowed_cell
            }
            f_c2c = set(filtered.cells_to_cells.get(cell_id, []))
            self.assertEqual(
                uf_c2c_kept, f_c2c,
                msg=f"{cell_id}: filter dropped tier-correct c2c dests "
                    f"(expected {uf_c2c_kept}, got {f_c2c})",
            )
            # c2z: same check.
            uf_c2z_kept = {
                z for z in unfiltered.cells_to_zones.get(cell_id, [])
                if z in allowed_c2z
            }
            f_c2z = set(filtered.cells_to_zones.get(cell_id, []))
            self.assertEqual(
                uf_c2z_kept, f_c2z,
                msg=f"{cell_id}: filter dropped tier-correct c2z dests "
                    f"(expected {uf_c2z_kept}, got {f_c2z})",
            )

    def test_costs_aligned_after_filter(self):
        """When an ODM is reindexed alongside pairs, filtered cost arrays must
        stay aligned with filtered dest arrays (same length, same drops applied)."""
        # Build a parallel cost ODM with sentinel values to track which entries
        # survive: encode the cost as `node_hash + dest_index`, so misalignment
        # would be immediately visible.
        odm = TieredODNodePairs(
            cells_to_cells={
                k: np.arange(len(v), dtype=float) + 1000.0
                for k, v in self.pairs.cells_to_cells.items()
            },
            cells_to_zones={
                k: np.arange(len(v), dtype=float) + 2000.0
                for k, v in self.pairs.cells_to_zones.items()
            },
        )
        new_pairs, new_odm = reindex_by_geo_unit(
            self.pairs, odm, self.cells,
            cell_node_column="node_id",
            zones=self.zones, zone_node_column="node_id",
            r_cells=self.R_CELLS, r_medium=self.R_MEDIUM, r_zones=self.R_ZONES,
        )
        for cell_id in self.cells.index:
            for tier_name in ("cells_to_cells", "cells_to_zones"):
                p = getattr(new_pairs, tier_name).get(cell_id)
                v = getattr(new_odm, tier_name).get(cell_id)
                if p is None and v is None:
                    continue
                self.assertEqual(
                    len(p), len(v),
                    msg=f"{cell_id} {tier_name}: pair/odm length mismatch",
                )

    def test_warning_emitted_without_radii(self):
        """When `zones` is given but the radii are not, a UserWarning fires so
        callers know they're getting the legacy unfiltered (buggy) path."""
        with self.assertWarns(UserWarning):
            reindex_by_geo_unit(
                self.pairs, None, self.cells,
                cell_node_column="node_id",
                zones=self.zones, zone_node_column="node_id",
            )

    def test_partial_radii_raises(self):
        """Passing one of (r_cells, r_medium) without the other is a user
        error — better to fail fast than silently disable the filter."""
        with self.assertRaises(ValueError):
            reindex_by_geo_unit(
                self.pairs, None, self.cells,
                cell_node_column="node_id",
                zones=self.zones, zone_node_column="node_id",
                r_cells=self.R_CELLS,  # r_medium missing
            )

    def test_filter_requires_zone_id_column(self):
        cells_no_zone = self.cells.drop(columns="zone_id")
        with self.assertRaises(ValueError):
            reindex_by_geo_unit(
                self.pairs, None, cells_no_zone,
                cell_node_column="node_id",
                zones=self.zones, zone_node_column="node_id",
                r_cells=self.R_CELLS, r_medium=self.R_MEDIUM,
            )


if __name__ == "__main__":
    unittest.main()
