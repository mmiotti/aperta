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

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
