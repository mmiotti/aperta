"""Tests for `aperta.traffic_flows`.

Run with:
    python -m unittest tests.test_traffic_flows

Covers `nested_node_sample` — sample destinations for sampled origins over
the combined row of all three OD tiers (cells_to_cells, cells_to_zones,
zones_to_zones), with per-cost-bin reweighting — and `percentile_bin_edges`.
"""

import unittest

import numpy as np

from aperta.od_pairs import TieredODNodePairs
from aperta.traffic_flows import (
    _bin_adjusted_scores,
    nested_node_sample,
    percentile_bin_edges,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_tiered_inputs():
    """4 cells in 2 zones.

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
            "c1": np.array(["c1", "c2"]),
            "c2": np.array(["c1", "c2"]),
            "c3": np.array(["c3", "c4"]),
            "c4": np.array(["c3", "c4"]),
        },
        zones_to_zones={
            "Z1": np.array(["Z2"]),
            "Z2": np.array(["Z1"]),
        },
    )
    weights = TieredODNodePairs(
        cells_to_cells={k: np.ones(len(v)) for k, v in pairs.cells_to_cells.items()},
        zones_to_zones={k: np.array([10.0]) for k in pairs.zones_to_zones},
    )
    costs = TieredODNodePairs(
        cells_to_cells={
            "c1": np.array([0.0, 100.0]),
            "c2": np.array([100.0, 0.0]),
            "c3": np.array([0.0, 100.0]),
            "c4": np.array([100.0, 0.0]),
        },
        zones_to_zones={
            "Z1": np.array([500.0]),
            "Z2": np.array([500.0]),
        },
    )
    cell_to_zone_node = {"c1": "Z1", "c2": "Z1", "c3": "Z2", "c4": "Z2"}
    orig_weights = np.array([1.0, 1.0, 1.0, 1.0])
    return pairs, weights, costs, cell_to_zone_node, orig_weights


# One bin covering every toy cost: sampling is then proportional to weights alone.
_ONE_BIN = np.array([0.0, 1_000.0])


# ---------------------------------------------------------------------------
# `nested_node_sample` — weighted-sampled origin / destination pairs
# ---------------------------------------------------------------------------


class NestedNodeSampleTestCase(unittest.TestCase):
    def setUp(self):
        (self.pairs, self.weights, self.costs, self.c2z, self.orig_weights) = _toy_tiered_inputs()

    def test_returns_dict_keyed_by_origin_cells(self):
        rs = np.random.RandomState(42)
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=self.orig_weights,
            bin_edges=_ONE_BIN,
            n_orig=4,
            n_dest=10,
            random_state=rs,
        )
        self.assertIsInstance(out, dict)
        for k in out.keys():
            self.assertIn(k, ("c1", "c2", "c3", "c4"))

    def test_n_dest_per_origin(self):
        """With-replacement origin sampling: each origin gets a multiple
        of `n_dest` destinations (== `n_picks × n_dest`), and the total
        across all origins equals the nominal sample budget
        `n_orig × n_dest`."""
        rs = np.random.RandomState(42)
        n_orig, n_dest = 4, 15
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=self.orig_weights,
            bin_edges=_ONE_BIN,
            n_orig=n_orig,
            n_dest=n_dest,
            random_state=rs,
        )
        # Every origin's array length is `n_picks × n_dest`.
        for dests in out.values():
            self.assertEqual(len(dests) % n_dest, 0)
            self.assertGreaterEqual(len(dests), n_dest)
        # Total OD pairs sampled is exactly `n_orig × n_dest` regardless
        # of how the picks distribute across unique origins — this is what
        # makes AADT scaling clean (no dedup correction needed).
        self.assertEqual(sum(len(d) for d in out.values()), n_orig * n_dest)

    def test_origin_weight_concentration(self):
        """All origin-weight on c2 → c2 is the only sampled origin. Under
        with-replacement sampling, c2 is picked `n_orig` times and so
        gets `n_orig × n_dest` destinations."""
        rs = np.random.RandomState(42)
        n_orig, n_dest = 4, 10
        weights_c2_only = np.array([0.0, 1.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=weights_c2_only,
            bin_edges=_ONE_BIN,
            n_orig=n_orig,
            n_dest=n_dest,
            random_state=rs,
        )
        self.assertEqual(list(out.keys()), ["c2"])
        self.assertEqual(len(out["c2"]), n_orig * n_dest)

    def test_single_bin_samples_proportional_to_weights(self):
        """With one bin, every tier's dests share it: c1's pool is c1 (w=1),
        c2 (w=1) and Z2 (w=10), so Z2 is drawn ~10/12 of the time."""
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=c1_only,
            bin_edges=_ONE_BIN,
            n_orig=1,
            n_dest=20_000,
            random_state=rs,
        )
        share_z2 = (out["c1"] == "Z2").mean()
        self.assertAlmostEqual(share_z2, 10 / 12, delta=0.01)

    def test_zone_tier_dests_appear_when_weighted_up(self):
        """If zone-tier dest weights are boosted enough, zone-tier dests
        (here Z2 reached from c1's zone Z1) start appearing in the output."""
        big_zone_weights = TieredODNodePairs(
            cells_to_cells=self.weights.cells_to_cells,
            zones_to_zones={k: np.array([10_000.0]) for k in self.pairs.zones_to_zones},
        )
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            big_zone_weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=c1_only,
            bin_edges=_ONE_BIN,
            n_orig=1,
            n_dest=1000,
            random_state=rs,
        )
        self.assertIn("Z2", set(out["c1"].tolist()))

    def test_reproducible_with_random_state(self):
        rs1 = np.random.RandomState(42)
        rs2 = np.random.RandomState(42)
        out1 = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=self.orig_weights,
            bin_edges=_ONE_BIN,
            n_orig=4,
            n_dest=10,
            random_state=rs1,
        )
        out2 = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=self.orig_weights,
            bin_edges=_ONE_BIN,
            n_orig=4,
            n_dest=10,
            random_state=rs2,
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
                "c1": np.array([False, True]),
                "c2": np.array([True, True]),
                "c3": np.array([True, True]),
                "c4": np.array([True, True]),
            },
            zones_to_zones={k: np.array([True]) for k in self.pairs.zones_to_zones},
        )
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=c1_only,
            bin_edges=_ONE_BIN,
            n_orig=1,
            n_dest=200,
            random_state=rs,
            mask=mask,
        )
        self.assertNotIn("c1", set(out["c1"].tolist()))

    def test_mask_filters_zone_tier_dests(self):
        """A zone-tier mask removes zone-tier dests for the affected zones."""
        mask = TieredODNodePairs(
            cells_to_cells={
                k: np.ones(len(v), dtype=bool) for k, v in self.pairs.cells_to_cells.items()
            },
            # Z1's outgoing zone-tier dest (Z2) is masked out.
            zones_to_zones={"Z1": np.array([False]), "Z2": np.array([True])},
        )
        big_zone_weights = TieredODNodePairs(
            cells_to_cells=self.weights.cells_to_cells,
            zones_to_zones={k: np.array([10_000.0]) for k in self.pairs.zones_to_zones},
        )
        rs = np.random.RandomState(42)
        # Sample only c1 and c3 (so we deterministically check both branches).
        c1_and_c3 = np.array([1.0, 0.0, 1.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            big_zone_weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=c1_and_c3,
            bin_edges=_ONE_BIN,
            n_orig=20,
            n_dest=200,
            random_state=rs,
            mask=mask,
        )
        # c1 (in Z1) had Z2 as zone-tier dest — should now be absent.
        self.assertNotIn("Z2", set(out["c1"].tolist()))
        # c3 (in Z2) still has Z1 as zone-tier dest — should still appear.
        self.assertIn("Z1", set(out["c3"].tolist()))


class NestedNodeSampleMiddleTierTestCase(unittest.TestCase):
    """Verifies the middle-tier (`cells_to_zones`) integration: per-cell
    cell-origin → zone-node dest pairs participate in the flattened sampling
    pool alongside cell- and far-tier dests.
    """

    def setUp(self):
        # Extend the 2-tier fixture with a middle tier: c1 reaches a single
        # middle-tier dest 'M1' at cost 200 with high weight; c2 reaches 'M2'
        # similarly. (Per Phase B semantics, cells in the same zone could
        # share the same dest zones; here we use distinct destinations so we
        # can distinguish c1's vs c2's middle-tier output deterministically.)
        (cells_pairs, cells_weights, cells_costs, c2z_map, orig_w) = _toy_tiered_inputs()
        self.pairs = TieredODNodePairs(
            cells_to_cells=cells_pairs.cells_to_cells,
            cells_to_zones={
                "c1": np.array(["M1"]),
                "c2": np.array(["M2"]),
                # c3, c4: no middle-tier dests — sampling falls back to other tiers.
            },
            zones_to_zones=cells_pairs.zones_to_zones,
        )
        self.weights = TieredODNodePairs(
            cells_to_cells=cells_weights.cells_to_cells,
            cells_to_zones={"c1": np.array([10_000.0]), "c2": np.array([10_000.0])},
            zones_to_zones=cells_weights.zones_to_zones,
        )
        self.costs = TieredODNodePairs(
            cells_to_cells=cells_costs.cells_to_cells,
            cells_to_zones={"c1": np.array([200.0]), "c2": np.array([200.0])},
            zones_to_zones=cells_costs.zones_to_zones,
        )
        self.c2z = c2z_map
        self.orig_weights = orig_w

    def test_middle_tier_dests_appear_when_weighted_up(self):
        """With middle-tier weights boosted, the middle-tier dest shows up
        in the per-cell sample — and the per-cell dest set differs between
        cells in the same zone (cells_to_zones is cell-keyed)."""
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=c1_only,
            bin_edges=_ONE_BIN,
            n_orig=1,
            n_dest=1000,
            random_state=rs,
        )
        self.assertIn("M1", set(out["c1"].tolist()))
        # c1 only routes to M1 (its own middle-tier dest), never M2 (c2's).
        self.assertNotIn("M2", set(out["c1"].tolist()))

    def test_middle_tier_mask_filters_dests(self):
        """A cells_to_zones mask removes specific middle-tier dests."""
        mask = TieredODNodePairs(
            cells_to_cells={
                k: np.ones(len(v), dtype=bool) for k, v in self.pairs.cells_to_cells.items()
            },
            cells_to_zones={
                "c1": np.array([False]),  # drop c1's M1
                "c2": np.array([True]),
            },
            zones_to_zones={
                k: np.ones(len(v), dtype=bool) for k, v in self.pairs.zones_to_zones.items()
            },
        )
        rs = np.random.RandomState(42)
        c1_only = np.array([1.0, 0.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=c1_only,
            bin_edges=_ONE_BIN,
            n_orig=1,
            n_dest=500,
            random_state=rs,
            mask=mask,
        )
        self.assertNotIn("M1", set(out["c1"].tolist()))

    def test_cells_without_middle_tier_entry_still_work(self):
        """Cells absent from `cells_to_zones` use the empty fallback — no
        crash, sampling just draws from cell + far tiers."""
        rs = np.random.RandomState(42)
        c3_only = np.array([0.0, 0.0, 1.0, 0.0])  # c3 has no cells_to_zones entry
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=c3_only,
            bin_edges=_ONE_BIN,
            n_orig=1,
            n_dest=100,
            random_state=rs,
        )
        # All sampled dests should be from c3's cell-tier or zone-tier pool.
        dests = set(out["c3"].tolist())
        allowed = {"c3", "c4", "Z1"}  # cell-tier dests + zone-tier dest from Z2
        self.assertTrue(dests.issubset(allowed), f"Unexpected dests: {dests - allowed}")


# ---------------------------------------------------------------------------
# Cost-bin reweighting
# ---------------------------------------------------------------------------


class PercentileBinEdgesTestCase(unittest.TestCase):
    def test_uniform_input_yields_equal_spacing(self):
        edges = percentile_bin_edges(np.linspace(0.0, 100.0, 1001), n_bins=10)
        self.assertEqual(edges.shape, (11,))
        np.testing.assert_allclose(np.diff(edges), 10.0, atol=0.1)

    def test_skewed_input_yields_unequal_spacing(self):
        """Each bin should still contain ~1/n of the data, so highly-skewed
        input produces unequal-width bins concentrating where data is dense."""
        rng = np.random.default_rng(0)
        skewed = rng.exponential(scale=1.0, size=10_000)
        edges = percentile_bin_edges(skewed, n_bins=20)
        bin_counts, _ = np.histogram(skewed, bins=edges)
        # Each bin should hold ~ 500 = 10_000/20 samples within 10%.
        np.testing.assert_allclose(bin_counts, 500, rtol=0.10)

    def test_drops_non_finite(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
        edges = percentile_bin_edges(data, n_bins=3)
        # Should equal percentiles of [1, 2, 3, 4].
        np.testing.assert_allclose(
            edges, np.percentile([1.0, 2.0, 3.0, 4.0], [0, 33.3, 66.7, 100]), atol=0.1
        )

    def test_empty_after_drop_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            percentile_bin_edges(np.array([np.nan, np.inf]))


class BinAdjustedScoresTestCase(unittest.TestCase):
    """Per-row reweighting: equal mass per populated bin, split by weight."""

    def setUp(self):
        self.bin_edges = np.array([0.0, 10.0, 20.0, 40.0, 60.0])  # 4 bins

    def test_each_populated_bin_gets_equal_mass(self):
        costs = np.array([5.0, 5.0, 55.0, 55.0, 55.0, 55.0])  # bins 0 and 3
        scores = _bin_adjusted_scores(costs, np.ones(6), self.bin_edges)
        per_bin = np.bincount(np.digitize(costs, self.bin_edges) - 1, weights=scores, minlength=4)
        np.testing.assert_allclose(per_bin[[0, 3]] / scores.sum(), [0.5, 0.5], rtol=1e-12)
        np.testing.assert_allclose(per_bin[[1, 2]], 0.0, atol=1e-12)

    def test_within_bin_split_follows_weights(self):
        scores = _bin_adjusted_scores(np.array([5.0, 5.0]), np.array([3.0, 1.0]), self.bin_edges)
        np.testing.assert_allclose(scores / scores.sum(), [0.75, 0.25], rtol=1e-12)

    def test_out_of_range_and_non_finite_get_zero(self):
        costs = np.array([-5.0, 30.0, 60.0, 100.0, np.inf, np.nan])
        scores = _bin_adjusted_scores(costs, np.ones(6), self.bin_edges)
        self.assertGreater(scores[1], 0.0)
        np.testing.assert_array_equal(scores[[0, 2, 3, 4, 5]], 0.0)

    def test_no_in_range_destination_yields_zeros(self):
        scores = _bin_adjusted_scores(np.array([100.0, 200.0]), np.ones(2), self.bin_edges)
        np.testing.assert_array_equal(scores, [0.0, 0.0])


class NestedNodeSampleBinTestCase(unittest.TestCase):
    """Bin adjustment runs on each origin's combined row across tiers."""

    def setUp(self):
        (self.pairs, self.weights, _, self.c2z, _) = _toy_tiered_inputs()
        # c1's row: c1 (cost 0, bin 0), c2 (cost 600, bin 1), Z2 (cost 500, bin 1).
        self.costs = TieredODNodePairs(
            cells_to_cells={k: np.array([0.0, 600.0]) for k in self.pairs.cells_to_cells},
            zones_to_zones={k: np.array([500.0]) for k in self.pairs.zones_to_zones},
        )
        self.bin_edges = np.array([0.0, 100.0, 1_000.0])

    def _sample_c1(self, n_dest=20_000, bin_edges=None):
        return nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=np.array([1.0, 0.0, 0.0, 0.0]),
            bin_edges=self.bin_edges if bin_edges is None else bin_edges,
            n_orig=1,
            n_dest=n_dest,
            random_state=np.random.RandomState(0),
        )["c1"]

    def test_bin_mass_spans_tiers(self):
        """Each bin gets half the mass. Bin 1 holds a cell-tier dest (c2, w=1)
        and a zone-tier dest (Z2, w=10), which split it 1:10. Normalising per
        tier would instead give the zone tier half the mass on its own."""
        dests = self._sample_c1()
        self.assertAlmostEqual((dests == "c1").mean(), 0.5, delta=0.01)
        self.assertAlmostEqual((dests == "c2").mean(), 0.5 / 11, delta=0.01)
        self.assertAlmostEqual((dests == "Z2").mean(), 0.5 * 10 / 11, delta=0.01)

    def test_out_of_range_dests_never_sampled(self):
        dests = self._sample_c1(n_dest=2_000, bin_edges=np.array([0.0, 100.0, 550.0]))
        self.assertNotIn("c2", set(dests.tolist()))  # cost 600 > last edge

    def test_origin_without_in_range_dests_is_skipped(self):
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            cell_to_zone_node=self.c2z,
            orig_weights=np.array([1.0, 0.0, 0.0, 0.0]),
            bin_edges=np.array([2_000.0, 3_000.0]),
            n_orig=1,
            n_dest=10,
            random_state=np.random.RandomState(0),
        )
        self.assertEqual(out, {})

    def test_invalid_bin_edges_raise(self):
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            self._sample_c1(n_dest=1, bin_edges=np.array([0.0, 10.0, 5.0]))
        with self.assertRaisesRegex(ValueError, "length >= 2"):
            self._sample_c1(n_dest=1, bin_edges=np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
