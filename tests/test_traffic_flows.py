"""Tests for `aperta.traffic_flows`.

Run with:
    python -m unittest tests.test_traffic_flows

Two public entry points:
    - `estimate_edge_flows` — naive edge-betweenness-based traffic flow with optional cutoff
      OR a `nested_node_sample` dict for nested betweenness.
    - `nested_node_sample` — sample destinations for sampled origins, weighted
      by cost-derived scores, integrating all three OD tiers (cells_to_cells,
      cells_to_zones, zones_to_zones).
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODNodePairs
from aperta.traffic_flows import (
    bin_adjusted_dest_weights,
    estimate_edge_flows,
    nested_node_sample,
    percentile_bin_edges,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_graph() -> nx.MultiDiGraph:
    """Tiny multigraph: A → B → C → D plus a shortcut A → C, all edges
    `length = cost = 1`."""
    g = nx.MultiDiGraph()
    for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")]:
        g.add_edge(u, v, length=1.0, cost=1.0)
    return g


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


def _inverse_distance(x: np.ndarray) -> np.ndarray:
    """Vectorised inverse-distance score; `+ 1` avoids div-by-zero at self-pairs."""
    return 1.0 / (x + 1.0)


# ---------------------------------------------------------------------------
# `estimate_edge_flows` — naive betweenness-based traffic flow estimation
# ---------------------------------------------------------------------------


class GetTrafficFlowsTestCase(unittest.TestCase):
    def test_returns_series_keyed_by_multidigraph_edge(self):
        g = _toy_graph()
        sample = {"A": ["D"]}
        flows = estimate_edge_flows(
            g, weight="cost", expected_km_driven=10.0, nested_node_sample=sample
        )
        self.assertIsInstance(flows, pd.Series)
        # Each edge is keyed by (u, v, k) (MultiDiGraph).
        for key in flows.index:
            self.assertEqual(len(key), 3)

    def test_nested_node_sample_path(self):
        """Only edges on the sampled paths get positive flow; the
        `expected_km_driven` normaliser is exact."""
        g = _toy_graph()
        sample = {"A": ["D"]}
        flows = estimate_edge_flows(
            g, weight="cost", expected_km_driven=10.0, nested_node_sample=sample
        )
        lengths = nx.get_edge_attributes(g, "length")
        total_vkt = sum(v * lengths[k] for k, v in flows.items())
        self.assertAlmostEqual(total_vkt, 10.0)
        # The shortcut A-C and the segment C-D carry the trip; A-B and B-C are unused.
        self.assertGreater(flows.get(("A", "C", 0), 0), 0)
        self.assertGreater(flows.get(("C", "D", 0), 0), 0)
        self.assertEqual(flows.get(("A", "B", 0), 0), 0)
        self.assertEqual(flows.get(("B", "C", 0), 0), 0)


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
            self.c2z,
            self.orig_weights,
            _inverse_distance,
            n_orig=4,
            n_dest=10,
            random_state=rs,
        )
        self.assertIsInstance(out, dict)
        for k in out.keys():
            self.assertIn(k, ("c1", "c2", "c3", "c4"))

    def test_n_dest_per_origin(self):
        rs = np.random.RandomState(42)
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            self.c2z,
            self.orig_weights,
            _inverse_distance,
            n_orig=4,
            n_dest=15,
            random_state=rs,
        )
        for dests in out.values():
            self.assertEqual(len(dests), 15)

    def test_origin_weight_concentration(self):
        """All origin-weight on c2 → c2 is the only sampled origin (and
        appears regardless of n_orig because dupes are de-duped)."""
        rs = np.random.RandomState(42)
        weights_c2_only = np.array([0.0, 1.0, 0.0, 0.0])
        out = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            self.c2z,
            weights_c2_only,
            _inverse_distance,
            n_orig=4,
            n_dest=10,
            random_state=rs,
        )
        self.assertEqual(list(out.keys()), ["c2"])

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
            self.pairs,
            self.weights,
            self.costs,
            self.c2z,
            c1_only,
            _inverse_distance,
            n_orig=1,
            n_dest=1000,
            random_state=rs,
        )
        c1_dests = out["c1"]
        self.assertGreater((c1_dests == "c1").sum() / len(c1_dests), 0.9)

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
            self.c2z,
            c1_only,
            _inverse_distance,
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
            self.c2z,
            self.orig_weights,
            _inverse_distance,
            n_orig=4,
            n_dest=10,
            random_state=rs1,
        )
        out2 = nested_node_sample(
            self.pairs,
            self.weights,
            self.costs,
            self.c2z,
            self.orig_weights,
            _inverse_distance,
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
            self.c2z,
            c1_only,
            _inverse_distance,
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
            self.c2z,
            c1_and_c3,
            _inverse_distance,
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
            self.c2z,
            c1_only,
            _inverse_distance,
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
            self.c2z,
            c1_only,
            _inverse_distance,
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
            self.c2z,
            c3_only,
            _inverse_distance,
            n_orig=1,
            n_dest=100,
            random_state=rs,
        )
        # All sampled dests should be from c3's cell-tier or zone-tier pool.
        dests = set(out["c3"].tolist())
        allowed = {"c3", "c4", "Z1"}  # cell-tier dests + zone-tier dest from Z2
        self.assertTrue(dests.issubset(allowed), f"Unexpected dests: {dests - allowed}")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Bin-adjusted destination weights (sampling-bias fix helpers)
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


class BinAdjustedDestWeightsTestCase(unittest.TestCase):
    """Per-origin per-bin reweighting fixes the cost-distribution bias."""

    def setUp(self):
        # Two origins, each with 6 destinations at known costs.
        # Origin "rich":   destinations spread across all 4 bins (cost 5, 15, 25, 35, 45, 55).
        # Origin "sparse": destinations only in bins 0 and 3 (cost 5, 5, 55, 55, 55, 55).
        self.bin_edges = np.array([0.0, 10.0, 20.0, 40.0, 60.0])  # 4 bins
        self.pairs = TieredODNodePairs(
            cells_to_cells={
                "rich": np.array(["d1", "d2", "d3", "d4", "d5", "d6"]),
                "sparse": np.array(["d1", "d2", "d3", "d4", "d5", "d6"]),
            },
        )
        self.costs = TieredODNodePairs(
            cells_to_cells={
                "rich": np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
                "sparse": np.array([5.0, 5.0, 55.0, 55.0, 55.0, 55.0]),
            },
        )
        # Uniform W(D) = 1 for all destinations.
        self.weights = TieredODNodePairs(
            cells_to_cells={
                "rich": np.ones(6),
                "sparse": np.ones(6),
            },
        )
        self.n_bins = 4

    def _bin_of(self, costs):
        """Map costs back to bin indices the same way the helper does."""
        return np.digitize(costs, self.bin_edges) - 1

    def test_returns_same_subclass(self):
        adj = bin_adjusted_dest_weights(self.pairs, self.costs, self.weights, self.bin_edges)
        self.assertIsInstance(adj, TieredODNodePairs)

    def test_per_bin_total_matches_target_for_rich_origin(self):
        """Origin with destinations in every bin: each bin's adjusted weight
        should equal target = 1/n_bins (before renormalize) or stay
        proportional after."""
        adj = bin_adjusted_dest_weights(
            self.pairs,
            self.costs,
            self.weights,
            self.bin_edges,
            renormalize_per_origin=False,
        )
        w_rich = adj.cells_to_cells["rich"]
        bin_idx = self._bin_of(self.costs.cells_to_cells["rich"])
        per_bin = np.bincount(bin_idx, weights=w_rich, minlength=self.n_bins)
        # Each of the 4 bins is populated: each should sum to 1/4.
        np.testing.assert_allclose(per_bin, 0.25, rtol=1e-12)

    def test_per_bin_relative_proportions_match_target_for_sparse_origin(self):
        """Origin with destinations only in 2 bins: with renormalize=True,
        those bins should each hold the SAME fraction of total weight (since
        target is uniform 1/n_bins), regardless of how many destinations
        each bin holds."""
        adj = bin_adjusted_dest_weights(
            self.pairs,
            self.costs,
            self.weights,
            self.bin_edges,
            renormalize_per_origin=True,
        )
        w_sparse = adj.cells_to_cells["sparse"]
        bin_idx = self._bin_of(self.costs.cells_to_cells["sparse"])
        per_bin = np.bincount(bin_idx, weights=w_sparse, minlength=self.n_bins)
        # Bin 0 (2 dests at cost=5) and bin 3 (4 dests at cost=55) are populated.
        # After renormalisation, each of those bins should hold 50% of weight.
        np.testing.assert_allclose(per_bin[0], 0.5, rtol=1e-12)
        np.testing.assert_allclose(per_bin[3], 0.5, rtol=1e-12)
        np.testing.assert_allclose(per_bin[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(per_bin[2], 0.0, atol=1e-12)

    def test_within_bin_proportions_follow_input_weights(self):
        """When two destinations share a bin, the within-bin allocation
        should be proportional to their input W(D)."""
        pairs = TieredODNodePairs(
            cells_to_cells={"o": np.array(["a", "b"])},
        )
        costs = TieredODNodePairs(
            cells_to_cells={"o": np.array([5.0, 5.0])},  # both in bin 0
        )
        weights = TieredODNodePairs(
            cells_to_cells={"o": np.array([3.0, 1.0])},  # a:b = 3:1
        )
        adj = bin_adjusted_dest_weights(
            pairs, costs, weights, self.bin_edges, renormalize_per_origin=True
        )
        w = adj.cells_to_cells["o"]
        # Total = 1, ratio 3:1 -> 0.75, 0.25.
        np.testing.assert_allclose(w, [0.75, 0.25], rtol=1e-12)

    def test_renormalize_makes_sparse_and_rich_have_same_total(self):
        adj = bin_adjusted_dest_weights(
            self.pairs,
            self.costs,
            self.weights,
            self.bin_edges,
            renormalize_per_origin=True,
        )
        np.testing.assert_allclose(adj.cells_to_cells["rich"].sum(), 1.0, rtol=1e-12)
        np.testing.assert_allclose(adj.cells_to_cells["sparse"].sum(), 1.0, rtol=1e-12)

    def test_no_renormalize_makes_sparse_total_less_than_rich(self):
        adj = bin_adjusted_dest_weights(
            self.pairs,
            self.costs,
            self.weights,
            self.bin_edges,
            renormalize_per_origin=False,
        )
        # rich: 4 populated bins × 0.25 = 1.0;
        # sparse: 2 populated bins × 0.25 = 0.5.
        np.testing.assert_allclose(adj.cells_to_cells["rich"].sum(), 1.0, rtol=1e-12)
        np.testing.assert_allclose(adj.cells_to_cells["sparse"].sum(), 0.5, rtol=1e-12)

    def test_out_of_range_destinations_get_zero_weight(self):
        """Destinations whose cost is outside [bin_edges[0], bin_edges[-1]]
        should be excluded."""
        pairs = TieredODNodePairs(
            cells_to_cells={"o": np.array(["below", "in", "above"])},
        )
        costs = TieredODNodePairs(
            cells_to_cells={"o": np.array([-5.0, 30.0, 100.0])},  # below 0 / in / above 60
        )
        weights = TieredODNodePairs(
            cells_to_cells={"o": np.array([1.0, 1.0, 1.0])},
        )
        adj = bin_adjusted_dest_weights(
            pairs, costs, weights, self.bin_edges, renormalize_per_origin=True
        )
        w = adj.cells_to_cells["o"]
        # Only "in" should be non-zero.
        self.assertEqual(w[0], 0.0)
        self.assertGreater(w[1], 0.0)
        self.assertEqual(w[2], 0.0)
        # And the in-range one carries all the renormalised weight.
        np.testing.assert_allclose(w[1], 1.0, rtol=1e-12)

    def test_origin_with_no_in_range_destinations_yields_zeros(self):
        """If every destination is out of range, adjusted weights are all 0;
        renormalisation must not divide by zero."""
        pairs = TieredODNodePairs(
            cells_to_cells={"o": np.array(["x", "y"])},
        )
        costs = TieredODNodePairs(
            cells_to_cells={"o": np.array([100.0, 200.0])},  # both above max edge
        )
        weights = TieredODNodePairs(
            cells_to_cells={"o": np.array([1.0, 1.0])},
        )
        adj = bin_adjusted_dest_weights(
            pairs, costs, weights, self.bin_edges, renormalize_per_origin=True
        )
        np.testing.assert_array_equal(adj.cells_to_cells["o"], [0.0, 0.0])

    def test_all_tiers_processed_when_present(self):
        """When zones_to_zones is populated, it should be adjusted too."""
        pairs = TieredODNodePairs(
            cells_to_cells={"c1": np.array(["c1"])},
            zones_to_zones={"z1": np.array(["z2"])},
        )
        costs = TieredODNodePairs(
            cells_to_cells={"c1": np.array([5.0])},
            zones_to_zones={"z1": np.array([55.0])},
        )
        weights = TieredODNodePairs(
            cells_to_cells={"c1": np.array([1.0])},
            zones_to_zones={"z1": np.array([1.0])},
        )
        adj = bin_adjusted_dest_weights(
            pairs, costs, weights, self.bin_edges, renormalize_per_origin=True
        )
        self.assertIsNotNone(adj.cells_to_cells)
        self.assertIsNotNone(adj.zones_to_zones)
        np.testing.assert_allclose(adj.cells_to_cells["c1"], [1.0])
        np.testing.assert_allclose(adj.zones_to_zones["z1"], [1.0])

    def test_absent_tier_stays_none(self):
        """When cells_to_zones is None in the input, it stays None in output."""
        pairs = TieredODNodePairs(
            cells_to_cells={"c1": np.array(["c1"])},
        )
        costs = TieredODNodePairs(
            cells_to_cells={"c1": np.array([5.0])},
        )
        weights = TieredODNodePairs(
            cells_to_cells={"c1": np.array([1.0])},
        )
        adj = bin_adjusted_dest_weights(pairs, costs, weights, self.bin_edges)
        self.assertIsNone(adj.cells_to_zones)
        self.assertIsNone(adj.zones_to_zones)

    def test_invalid_bin_edges_raises(self):
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            bin_adjusted_dest_weights(
                self.pairs,
                self.costs,
                self.weights,
                np.array([0.0, 10.0, 5.0, 20.0]),  # not monotone
            )
        with self.assertRaisesRegex(ValueError, "length >= 2"):
            bin_adjusted_dest_weights(self.pairs, self.costs, self.weights, np.array([0.0]))
