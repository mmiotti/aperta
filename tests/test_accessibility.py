"""Tests for `aperta.accessibility` — accessibility metrics over tiered ODMs.

Run with:
    cd src && python -m unittest tests.test_accessibility

Each accessibility flavor gets its own `TestCase`. Shared fixtures use small,
hand-computable inputs so expected values can be inlined and checked exactly.
"""
import unittest
import warnings

import numpy as np
import pandas as pd

from aperta.accessibility import (
    Bin, Decay, count_in_bins, exp_decay, gravity, nearest_k, power_decay,
)
from aperta.od_pairs import TieredODGeoPairs, TieredODNodePairs, TieredODPairs
from aperta.overhead import add_origin_cell_overhead


def _toy_inputs() -> tuple[TieredODNodePairs, TieredODNodePairs, TieredODNodePairs, dict]:
    """Two origins ('a', 'b') in zone 'Z'; cell + zone tiers populated, no region tier.

    Layout (costs in metres):
        a:  cells [100, 200, 800]   zones [1500, 5000]
        b:  cells [150, 600]        (same zone Z, so same zone-tier dests)

    Per-tier weights (one property each in this helper):
        pop:   a cells [10, 20, 30]    b cells [5, 7]     zone [100, 200]
        emp:   a cells [1, 2, 3]       b cells [4, 5]     zone [50, 60]
    """
    costs = TieredODNodePairs(
        cells_to_cells={'a': np.array([100., 200., 800.]),
                        'b': np.array([150., 600.])},
        zones_to_zones={'Z': np.array([1500., 5000.])},
    )
    w_pop = TieredODNodePairs(
        cells_to_cells={'a': np.array([10., 20., 30.]),
                        'b': np.array([5., 7.])},
        zones_to_zones={'Z': np.array([100., 200.])},
    )
    w_emp = TieredODNodePairs(
        cells_to_cells={'a': np.array([1., 2., 3.]),
                        'b': np.array([4., 5.])},
        zones_to_zones={'Z': np.array([50., 60.])},
    )
    c2z = {'a': 'Z', 'b': 'Z'}
    return costs, w_pop, w_emp, c2z


class CountInBinsTestCase(unittest.TestCase):
    """`count_in_bins` sums per-property weights over destinations whose cost falls in each bin."""

    def setUp(self):
        self.costs, self.w_pop, self.w_emp, self.c2z = _toy_inputs()
        self.bins = [Bin('short', 0, 300), Bin('medium', 300, 1000), Bin('long', 1000, 6000)]

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_output_shape_and_index(self):
        df = count_in_bins(self.costs, {'pop': self.w_pop}, self.c2z, self.bins)
        self.assertEqual(list(df.index), ['a', 'b'])
        self.assertEqual(df.index.name, 'node')
        self.assertEqual(df.columns.names, ['bin', 'property'])
        self.assertEqual(list(df.columns),
                         [('short', 'pop'), ('medium', 'pop'), ('long', 'pop')])

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_single_property_known_values(self):
        df = count_in_bins(self.costs, {'pop': self.w_pop}, self.c2z, self.bins)
        # a: short = 10+20 = 30; medium = 30; long = 100+200 = 300
        self.assertEqual(df.loc['a', ('short', 'pop')], 30.0)
        self.assertEqual(df.loc['a', ('medium', 'pop')], 30.0)
        self.assertEqual(df.loc['a', ('long', 'pop')], 300.0)
        # b: short = 5; medium = 7; long = 100+200 = 300
        self.assertEqual(df.loc['b', ('short', 'pop')], 5.0)
        self.assertEqual(df.loc['b', ('medium', 'pop')], 7.0)
        self.assertEqual(df.loc['b', ('long', 'pop')], 300.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_multi_property_amortized_same_result(self):
        """Adding a second property should not change the first's values."""
        single = count_in_bins(self.costs, {'pop': self.w_pop}, self.c2z, self.bins)
        multi = count_in_bins(self.costs, {'pop': self.w_pop, 'emp': self.w_emp},
                              self.c2z, self.bins)
        for col in single.columns:
            pd.testing.assert_series_equal(single[col], multi[col], check_names=False)
        # And the new property's values are correct: a-medium-emp = 3; a-long-emp = 50+60 = 110.
        self.assertEqual(multi.loc['a', ('medium', 'emp')], 3.0)
        self.assertEqual(multi.loc['a', ('long', 'emp')], 110.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_half_open_boundary(self):
        """Bin is `[lo, hi)`: a destination at exactly `lo` is in; at `hi` is out."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([300., 600.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1.])})
        bins = [Bin('lo_inclusive', 300, 600), Bin('hi_inclusive', 600, 900)]
        df = count_in_bins(costs, {'w': w}, {'a': None}, bins)
        # 300 falls in the first bin (lo-inclusive); 600 falls in the second.
        self.assertEqual(df.loc['a', ('lo_inclusive', 'w')], 1.0)
        self.assertEqual(df.loc['a', ('hi_inclusive', 'w')], 1.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_non_finite_costs_dropped(self):
        """`np.inf` / `np.nan` costs never match any finite bin."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., np.inf, np.nan, 200.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 9., 9., 1.])})
        bins = [Bin('any', 0, 1e9)]
        df = count_in_bins(costs, {'w': w}, {'a': None}, bins)
        # Only the two finite-cost rows contribute (1 + 1 = 2). The 9-valued
        # rows at inf/NaN cost are dropped, not counted toward 'any'.
        self.assertEqual(df.loc['a', ('any', 'w')], 2.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_cell_only_no_zone_tier(self):
        """Works when `zones_to_zones` is `None` (cells-only tiered pairs)."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 500.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 2.])})
        bins = [Bin('short', 0, 300), Bin('long', 300, 1000)]
        df = count_in_bins(costs, {'w': w}, {'a': None}, bins)
        self.assertEqual(df.loc['a', ('short', 'w')], 1.0)
        self.assertEqual(df.loc['a', ('long', 'w')], 2.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_origin_with_no_matches_zero(self):
        """An origin whose dests miss every bin gets all-zero output, not NaN."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([5000., 6000.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1.])})
        bins = [Bin('short', 0, 100)]
        df = count_in_bins(costs, {'w': w}, {'a': None}, bins)
        self.assertEqual(df.loc['a', ('short', 'w')], 0.0)
        self.assertFalse(df.isna().any().any())

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_three_tiers_all_contribute(self):
        """Cell + zone + region tiers all stitched and counted correctly."""
        costs = TieredODNodePairs(
            cells_to_cells={'a': np.array([100., 200.])},
            zones_to_zones={'Z': np.array([1500.])},
            zones_to_regions={'Z': np.array([10_000.])},
        )
        w = TieredODNodePairs(
            cells_to_cells={'a': np.array([1., 2.])},
            zones_to_zones={'Z': np.array([10.])},
            zones_to_regions={'Z': np.array([100.])},
        )
        bins = [Bin('short', 0, 300), Bin('medium', 300, 5000), Bin('long', 5000, 100_000)]
        df = count_in_bins(costs, {'w': w}, {'a': 'Z'}, bins)
        self.assertEqual(df.loc['a', ('short', 'w')], 3.0)    # 1 + 2 cell-tier
        self.assertEqual(df.loc['a', ('medium', 'w')], 10.0)  # zone-tier
        self.assertEqual(df.loc['a', ('long', 'w')], 100.0)   # region-tier


class CountInBinsGeoKeyedTestCase(unittest.TestCase):
    """`count_in_bins` with `TieredODGeoPairs` input: cell-indexed output;
    per-cell origin overhead via `add_origin_cell_overhead` upstream."""

    def setUp(self):
        # Geo-keyed fixture: 3 cells (c1, c2 in zone Z; c3 in zone Z too).
        self.costs = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array([100., 200., 800.]),
                'c2': np.array([100., 200., 800.]),  # c2 shares c1's outgoing pattern
                'c3': np.array([150., 600.]),
            },
            zones_to_zones={'Z': np.array([1500., 5000.])},
        )
        self.pairs = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array(['d1', 'd2', 'd3']),
                'c2': np.array(['d1', 'd2', 'd3']),
                'c3': np.array(['d1', 'd2']),
            },
            zones_to_zones={'Z': np.array(['Z2', 'Z3'])},
        )
        self.w_pop = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array([10., 20., 30.]),
                'c2': np.array([10., 20., 30.]),
                'c3': np.array([5., 7.]),
            },
            zones_to_zones={'Z': np.array([100., 200.])},
        )
        self.cells_df = pd.DataFrame(
            {'zone_id': ['Z', 'Z', 'Z'],
             'walk_overhead_s': [0.0, 100.0, 50.0]},
            index=pd.Index(['c1', 'c2', 'c3'], name='cell_id'),
        )
        self.c2z = self.cells_df['zone_id'].to_dict()
        self.bins = [Bin('short', 0, 300), Bin('medium', 300, 1000),
                     Bin('long', 1000, 6000)]

    def test_output_indexed_by_cell(self):
        """Geo-keyed input → output indexed by cell_id (not by node)."""
        df = count_in_bins(self.costs, {'pop': self.w_pop}, self.c2z, self.bins)
        self.assertEqual(df.index.name, 'cell')
        self.assertSetEqual(set(df.index), {'c1', 'c2', 'c3'})

    def test_no_overhead_baked_matches_zero_overhead(self):
        """With zero overhead, c1 result matches the raw cost bins."""
        df = count_in_bins(self.costs, {'pop': self.w_pop}, self.c2z, self.bins)
        # c1: cell dests at 100, 200, 800 (in bins short, short, medium);
        # zone-tier 1500, 5000 (both long).
        self.assertEqual(df.loc['c1', ('short', 'pop')], 30.0)   # 10 + 20
        self.assertEqual(df.loc['c1', ('medium', 'pop')], 30.0)
        self.assertEqual(df.loc['c1', ('long', 'pop')], 300.0)   # 100 + 200

    def test_baked_overhead_shifts_bins(self):
        """`add_origin_cell_overhead` shifts every cost by the per-cell value."""
        baked = add_origin_cell_overhead(
            self.costs, self.pairs, self.cells_df, 'walk_overhead_s')
        df = count_in_bins(baked, {'pop': self.w_pop}, self.c2z, self.bins)
        # c1 has overhead 0 → unchanged from raw.
        self.assertEqual(df.loc['c1', ('short', 'pop')], 30.0)
        self.assertEqual(df.loc['c1', ('medium', 'pop')], 30.0)
        # c2 has overhead 100 → effective cell costs 200, 300, 900;
        # zone-tier: zone average overhead is (0+100+50)/3 = 50; zone costs
        # become 1550, 5050. Bins:
        # short [0,300): {200} → 10
        # medium [300,1000): {300, 900} → 20 + 30 = 50
        # long [1000,6000): {1550, 5050} → 100 + 200 = 300
        self.assertEqual(df.loc['c2', ('short', 'pop')], 10.0)
        self.assertEqual(df.loc['c2', ('medium', 'pop')], 50.0)
        self.assertEqual(df.loc['c2', ('long', 'pop')], 300.0)


class GravityTestCase(unittest.TestCase):
    """`gravity` sums each property's weights, weighted by f(cost), across all destinations."""

    def setUp(self):
        self.costs, self.w_pop, self.w_emp, self.c2z = _toy_inputs()

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_output_shape_and_index(self):
        df = gravity(self.costs, {'pop': self.w_pop}, self.c2z, exp_decay('d', 0.001))
        self.assertEqual(list(df.index), ['a', 'b'])
        self.assertEqual(df.index.name, 'node')
        self.assertEqual(df.columns.names, ['decay', 'property'])
        self.assertEqual(list(df.columns), [('d', 'pop')])

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_single_decay_known_values(self):
        """Hand-computed exponential decay sum across all tiers for origin 'a'."""
        beta = 0.001
        df = gravity(self.costs, {'pop': self.w_pop}, self.c2z, exp_decay('d', beta))
        # a's costs: cells [100, 200, 800], zones [1500, 5000]; weights match.
        expected_a = (
            10 * np.exp(-beta * 100)
            + 20 * np.exp(-beta * 200)
            + 30 * np.exp(-beta * 800)
            + 100 * np.exp(-beta * 1500)
            + 200 * np.exp(-beta * 5000)
        )
        self.assertAlmostEqual(df.loc['a', ('d', 'pop')], expected_a)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_multiple_decays_share_stitching(self):
        """Calling with N decays gives N independent columns; values match per-decay calls."""
        decays = [exp_decay('slow', 0.0005), exp_decay('fast', 0.005)]
        multi = gravity(self.costs, {'pop': self.w_pop}, self.c2z, decays)
        for d in decays:
            single = gravity(self.costs, {'pop': self.w_pop}, self.c2z, d)
            pd.testing.assert_series_equal(
                multi[(d.name, 'pop')], single[(d.name, 'pop')], check_names=False)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_multi_property_amortized_same_result(self):
        """Adding a second property does not change the first's column."""
        d = exp_decay('d', 0.001)
        single = gravity(self.costs, {'pop': self.w_pop}, self.c2z, d)
        multi = gravity(self.costs, {'pop': self.w_pop, 'emp': self.w_emp}, self.c2z, d)
        pd.testing.assert_series_equal(
            single[('d', 'pop')], multi[('d', 'pop')], check_names=False)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_power_decay_constructor(self):
        """`power_decay` produces `c**(-beta)`. Check against hand computation."""
        beta = 1.0
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([1., 2., 4.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1., 1.])})
        df = gravity(costs, {'w': w}, {'a': None}, power_decay('inv', beta))
        # 1/1 + 1/2 + 1/4 = 1.75
        self.assertAlmostEqual(df.loc['a', ('inv', 'w')], 1.75)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_custom_decay_callable(self):
        """Decay can be any vectorised callable."""
        # Linear decay: f(c) = max(0, 1 - c/threshold).
        threshold = 1000.0
        decay = Decay('linear', lambda c: np.maximum(0.0, 1.0 - c / threshold))
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([0., 500., 1000., 2000.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1., 1., 1.])})
        df = gravity(costs, {'w': w}, {'a': None}, decay)
        # f(0)=1, f(500)=0.5, f(1000)=0, f(2000)=0  → sum = 1.5
        self.assertAlmostEqual(df.loc['a', ('linear', 'w')], 1.5)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_non_finite_costs_dropped(self):
        """`np.inf` / `np.nan` costs contribute zero to the gravity sum."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([0., np.inf, np.nan, 100.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 99., 99., 1.])})
        df = gravity(costs, {'w': w}, {'a': None}, exp_decay('d', 0.0))
        # exp(0) = 1 for the two finite-cost entries; inf/nan dropped.
        self.assertAlmostEqual(df.loc['a', ('d', 'w')], 2.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_power_decay_with_zero_cost_does_not_corrupt(self):
        """power-law decay produces inf at c=0; defensive drop keeps the sum finite."""
        # Without intrazonal-cost handling, a self-pair at cost 0 would give inf.
        # The function's defensive `isfinite` drop should treat that entry as zero
        # so other destinations' contributions remain visible. The c=0 entry
        # triggers an expected divide-by-zero RuntimeWarning we suppress.
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([0., 1., 2.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1., 1.])})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            df = gravity(costs, {'w': w}, {'a': None}, power_decay('inv', 1.0))
        # Defensive drop: 1/1 + 1/2 = 1.5; the c=0 entry is dropped.
        self.assertAlmostEqual(df.loc['a', ('inv', 'w')], 1.5)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_three_tiers_all_contribute(self):
        """Cell + zone + region tiers all stitched and gravity-weighted correctly."""
        costs = TieredODNodePairs(
            cells_to_cells={'a': np.array([100., 200.])},
            zones_to_zones={'Z': np.array([1500.])},
            zones_to_regions={'Z': np.array([10_000.])},
        )
        w = TieredODNodePairs(
            cells_to_cells={'a': np.array([1., 2.])},
            zones_to_zones={'Z': np.array([10.])},
            zones_to_regions={'Z': np.array([100.])},
        )
        beta = 0.001
        df = gravity(costs, {'w': w}, {'a': 'Z'}, exp_decay('d', beta))
        expected = (1 * np.exp(-beta * 100) + 2 * np.exp(-beta * 200)
                    + 10 * np.exp(-beta * 1500) + 100 * np.exp(-beta * 10_000))
        self.assertAlmostEqual(df.loc['a', ('d', 'w')], expected)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_empty_decays_raises(self):
        with self.assertRaisesRegex(ValueError, "decays.*non-empty"):
            gravity(self.costs, {'pop': self.w_pop}, self.c2z, [])

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_geo_keyed_overhead_shifts_decay(self):
        """Geo-keyed: per-cell overhead baked via `add_origin_cell_overhead`
        shifts every cost; for exp decay that's a uniform multiplicative
        factor exp(-beta · overhead)."""
        # Build a geo-keyed mirror of the node-keyed fixture.
        costs_geo = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array([100., 200., 800.]),
                'c2': np.array([100., 200., 800.]),
            },
            zones_to_zones={'Z': np.array([1500., 5000.])},
        )
        pairs_geo = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array(['d1', 'd2', 'd3']),
                'c2': np.array(['d1', 'd2', 'd3']),
            },
            zones_to_zones={'Z': np.array(['Z2', 'Z3'])},
        )
        w_pop_geo = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array([10., 20., 30.]),
                'c2': np.array([10., 20., 30.]),
            },
            zones_to_zones={'Z': np.array([100., 200.])},
        )
        cells_df = pd.DataFrame(
            {'zone_id': ['Z', 'Z'],
             'walk_overhead_s': [0.0, 100.0]},
            index=pd.Index(['c1', 'c2'], name='cell_id'),
        )
        c2z = cells_df['zone_id'].to_dict()
        beta = 0.001
        baked = add_origin_cell_overhead(costs_geo, pairs_geo, cells_df,
                                         'walk_overhead_s')
        df = gravity(baked, {'pop': w_pop_geo}, c2z, exp_decay('d', beta))
        # Zone-tier overhead = per-zone-mean of per-cell overheads. Z has cells
        # c1 (0) and c2 (100), mean = 50 → ALL cells in Z see +50 at zone tier.
        # Cell tier shifts per-cell (c1: 0, c2: 100).
        c1_total = (
            10 * np.exp(-beta * 100) + 20 * np.exp(-beta * 200) + 30 * np.exp(-beta * 800)
            + 100 * np.exp(-beta * 1550) + 200 * np.exp(-beta * 5050)  # zone-tier +50
        )
        c2_total = (
            10 * np.exp(-beta * 200) + 20 * np.exp(-beta * 300) + 30 * np.exp(-beta * 900)
            + 100 * np.exp(-beta * 1550) + 200 * np.exp(-beta * 5050)  # zone-tier +50
        )
        self.assertAlmostEqual(df.loc['c1', ('d', 'pop')], c1_total)
        self.assertAlmostEqual(df.loc['c2', ('d', 'pop')], c2_total)
        self.assertEqual(df.index.name, 'cell')


class NearestKTestCase(unittest.TestCase):
    """`nearest_k` returns the mean cost (or cost-at-k) over the k nearest weight-units."""

    def setUp(self):
        self.costs, self.w_pop, self.w_emp, self.c2z = _toy_inputs()

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_output_shape_and_index(self):
        df = nearest_k(self.costs, {'pop': self.w_pop}, self.c2z, ks=[1, 3])
        self.assertEqual(list(df.index), ['a', 'b'])
        self.assertEqual(df.index.name, 'node')
        self.assertEqual(df.columns.names, ['k', 'property'])
        self.assertEqual(list(df.columns), [(1, 'pop'), (3, 'pop')])

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_k_equals_one_is_cost_of_nearest_opportunity(self):
        """k=1: cost of the first (lowest-cost) weight-unit — the nearest opportunity."""
        df = nearest_k(self.costs, {'pop': self.w_pop}, self.c2z, ks=1)
        # 'a': nearest dest at cost 100 has weight 10 — the first opportunity is at cost 100.
        # 'b': nearest dest at cost 150 has weight 5 — first opportunity at cost 150.
        self.assertEqual(df.loc['a', (1, 'pop')], 100.0)
        self.assertEqual(df.loc['b', (1, 'pop')], 150.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_cost_mean_unit_weights(self):
        """With weight=1 at every dest, `cost_mean` is just the arithmetic mean of the k cheapest costs."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 200., 300., 400.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1., 1., 1.])})
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 2, 3, 4])
        self.assertEqual(df.loc['a', (1, 'w')], 100.0)
        self.assertEqual(df.loc['a', (2, 'w')], 150.0)  # (100 + 200) / 2
        self.assertEqual(df.loc['a', (3, 'w')], 200.0)  # (100 + 200 + 300) / 3
        self.assertEqual(df.loc['a', (4, 'w')], 250.0)  # (100 + 200 + 300 + 400) / 4

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_cost_mean_multi_unit_destination(self):
        """A destination with weight=3 contributes 3 weight-units at the same cost."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 200.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([3., 1.])})
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 2, 3, 4])
        # k <= 3: all units come from dest at cost 100 → mean = 100.
        self.assertEqual(df.loc['a', (1, 'w')], 100.0)
        self.assertEqual(df.loc['a', (2, 'w')], 100.0)
        self.assertEqual(df.loc['a', (3, 'w')], 100.0)
        # k = 4: 3 units at 100 + 1 unit at 200 → mean = (300 + 200) / 4 = 125.
        self.assertEqual(df.loc['a', (4, 'w')], 125.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_cost_mean_fractional_boundary(self):
        """When k boundary lands mid-destination, partial contribution from that destination."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([10., 100.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([2., 3.])})
        # k = 3: 2 units at 10 + 1 unit at 100 → mean = (20 + 100) / 3 ≈ 40.
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[3])
        self.assertAlmostEqual(df.loc['a', (3, 'w')], 40.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_cost_at_k_aggregator(self):
        """`cost_at_k` returns the cost where cumulative weight first reaches k."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 200., 300., 400.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1., 1., 1.])})
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 2, 3, 4], aggregator='cost_at_k')
        # cost_at_k = cost of the k-th opportunity (1-indexed, in cumulative-weight space).
        self.assertEqual(df.loc['a', (1, 'w')], 100.0)
        self.assertEqual(df.loc['a', (2, 'w')], 200.0)
        self.assertEqual(df.loc['a', (3, 'w')], 300.0)
        self.assertEqual(df.loc['a', (4, 'w')], 400.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_cost_at_k_with_multi_unit_destination(self):
        """cost_at_k locks to the destination that pushed cum_weight over k."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 200.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([3., 1.])})
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 2, 3, 4], aggregator='cost_at_k')
        # k=1,2,3: still inside dest 0 → cost = 100. k=4: crossed into dest 1 → 200.
        self.assertEqual(df.loc['a', (1, 'w')], 100.0)
        self.assertEqual(df.loc['a', (2, 'w')], 100.0)
        self.assertEqual(df.loc['a', (3, 'w')], 100.0)
        self.assertEqual(df.loc['a', (4, 'w')], 200.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_k_exceeds_available_returns_nan(self):
        """If total positive weight < k, return NaN — the k-th opportunity is unreachable."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 200.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 1.])})  # total = 2
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 2, 3, 100])
        self.assertEqual(df.loc['a', (1, 'w')], 100.0)
        self.assertEqual(df.loc['a', (2, 'w')], 150.0)
        self.assertTrue(np.isnan(df.loc['a', (3, 'w')]))
        self.assertTrue(np.isnan(df.loc['a', (100, 'w')]))

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_zero_weight_destinations_excluded(self):
        """Destinations with weight 0 don't count toward k, even if their cost is lowest."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 200., 300.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([0., 1., 1.])})
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 2])
        # Nearest unit is at cost 200 (weight 1); next at cost 300 (weight 1).
        self.assertEqual(df.loc['a', (1, 'w')], 200.0)
        self.assertEqual(df.loc['a', (2, 'w')], 250.0)  # (200 + 300) / 2

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_no_positive_weights_returns_nan(self):
        """An origin with no positive-weight destinations gets NaN everywhere."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., 200.])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([0., 0.])})
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 5])
        self.assertTrue(np.isnan(df.loc['a', (1, 'w')]))
        self.assertTrue(np.isnan(df.loc['a', (5, 'w')]))

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_non_finite_costs_excluded(self):
        """`inf` / `nan` costs are never the nearest, regardless of weight."""
        costs = TieredODNodePairs(cells_to_cells={'a': np.array([100., np.inf, 200., np.nan])})
        w = TieredODNodePairs(cells_to_cells={'a': np.array([1., 9., 1., 9.])})
        df = nearest_k(costs, {'w': w}, {'a': None}, ks=[1, 2])
        # Only two finite-cost destinations: 100 (w=1), 200 (w=1).
        self.assertEqual(df.loc['a', (1, 'w')], 100.0)
        self.assertEqual(df.loc['a', (2, 'w')], 150.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_multi_property(self):
        """Different properties give independent cost_means against the same sort."""
        df = nearest_k(self.costs, {'pop': self.w_pop, 'emp': self.w_emp}, self.c2z, ks=[2])
        # 'a' sorted by cost: [100, 200, 800, 1500, 5000].
        # pop weights in that order: [10, 20, 30, 100, 200]. dest 0 (w=10) covers k=2 alone → 100.
        # emp weights in that order: [1, 2, 3, 50, 60]. dest 0 (w=1) covers 1 unit; partial
        #   1 unit from dest 1 (cost 200) → mean = (100 + 200) / 2 = 150.
        self.assertEqual(df.loc['a', (2, 'pop')], 100.0)
        self.assertEqual(df.loc['a', (2, 'emp')], 150.0)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_three_tiers_sort_across(self):
        """Sort spans across cell / zone / region tiers with proper weight accumulation."""
        costs = TieredODNodePairs(
            cells_to_cells={'a': np.array([2000., 100.])},
            zones_to_zones={'Z': np.array([1000., 500.])},
            zones_to_regions={'Z': np.array([10_000.])},
        )
        w = TieredODNodePairs(
            cells_to_cells={'a': np.array([100., 1.])},
            zones_to_zones={'Z': np.array([20., 10.])},
            zones_to_regions={'Z': np.array([1000.])},
        )
        # Sort by cost: 100 (w=1), 500 (w=10), 1000 (w=20), 2000 (w=100), 10000 (w=1000).
        # cum_w: 1, 11, 31, 131, 1131.
        # k=1: dest 0 (w=1, cost 100). mean = 100.
        # k=2: 1 unit at 100, 1 unit at 500 → (100 + 500) / 2 = 300.
        # k=3: 1 at 100, 2 at 500 → (100 + 1000) / 3 ≈ 366.67.
        df = nearest_k(costs, {'w': w}, {'a': 'Z'}, ks=[1, 2, 3])
        self.assertEqual(df.loc['a', (1, 'w')], 100.0)
        self.assertAlmostEqual(df.loc['a', (2, 'w')], 300.0)
        self.assertAlmostEqual(df.loc['a', (3, 'w')], (100 + 500 * 2) / 3)

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_geo_keyed_overhead_shifts_cost(self):
        """Geo-keyed: with one cell per zone, per-cell overhead = per-zone-mean
        overhead, so cell + zone tier shift by the same amount → cost_mean
        shifts by exactly the overhead."""
        # Single-cell-per-zone fixture so cell shift == zone shift, isolating
        # the test from the mean-aggregation subtlety.
        costs_geo = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array([100., 200., 800.]),
                'c2': np.array([100., 200., 800.]),
            },
            zones_to_zones={
                'Z1': np.array([1500., 5000.]),
                'Z2': np.array([1500., 5000.]),
            },
        )
        pairs_geo = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array(['d1', 'd2', 'd3']),
                'c2': np.array(['d1', 'd2', 'd3']),
            },
            zones_to_zones={
                'Z1': np.array(['ZA', 'ZB']),
                'Z2': np.array(['ZA', 'ZB']),
            },
        )
        w_pop_geo = TieredODGeoPairs(
            cells_to_cells={
                'c1': np.array([10., 20., 30.]),
                'c2': np.array([10., 20., 30.]),
            },
            zones_to_zones={
                'Z1': np.array([100., 200.]),
                'Z2': np.array([100., 200.]),
            },
        )
        cells_df = pd.DataFrame(
            {'zone_id': ['Z1', 'Z2'],
             'walk_overhead_s': [0.0, 100.0]},
            index=pd.Index(['c1', 'c2'], name='cell_id'),
        )
        c2z = cells_df['zone_id'].to_dict()
        baked = add_origin_cell_overhead(costs_geo, pairs_geo, cells_df,
                                         'walk_overhead_s')
        df = nearest_k(baked, {'pop': w_pop_geo}, c2z, ks=[1, 2, 3])
        # c1: overhead 0 → unshifted. c2: +100 uniformly across cell + zone tiers
        # (because each zone has only one cell, so per-zone-mean = per-cell value).
        np.testing.assert_array_almost_equal(
            df.loc['c2'].values, df.loc['c1'].values + 100.0)
        self.assertEqual(df.index.name, 'cell')

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_unknown_aggregator_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown aggregator"):
            nearest_k(self.costs, {'pop': self.w_pop}, self.c2z, ks=[1], aggregator='nope')

    @unittest.skip("Phase A refactor: pending Phase B/D for cells_to_zones replacement")
    def test_invalid_k_raises(self):
        with self.assertRaisesRegex(ValueError, "> 0"):
            nearest_k(self.costs, {'pop': self.w_pop}, self.c2z, ks=[0, 1])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            nearest_k(self.costs, {'pop': self.w_pop}, self.c2z, ks=[])


if __name__ == '__main__':
    unittest.main()
