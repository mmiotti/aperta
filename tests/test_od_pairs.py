"""Tests for `aperta.od_pairs.get_pairs` and friends (the tiered OD API).

Run with:
    cd src && python -m unittest tests.test_od_pairs

The new contract:
    `get_pairs` returns a `TieredODPairs` dataclass with up to three OD dicts —
    `cells_to_cells`, `zones_to_zones`, `zones_to_regions` — each at its tier's
    resolution. Tier assignment is per-region-pair (with same-region as implicit
    zone tier and promotion when any cross-region cell-tier crossing exists).
    Cell tier is per-zone-pair (only thing that uses zone-centroid distance).

Synthetic-world fixture: a 3×3 cell grid grouped into 3 zones and 2 regions:

    cells (unit spacing, integer coords):
        C6 C7 C8        zones: Z0 = {C0..C3}    regions: R0 = {Z0, Z1}
        C3 C4 C5               Z1 = {C4, C5}             R1 = {Z2}
        C0 C1 C2               Z2 = {C6, C7, C8}

    Cell Ci is mapped to network node Ni at the same coords.
    Zone Zk → representative network node ZNk at the zone centroid.
    Region Rk → representative network node RNk at the region centroid.

Populations: each cell `Ci` has `population = (i + 1) * 100` (total = 4500).
Zones and regions are aggregated through the cell→zone→region mapping so the
conservation invariant is testable.
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
    return Polygon([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5),
                    (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)])


def _build_world() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cell_coords = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    node_specs = (
        [(f'N{i}', float(x), float(y)) for i, (x, y) in enumerate(cell_coords)]
        + [('ZN0', 1.0, 0.5), ('ZN1', 1.0, 1.0), ('ZN2', 1.0, 2.0)]
        + [('RN0', 1.0, 0.5), ('RN1', 1.0, 2.0)]
    )
    nodes = gpd.GeoDataFrame(
        [{'node_id': nid, 'geometry': Point(x, y)} for nid, x, y in node_specs],
        crs='EPSG:2056',
    ).set_index('node_id')

    cell_rows = []
    for i, (x, y) in enumerate(cell_coords):
        zone_id = 'Z0' if i <= 3 else ('Z1' if i <= 5 else 'Z2')
        cell_rows.append({
            'cell_id': f'C{i}', 'node_id': f'N{i}', 'zone_id': zone_id,
            'population': _CELL_POPULATIONS[i],
            'geometry': _square(x, y),
        })
    cells = gpd.GeoDataFrame(cell_rows, crs='EPSG:2056').set_index('cell_id')

    zones = gpd.GeoDataFrame([
        {'zone_id': 'Z0', 'node_id': 'ZN0', 'region_id': 'R0',
         'geometry': Polygon([(-0.5, -0.5), (2.5, -0.5), (2.5, 0.5), (-0.5, 0.5)])},
        {'zone_id': 'Z1', 'node_id': 'ZN1', 'region_id': 'R0',
         'geometry': Polygon([(-0.5, 0.5), (2.5, 0.5), (2.5, 1.5), (-0.5, 1.5)])},
        {'zone_id': 'Z2', 'node_id': 'ZN2', 'region_id': 'R1',
         'geometry': Polygon([(-0.5, 1.5), (2.5, 1.5), (2.5, 2.5), (-0.5, 2.5)])},
    ], crs='EPSG:2056').set_index('zone_id')
    zones['population'] = cells.groupby('zone_id')['population'].sum()

    regions = gpd.GeoDataFrame([
        {'region_id': 'R0', 'node_id': 'RN0',
         'geometry': Polygon([(-0.5, -0.5), (2.5, -0.5), (2.5, 1.5), (-0.5, 1.5)])},
        {'region_id': 'R1', 'node_id': 'RN1',
         'geometry': Polygon([(-0.5, 1.5), (2.5, 1.5), (2.5, 2.5), (-0.5, 2.5)])},
    ], crs='EPSG:2056').set_index('region_id')
    regions['population'] = zones.groupby('region_id')['population'].sum()
    return nodes, cells, zones, regions


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
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_returns_only_cells_tier(self):
        pairs = get_pairs(self.cells, r_cells=0.5, node_column='node_id')
        self.assertIsInstance(pairs, TieredODPairs)
        self.assertIsNone(pairs.zones_to_zones)
        self.assertIsNone(pairs.zones_to_regions)
        for cn in self.cells['node_id']:
            self.assertIn(cn, pairs.cells_to_cells[cn].tolist())

    def test_per_cell_distance(self):
        pairs = get_pairs(self.cells, r_cells=1.01, node_column='node_id')
        self.assertCountEqual(pairs.cells_to_cells['N0'].tolist(), ['N0', 'N1', 'N3'])
        self.assertCountEqual(pairs.cells_to_cells['N4'].tolist(),
                              ['N1', 'N3', 'N4', 'N5', 'N7'])

    def test_symmetric(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id')
        ok, bad = _is_symmetric(pairs.cells_to_cells)
        self.assertTrue(ok, f'asymmetric pair: {bad}')


class TieredAssignmentTestCase(unittest.TestCase):
    """Per-region-pair tier classification with intra-region/same-zone carve-outs."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_same_zone_always_cell_tier(self):
        # r_cells=0: only same-zone carve-out fires.
        pairs = get_pairs(self.cells, r_cells=0.0, node_column='node_id',
                          zones=self.zones, r_zones=0.0)
        for orig in ['N0', 'N1', 'N2', 'N3']:
            self.assertCountEqual(pairs.cells_to_cells[orig].tolist(),
                                  ['N0', 'N1', 'N2', 'N3'])

    def test_same_region_always_zone_tier(self):
        # Same-region (Z0, Z1 both in R0) → always zone-tier regardless of r_zones.
        pairs = get_pairs(self.cells, r_cells=0.0, node_column='node_id',
                          zones=self.zones, r_zones=0.0,
                          regions=self.regions, r_regions=0.0)
        assert pairs.zones_to_zones is not None
        self.assertIn('ZN1', pairs.zones_to_zones['ZN0'].tolist())
        self.assertIn('ZN0', pairs.zones_to_zones['ZN1'].tolist())
        # Z2 in different region; r_regions=0 → no link from Z0/Z1 to Z2 anywhere.
        self.assertNotIn('ZN2', pairs.zones_to_zones.get('ZN0', np.array([])).tolist())

    def test_cross_region_zone_tier_when_regions_close(self):
        # R0 centroid (1, 0.5), R1 centroid (1, 2.0). d(R0, R1) = 1.5.
        # r_zones=2.0 → R0, R1 close enough; all (Z⊂R0, Z'⊂R1) zone-tier.
        pairs = get_pairs(self.cells, r_cells=0.0, node_column='node_id',
                          zones=self.zones, r_zones=2.0,
                          regions=self.regions, r_regions=10.0)
        assert pairs.zones_to_zones is not None
        # ZN0 should now reach ZN2 (cross-region zone-tier).
        self.assertIn('ZN2', pairs.zones_to_zones['ZN0'].tolist())
        # No region-tier emission for (R0, R1) since they're at zone-tier.
        assert pairs.zones_to_regions is not None
        self.assertNotIn('RN1', pairs.zones_to_regions.get('ZN0', np.array([])).tolist())

    def test_region_tier_when_regions_far(self):
        # d(R0, R1) = 1.5. r_zones=1.0 (R0/R1 not close enough), r_regions=2.0 (covers).
        pairs = get_pairs(self.cells, r_cells=0.0, node_column='node_id',
                          zones=self.zones, r_zones=1.0,
                          regions=self.regions, r_regions=2.0)
        assert pairs.zones_to_regions is not None
        # All zones in R0 should reach R1 (per-zone).
        self.assertCountEqual(pairs.zones_to_regions['ZN0'].tolist(), ['RN1'])
        self.assertCountEqual(pairs.zones_to_regions['ZN1'].tolist(), ['RN1'])
        # And the (only) zone in R1 should reach R0.
        self.assertCountEqual(pairs.zones_to_regions['ZN2'].tolist(), ['RN0'])

    def test_region_tier_dropped_when_too_far(self):
        # r_regions=0 → nothing cross-region.
        pairs = get_pairs(self.cells, r_cells=0.0, node_column='node_id',
                          zones=self.zones, r_zones=0.0,
                          regions=self.regions, r_regions=0.0)
        assert pairs.zones_to_regions is not None
        self.assertEqual(len(pairs.zones_to_regions), 0)


class PromotionTestCase(unittest.TestCase):
    """If any cross-region (Z, Z') is cell-tier, the whole (R, R') is promoted to
    zone-tier (instead of region-tier) — preserves conservation exactly.
    """

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_cross_region_cell_tier_promotes_to_zone_tier(self):
        # Without promotion: with r_cells big enough to span R0-R1 (e.g. cells C5
        # in Z1⊂R0 at (2, 1) and C8 in Z2⊂R1 at (2, 2) have d=1.0), AND r_zones too
        # small for the regions, (R0, R1) would land in region tier with C5-C8 also
        # at cell tier → double-count.
        # With promotion: (R0, R1) is forced to zone tier; cells get clean coverage.
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=0.5,  # too small for d(R0,R1)=1.5
                          regions=self.regions, r_regions=10.0)
        assert pairs.zones_to_zones is not None
        assert pairs.zones_to_regions is not None
        # Promotion fired: ZN0 should reach ZN2 at zone tier (not RN1 at region tier).
        self.assertIn('ZN2', pairs.zones_to_zones['ZN0'].tolist())
        # No region-tier emission for the promoted pair.
        self.assertNotIn('RN1', pairs.zones_to_regions.get('ZN0', np.array([])).tolist())


class SymmetryTestCase(unittest.TestCase):
    """Tiered output is inherently symmetric across all three tiers."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_three_tiers_symmetric(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          regions=self.regions, r_regions=10.0)
        for name, d in [('cells', pairs.cells_to_cells),
                        ('zones', pairs.zones_to_zones),
                        ('zones_to_regions', pairs.zones_to_regions)]:
            if d is None:
                continue
            ok, bad = _is_symmetric(d)
            # Note: zones_to_regions can be asymmetric in general (different node
            # spaces) but here zone nodes and region nodes are disjoint sets, so
            # the symmetry check only catches issues within cells/zones tiers.
            if name in ('cells', 'zones'):
                self.assertTrue(ok, f'{name} asymmetric at {bad}')


class CustomZoneCentroidsTestCase(unittest.TestCase):
    """User-supplied `zones_centroids` override `zones.geometry.centroid`."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_custom_centroids_change_cell_tier(self):
        # Move Z1's centroid far out so (Z0, Z1) is no longer cell-tier.
        custom = gpd.GeoSeries(
            [Point(1.0, 0.0), Point(100.0, 100.0), Point(1.0, 2.0)],
            index=self.zones.index, crs=self.zones.crs,
        )
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          zones_centroids=custom)
        # N0 should now see only Z0 cells (no cross to Z1 at cell tier).
        self.assertSetEqual(set(pairs.cells_to_cells['N0'].tolist()),
                            {'N0', 'N1', 'N2', 'N3'})


class SharedNodeDedupTestCase(unittest.TestCase):
    """Multiple cells sharing a node: dedup, and populations sum in dest_values."""

    def test_shared_dest_node(self):
        cells = gpd.GeoDataFrame({
            'node_id': ['N0', 'N0', 'N1'],
            'zone_id': ['Z0', 'Z0', 'Z1'],
            'population': [10, 20, 30],
            'geometry': [Point(0, 0), Point(0.1, 0), Point(1, 0)],
        }, index=pd.Index(['C0', 'C1', 'C2'], name='cell_id'), crs='EPSG:2056')
        zones = gpd.GeoDataFrame({
            'node_id': ['ZN0', 'ZN1'],
            'region_id': ['R0', 'R0'],
            'population': [30, 30],
            'geometry': [Point(0.05, 0), Point(1, 0)],
        }, index=pd.Index(['Z0', 'Z1'], name='zone_id'), crs='EPSG:2056')
        pairs = get_pairs(cells, r_cells=2.0, node_column='node_id',
                          zones=zones, r_zones=2.0)
        self.assertCountEqual(pairs.cells_to_cells['N0'].tolist(), ['N0', 'N1'])
        vals = dest_values('population', pairs, cells, 'node_id', zones=zones)
        n0_dests = pairs.cells_to_cells['N0']
        n0_vals = dict(zip(n0_dests.tolist(), vals.cells_to_cells['N0'].tolist()))
        self.assertEqual(n0_vals['N0'], 30)  # 10 + 20
        self.assertEqual(n0_vals['N1'], 30)


class DestValuesTestCase(unittest.TestCase):
    """`dest_values` looks up the per-tier value matching each tier's destinations."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_returns_three_dicts_same_shape(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          regions=self.regions, r_regions=10.0)
        vals = dest_values('population', pairs, self.cells, 'node_id',
                           zones=self.zones, regions=self.regions)
        for ot_pairs, ot_vals in [(pairs.cells_to_cells, vals.cells_to_cells),
                                  (pairs.zones_to_zones, vals.zones_to_zones),
                                  (pairs.zones_to_regions, vals.zones_to_regions)]:
            if ot_pairs is None:
                self.assertIsNone(ot_vals)
                continue
            assert ot_vals is not None
            self.assertSetEqual(set(ot_pairs.keys()), set(ot_vals.keys()))
            for k in ot_pairs:
                self.assertEqual(len(ot_pairs[k]), len(ot_vals[k]))

    def test_raises_without_required_frame(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5)
        with self.assertRaisesRegex(ValueError, 'zones'):
            dest_values('population', pairs, self.cells, 'node_id')

    def test_raises_on_missing_column(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id')
        with self.assertRaisesRegex(ValueError, 'unknown_col'):
            dest_values('unknown_col', pairs, self.cells, 'node_id')

    def test_region_tier_looks_up_regions_column(self):
        # Set up a case where ZN0 reaches RN1 at region tier; vals should be pop(R1).
        pairs = get_pairs(self.cells, r_cells=0.0, node_column='node_id',
                          zones=self.zones, r_zones=1.0,
                          regions=self.regions, r_regions=2.0)
        vals = dest_values('population', pairs, self.cells, 'node_id',
                           zones=self.zones, regions=self.regions)
        assert vals.zones_to_regions is not None
        # R1 has only Z2 → pop(R1) = pop(Z2) = 700+800+900 = 2400.
        self.assertEqual(vals.zones_to_regions['ZN0'].tolist(), [2400])


class ConservationTestCase(unittest.TestCase):
    """Population invariant: for every origin cell, sum across all tiers = total."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def _check_conservation(self, pairs):
        vals = dest_values('population', pairs, self.cells, 'node_id',
                           zones=self.zones, regions=self.regions)
        total = sum(_CELL_POPULATIONS)
        cell_to_zone = self.cells.set_index('node_id')['zone_id']
        zone_to_node = self.zones['node_id']
        zone_to_region = self.zones['region_id']
        region_to_node = self.regions['node_id']
        for origin in self.cells['node_id']:
            tier_sum = vals.cells_to_cells.get(origin, np.array([])).sum()
            zone_origin = zone_to_node[cell_to_zone[origin]]
            region_origin = region_to_node[zone_to_region[cell_to_zone[origin]]]
            if vals.zones_to_zones is not None:
                tier_sum += vals.zones_to_zones.get(zone_origin, np.array([])).sum()
            if vals.zones_to_regions is not None:
                # zones_to_regions is keyed by zone node, not region node.
                tier_sum += vals.zones_to_regions.get(zone_origin, np.array([])).sum()
            self.assertEqual(tier_sum, total, f'origin {origin}: {tier_sum} != {total}')

    def test_conservation_three_tier_full_coverage(self):
        pairs = get_pairs(self.cells, r_cells=0.5, node_column='node_id',
                          zones=self.zones, r_zones=1.5,
                          regions=self.regions, r_regions=100.0)
        self._check_conservation(pairs)

    def test_conservation_under_promotion(self):
        # Cross-region cell-tier exists (r_cells=1.5 catches Z1-Z2 d=1.0); r_zones
        # too small for (R0, R1). Promotion ensures conservation.
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=0.5,
                          regions=self.regions, r_regions=10.0)
        self._check_conservation(pairs)

    def test_conservation_no_regions(self):
        # Two-tier (cells + zones), r_zones large enough to cover everything.
        pairs = get_pairs(self.cells, r_cells=0.5, node_column='node_id',
                          zones=self.zones, r_zones=10.0)
        # Conservation: cells_to_cells + zones_to_zones (no regions tier).
        vals = dest_values('population', pairs, self.cells, 'node_id', zones=self.zones)
        total = sum(_CELL_POPULATIONS)
        cell_to_zone = self.cells.set_index('node_id')['zone_id']
        zone_to_node = self.zones['node_id']
        for origin in self.cells['node_id']:
            tier_sum = vals.cells_to_cells.get(origin, np.array([])).sum()
            zone_origin = zone_to_node[cell_to_zone[origin]]
            if vals.zones_to_zones is not None:
                tier_sum += vals.zones_to_zones.get(zone_origin, np.array([])).sum()
            self.assertEqual(tier_sum, total, f'origin {origin}: {tier_sum} != {total}')


class GetEuclidianDistsTestCase(unittest.TestCase):
    """`get_euclidian_dists` returns a TieredODPairs of float distance arrays."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_dists_pair_position_wise_with_pairs(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5)
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
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id')
        dists = get_euclidian_dists(self.nodes, pairs, dtype=np.float32)
        first = next(iter(dists.cells_to_cells.values()))
        self.assertEqual(first.dtype, np.float32)

    def test_dists_third_tier_zone_to_region(self):
        pairs = get_pairs(self.cells, r_cells=0.0, node_column='node_id',
                          zones=self.zones, r_zones=1.0,
                          regions=self.regions, r_regions=2.0)
        dists = get_euclidian_dists(self.nodes, pairs)
        assert dists.zones_to_regions is not None
        # ZN0 at (1.0, 0.5), RN1 at (1.0, 2.0) → d = 1.5.
        self.assertAlmostEqual(dists.zones_to_regions['ZN0'][0], 1.5, places=6)


class ValidationTestCase(unittest.TestCase):
    """Input contract enforcement."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_missing_node_column_raises(self):
        bad = self.cells.drop(columns='node_id')
        with self.assertRaisesRegex(ValueError, "missing required column 'node_id'"):
            get_pairs(bad, r_cells=1.0, node_column='node_id')

    def test_zones_without_r_zones_raises(self):
        with self.assertRaisesRegex(ValueError, 'r_zones'):
            get_pairs(self.cells, r_cells=1.0, node_column='node_id', zones=self.zones)

    def test_regions_without_zones_raises(self):
        with self.assertRaisesRegex(ValueError, 'requires.*zones'):
            get_pairs(self.cells, r_cells=1.0, node_column='node_id',
                      regions=self.regions, r_regions=1.0)

    def test_cells_without_zone_id_raises(self):
        bad = self.cells.drop(columns='zone_id')
        with self.assertRaisesRegex(ValueError, "'zone_id'"):
            get_pairs(bad, r_cells=1.0, node_column='node_id',
                      zones=self.zones, r_zones=1.0)


class DescribeTestCase(unittest.TestCase):
    """`TieredODPairs.describe` prints a per-tier summary for both ID and value tables."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    def test_describe_id_pairs_cells_only(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id')
        out = pairs.describe()
        self.assertIn('cells_to_cells', out)
        self.assertIn('zones_to_zones: None', out)

    def test_describe_id_pairs_all_tiers(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5)
        out = pairs.describe()
        self.assertIn('cells_to_cells', out)
        self.assertIn('zones_to_zones', out)

    def test_describe_distance_values_shows_stats(self):
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5)
        dists = get_euclidian_dists(self.nodes, pairs)
        out = dists.describe()
        # Numeric tier should produce a stats line with 'mean' / 'median'.
        self.assertIn('mean', out)
        self.assertIn('median', out)


class GetPairsMaskFiltersTestCase(unittest.TestCase):
    """`orig_cells` / `dest_cells` / `dest_zones` / `dest_regions` masks
    restrict origins and destinations to subsets of the full universe.
    """

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.cells, cls.zones, cls.regions = _build_world()

    # ----- baseline: no mask = current behaviour -----

    def test_no_masks_matches_baseline(self):
        """All four mask kwargs unspecified = identical to the no-filter call."""
        pairs_a = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                            zones=self.zones, r_zones=2.5)
        pairs_b = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                            zones=self.zones, r_zones=2.5,
                            orig_cells=None, dest_cells=None, dest_zones=None)
        assert pairs_a.cells_to_cells is not None
        assert pairs_b.cells_to_cells is not None
        self.assertSetEqual(set(pairs_a.cells_to_cells.keys()),
                            set(pairs_b.cells_to_cells.keys()))
        for k in pairs_a.cells_to_cells:
            self.assertCountEqual(pairs_a.cells_to_cells[k].tolist(),
                                  pairs_b.cells_to_cells[k].tolist())

    # ----- orig_cells: filter origins -----

    def test_orig_cells_filters_origins(self):
        """orig_cells=mask drops cells where False from being origins entirely."""
        # Mark only C0 and C4 as origins (rest are excluded).
        orig = self.cells.index.isin(['C0', 'C4'])
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          orig_cells=orig)
        # Only N0 and N4 (cells C0 and C4) should appear as cells_to_cells origins.
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {'N0', 'N4'})

    def test_orig_cells_excludes_zone_tier_origin_when_no_origins_in_zone(self):
        """A zone with no origin cells in it should not appear as zone-tier origin."""
        # No cells in Z2 (cells C6-C8) are origins.
        orig = ~self.cells['zone_id'].isin(['Z2']).to_numpy()
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          regions=self.regions, r_regions=10.0,
                          orig_cells=orig)
        assert pairs.zones_to_zones is not None
        # ZN2 is Z2's zone node — it should NOT appear as a zones_to_zones origin.
        self.assertNotIn('ZN2', pairs.zones_to_zones.keys())
        assert pairs.zones_to_regions is not None
        # Same for zones_to_regions origins.
        self.assertNotIn('ZN2', pairs.zones_to_regions.keys())

    # ----- dest_cells: filter cell-tier destinations -----

    def test_dest_cells_filters_destinations(self):
        """dest_cells=mask drops cells where False from being cell-tier destinations."""
        # Only C0, C1, C2 are valid destinations.
        dest = self.cells.index.isin(['C0', 'C1', 'C2'])
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          dest_cells=dest)
        # Every origin's cell-tier destinations should be a subset of {N0, N1, N2}.
        allowed = {'N0', 'N1', 'N2'}
        for orig, dests in pairs.cells_to_cells.items():
            self.assertTrue(
                set(dests.tolist()).issubset(allowed),
                f"Origin {orig!r} has dests {dests.tolist()} outside allowed set.")

    def test_dest_cells_does_not_affect_zone_tier(self):
        """dest_cells filters cell-tier dests only; zone-tier dests are independent."""
        dest = self.cells.index.isin(['C0', 'C1'])
        pairs_filt = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                               zones=self.zones, r_zones=2.5, dest_cells=dest)
        pairs_full = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                               zones=self.zones, r_zones=2.5)
        assert pairs_filt.zones_to_zones is not None
        assert pairs_full.zones_to_zones is not None
        # zones_to_zones structure should be identical for the two calls.
        self.assertSetEqual(set(pairs_filt.zones_to_zones.keys()),
                            set(pairs_full.zones_to_zones.keys()))
        for k in pairs_filt.zones_to_zones:
            self.assertCountEqual(pairs_filt.zones_to_zones[k].tolist(),
                                  pairs_full.zones_to_zones[k].tolist())

    # ----- dest_zones: filter zone-tier destinations -----

    def test_dest_zones_filters_zone_destinations(self):
        """dest_zones=mask drops zones where False from being zone-tier destinations."""
        # Only Z1 is a valid zone-tier destination.
        dest_z = self.zones.index.isin(['Z1'])
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          dest_zones=dest_z)
        # Every zones_to_zones destination should be ZN1.
        assert pairs.zones_to_zones is not None
        for orig, dests in pairs.zones_to_zones.items():
            self.assertTrue(
                all(d == 'ZN1' for d in dests.tolist()),
                f"Origin {orig!r} has unexpected dests {dests.tolist()}.")

    # ----- dest_regions: filter region-tier destinations -----

    def test_dest_regions_filters_region_destinations(self):
        """dest_regions=mask drops regions where False from being region-tier dests."""
        dest_r = self.regions.index.isin(['R0'])
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          regions=self.regions, r_regions=10.0,
                          dest_regions=dest_r)
        assert pairs.zones_to_regions is not None
        # Every region-tier dest should be RN0 (R0's node).
        for orig, dests in pairs.zones_to_regions.items():
            for d in dests.tolist():
                self.assertEqual(d, 'RN0', f"Unexpected dest {d!r}")

    # ----- combined filters -----

    def test_combined_orig_and_dest_filters(self):
        """orig + dest filters compose: only filtered origins route to filtered dests."""
        orig = self.cells.index.isin(['C0', 'C4'])
        dest = self.cells.index.isin(['C0', 'C1', 'C4'])
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          orig_cells=orig, dest_cells=dest)
        # Only N0 and N4 appear as origins.
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {'N0', 'N4'})
        # Each origin's dests are in {N0, N1, N4} (cells C0, C1, C4).
        allowed = {'N0', 'N1', 'N4'}
        for orig_node, dests in pairs.cells_to_cells.items():
            self.assertTrue(set(dests.tolist()).issubset(allowed))

    # ----- cells-only (no zones) variant -----

    def test_masks_work_in_cells_only_mode(self):
        """orig_cells and dest_cells filters work when zones aren't provided."""
        orig = self.cells.index.isin(['C0', 'C1'])
        dest = self.cells.index.isin(['C0', 'C3'])
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          orig_cells=orig, dest_cells=dest)
        self.assertIsNone(pairs.zones_to_zones)
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {'N0', 'N1'})
        allowed = {'N0', 'N3'}
        for orig_node, dests in pairs.cells_to_cells.items():
            self.assertTrue(set(dests.tolist()).issubset(allowed))

    # ----- validation -----

    def test_wrong_length_mask_raises(self):
        wrong = np.array([True] * (len(self.cells) + 1))
        with self.assertRaisesRegex(ValueError, "length"):
            get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                      zones=self.zones, r_zones=2.5, orig_cells=wrong)

    def test_non_boolean_mask_raises(self):
        not_bool = np.arange(len(self.cells))
        with self.assertRaisesRegex(ValueError, "boolean"):
            get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                      zones=self.zones, r_zones=2.5, dest_cells=not_bool)

    def test_series_mask_accepted(self):
        """pd.Series boolean masks work the same as numpy arrays."""
        orig = self.cells['zone_id'] == 'Z0'   # pd.Series of booleans
        pairs = get_pairs(self.cells, r_cells=1.5, node_column='node_id',
                          zones=self.zones, r_zones=2.5,
                          orig_cells=orig)
        # Origins limited to N0-N3 (Z0's cells).
        self.assertSetEqual(set(pairs.cells_to_cells.keys()), {'N0', 'N1', 'N2', 'N3'})


class AggregateAcrossModesTestCase(unittest.TestCase):
    """`aggregate_across_modes` combines per-mode geo-keyed (pairs, costs) into
    a combined (union_pairs, combined_costs)."""

    def _aligned_geo_odms(self):
        """Two modes with the SAME dest sets per origin (no union needed) —
        used for the basic aggregator semantics tests."""
        walk_pairs = TieredODGeoPairs(
            cells_to_cells={'A': np.array(['X', 'Y', 'Z'])},
            zones_to_zones={'ZA': np.array(['ZX', 'ZY'])},
        )
        walk_costs = TieredODGeoPairs(
            cells_to_cells={'A': np.array([100., 200., 300.])},
            zones_to_zones={'ZA': np.array([1500., 5000.])},
        )
        car_pairs = TieredODGeoPairs(
            cells_to_cells={'A': np.array(['X', 'Y', 'Z'])},
            zones_to_zones={'ZA': np.array(['ZX', 'ZY'])},
        )
        car_costs = TieredODGeoPairs(
            cells_to_cells={'A': np.array([50., 250., 100.])},
            zones_to_zones={'ZA': np.array([800., 4000.])},
        )
        return {'walk': (walk_pairs, walk_costs), 'car': (car_pairs, car_costs)}

    def test_returns_geo_pairs(self):
        union_pairs, combined = aggregate_across_modes(
            self._aligned_geo_odms(), aggregator='min')
        self.assertIsInstance(union_pairs, TieredODGeoPairs)
        self.assertIsInstance(combined, TieredODGeoPairs)

    def test_min_per_pair(self):
        """`min` aggregator picks the cheapest mode per OD pair."""
        union_pairs, combined = aggregate_across_modes(
            self._aligned_geo_odms(), aggregator='min')
        # dest order in union is sorted: X, Y, Z.
        self.assertEqual(list(union_pairs.cells_to_cells['A']), ['X', 'Y', 'Z'])
        np.testing.assert_array_equal(combined.cells_to_cells['A'],
                                      np.array([50., 200., 100.]))
        self.assertEqual(list(union_pairs.zones_to_zones['ZA']), ['ZX', 'ZY'])
        np.testing.assert_array_equal(combined.zones_to_zones['ZA'],
                                      np.array([800., 4000.]))

    def test_logsum_scale_1_canonical(self):
        """`logsum` with scale=1 returns -ln Σ exp(-cost). Hand-check one entry."""
        _, combined = aggregate_across_modes(
            self._aligned_geo_odms(), aggregator='logsum', scale=1.0)
        # First cell-tier OD (dest X): walk=100, car=50 → ≈ 50.
        self.assertAlmostEqual(combined.cells_to_cells['A'][0], 50.0, places=5)

    def test_logsum_two_equal_costs(self):
        """Two modes at identical cost: logsum = cost - ln(2)."""
        a_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D'])})
        a_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([10.])})
        b_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D'])})
        b_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([10.])})
        _, combined = aggregate_across_modes(
            {'a': (a_pairs, a_costs), 'b': (b_pairs, b_costs)},
            aggregator='logsum', scale=1.0)
        # -ln(2 · exp(-10)) = 10 - ln(2)
        self.assertAlmostEqual(combined.cells_to_cells['O'][0], 10.0 - np.log(2))

    def test_logsum_with_scale(self):
        a_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D'])})
        a_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([20.])})
        b_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D'])})
        b_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([20.])})
        scale = 5.0
        _, combined = aggregate_across_modes(
            {'a': (a_pairs, a_costs), 'b': (b_pairs, b_costs)},
            aggregator='logsum', scale=scale)
        # -5 · ln(2 · exp(-4)) = 20 - 5·ln(2)
        self.assertAlmostEqual(combined.cells_to_cells['O'][0],
                               20.0 - scale * np.log(2))

    def test_custom_callable_aggregator(self):
        """Callable aggregator: takes (n_modes, n_dests) and returns (n_dests,)."""
        def weighted_mean(stacked: np.ndarray) -> np.ndarray:
            return (1 / 3) * stacked[0] + (2 / 3) * stacked[1]
        _, combined = aggregate_across_modes(
            self._aligned_geo_odms(), aggregator=weighted_mean)
        expected_first = (1 / 3) * 100. + (2 / 3) * 50.
        self.assertAlmostEqual(combined.cells_to_cells['A'][0], expected_first)

    # ---------- Union + inf-fill semantics ----------

    def test_disjoint_dests_per_origin_union_with_inf_fill(self):
        """When modes have different dest sets, the result has the union; the
        mode that lacks a given dest contributes inf to that position."""
        walk_pairs = TieredODGeoPairs(
            cells_to_cells={'A': np.array(['X', 'Y'])})
        walk_costs = TieredODGeoPairs(
            cells_to_cells={'A': np.array([10., 20.])})
        car_pairs = TieredODGeoPairs(
            cells_to_cells={'A': np.array(['Y', 'Z'])})
        car_costs = TieredODGeoPairs(
            cells_to_cells={'A': np.array([5., 7.])})
        union_pairs, combined = aggregate_across_modes(
            {'walk': (walk_pairs, walk_costs), 'car': (car_pairs, car_costs)},
            aggregator='min')
        # Union dests = {X, Y, Z}, sorted.
        self.assertEqual(list(union_pairs.cells_to_cells['A']), ['X', 'Y', 'Z'])
        # X: walk=10, car=inf → min=10. Y: walk=20, car=5 → min=5. Z: walk=inf, car=7 → 7.
        np.testing.assert_array_equal(combined.cells_to_cells['A'],
                                      np.array([10., 5., 7.]))

    def test_disjoint_origins_union(self):
        """Modes with different origin sets → union of origins appears in result."""
        walk_pairs = TieredODGeoPairs(
            cells_to_cells={'A': np.array(['D'])})
        walk_costs = TieredODGeoPairs(
            cells_to_cells={'A': np.array([10.])})
        car_pairs = TieredODGeoPairs(
            cells_to_cells={'B': np.array(['D'])})
        car_costs = TieredODGeoPairs(
            cells_to_cells={'B': np.array([5.])})
        union_pairs, combined = aggregate_across_modes(
            {'walk': (walk_pairs, walk_costs), 'car': (car_pairs, car_costs)},
            aggregator='min')
        self.assertSetEqual(set(union_pairs.cells_to_cells.keys()), {'A', 'B'})
        # A only routed by walk → A's combined = walk cost.
        np.testing.assert_array_equal(combined.cells_to_cells['A'], np.array([10.]))
        # B only routed by car → B's combined = car cost.
        np.testing.assert_array_equal(combined.cells_to_cells['B'], np.array([5.]))

    def test_min_with_inf_treated_as_unreachable(self):
        """`inf` is finite-worst; if any mode is reachable, min picks that one."""
        a_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D1', 'D2'])})
        a_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([np.inf, 100.])})
        b_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D1', 'D2'])})
        b_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([50., np.inf])})
        _, combined = aggregate_across_modes(
            {'a': (a_pairs, a_costs), 'b': (b_pairs, b_costs)}, aggregator='min')
        np.testing.assert_array_equal(combined.cells_to_cells['O'], np.array([50., 100.]))

    def test_logsum_inf_contributes_nothing(self):
        """exp(-inf) = 0, so an unreachable mode does not affect the logsum."""
        a_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D1', 'D2'])})
        a_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([10., np.inf])})
        b_pairs = TieredODGeoPairs(cells_to_cells={'O': np.array(['D1', 'D2'])})
        b_costs = TieredODGeoPairs(cells_to_cells={'O': np.array([np.inf, 10.])})
        _, combined = aggregate_across_modes(
            {'a': (a_pairs, a_costs), 'b': (b_pairs, b_costs)},
            aggregator='logsum', scale=1.0)
        np.testing.assert_array_almost_equal(combined.cells_to_cells['O'],
                                             np.array([10., 10.]))

    def test_single_mode_passthrough(self):
        """One mode: combined ODM equals that mode's ODM (no-op)."""
        only_pairs = TieredODGeoPairs(cells_to_cells={'A': np.array(['X', 'Y', 'Z'])})
        only_costs = TieredODGeoPairs(cells_to_cells={'A': np.array([1., 2., 3.])})
        union_pairs, combined = aggregate_across_modes(
            {'only': (only_pairs, only_costs)}, aggregator='min')
        self.assertEqual(list(union_pairs.cells_to_cells['A']), ['X', 'Y', 'Z'])
        np.testing.assert_array_equal(combined.cells_to_cells['A'], np.array([1., 2., 3.]))

    def test_missing_tier_stays_none(self):
        """Tiers that are None in all inputs stay None in the output."""
        a_pairs = TieredODGeoPairs(cells_to_cells={'A': np.array(['X'])})
        a_costs = TieredODGeoPairs(cells_to_cells={'A': np.array([1.])})
        b_pairs = TieredODGeoPairs(cells_to_cells={'A': np.array(['X'])})
        b_costs = TieredODGeoPairs(cells_to_cells={'A': np.array([2.])})
        union_pairs, combined = aggregate_across_modes(
            {'walk': (a_pairs, a_costs), 'car': (b_pairs, b_costs)},
            aggregator='min')
        self.assertIsNone(union_pairs.zones_to_zones)
        self.assertIsNone(combined.zones_to_zones)

    def test_inconsistent_tier_structure_raises(self):
        """If one mode has a tier and another doesn't, raise."""
        a_pairs = TieredODGeoPairs(cells_to_cells={'A': np.array(['X'])})
        a_costs = TieredODGeoPairs(cells_to_cells={'A': np.array([1.])})
        b_pairs = TieredODGeoPairs(
            cells_to_cells={'A': np.array(['X'])},
            zones_to_zones={'ZA': np.array(['ZX'])})
        b_costs = TieredODGeoPairs(
            cells_to_cells={'A': np.array([2.])},
            zones_to_zones={'ZA': np.array([5.])})
        with self.assertRaisesRegex(ValueError, "tier"):
            aggregate_across_modes(
                {'walk': (a_pairs, a_costs), 'car': (b_pairs, b_costs)},
                aggregator='min')

    def test_empty_odms_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            aggregate_across_modes({}, aggregator='min')

    def test_unknown_aggregator_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown aggregator"):
            aggregate_across_modes(self._aligned_geo_odms(), aggregator='nope')

    def test_node_keyed_input_raises_type_error(self):
        """Cross-modal aggregation must run in geo-unit space, not node space."""
        walk_node = TieredODNodePairs(
            cells_to_cells={'A': np.array(['X'])})
        walk_node_costs = TieredODNodePairs(
            cells_to_cells={'A': np.array([1.])})
        with self.assertRaisesRegex(TypeError, "TieredODGeoPairs"):
            aggregate_across_modes(
                {'walk': (walk_node, walk_node_costs)}, aggregator='min')


if __name__ == '__main__':
    unittest.main()
