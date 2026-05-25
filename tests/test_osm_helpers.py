"""Tests for `aperta.osm_helpers` — the pure-logic helpers (category-map
parsing + POI categorisation). The end-to-end `fetch_pois` / `fetch_network`
wrappers require osmnx + network access; they're exercised by the Swiss
prep notebook rather than unit-tested here.

Run with:
    cd src && python -m unittest aperta.tests.test_osm_helpers
"""
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from aperta.osm_helpers import (
    categorize_pois,
    osm_tag_query_for_categories,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    'groceries': [
        ('shop:supermarket', 1.0),
        ('shop:convenience', 0.5),
        ('shop:bakery',      0.5),
    ],
    'schools': [
        ('amenity:school', 1.0),
    ],
    # Test multi-tag-key category and integer weight.
    'transit_rail': [
        ('railway:station', 3),
        ('railway:halt', 1),
    ],
}


def _toy_pois() -> gpd.GeoDataFrame:
    """Six POIs covering: 1 supermarket, 1 convenience (also has a bus stop
    tag), 1 bakery, 1 school, 1 railway station, 1 unrelated (fuel)."""
    data = {
        'amenity':  ['fuel', None, None,  'school', None,        'restaurant'],
        'shop':     [None,   'convenience', 'bakery', None,    None,           None],
        'railway':  [None,   None,        None,     None,    'station',       None],
        'highway':  [None,   'bus_stop',  None,     None,    None,           None],
    }
    geom = [Point(i, 0) for i in range(len(data['amenity']))]
    return gpd.GeoDataFrame(
        data, geometry=geom,
        index=pd.Index(['p0', 'p1', 'p2', 'p3', 'p4', 'p5'], name='osm_id'),
        crs='EPSG:4326',
    )


# ---------------------------------------------------------------------------
# osm_tag_query_for_categories
# ---------------------------------------------------------------------------

class OsmTagQueryTestCase(unittest.TestCase):

    def test_unions_keys_across_categories(self):
        q = osm_tag_query_for_categories(CATEGORY_MAP)
        self.assertSetEqual(set(q.keys()), {'shop', 'amenity', 'railway'})

    def test_unions_values_per_key(self):
        q = osm_tag_query_for_categories(CATEGORY_MAP)
        self.assertSetEqual(set(q['shop']), {'supermarket', 'convenience', 'bakery'})
        self.assertSetEqual(set(q['amenity']), {'school'})
        self.assertSetEqual(set(q['railway']), {'station', 'halt'})

    def test_values_are_sorted_for_determinism(self):
        q = osm_tag_query_for_categories(CATEGORY_MAP)
        for key, vals in q.items():
            self.assertEqual(vals, sorted(vals), f"values for {key!r} not sorted")

    def test_duplicate_tag_across_categories_deduped(self):
        """Same (tag:value) reused across categories collapses to one value."""
        cm = {
            'a': [('shop:supermarket', 1.0)],
            'b': [('shop:supermarket', 0.5)],  # same OSM tag, different weight
        }
        q = osm_tag_query_for_categories(cm)
        self.assertEqual(q['shop'], ['supermarket'])

    def test_missing_colon_raises(self):
        with self.assertRaisesRegex(ValueError, "'key:value'"):
            osm_tag_query_for_categories({'bad': [('no_colon_here', 1.0)]})


# ---------------------------------------------------------------------------
# categorize_pois
# ---------------------------------------------------------------------------

class CategorizePoisTestCase(unittest.TestCase):

    def test_count_and_weight_columns_added(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP)
        for cat in CATEGORY_MAP:
            self.assertIn(cat, out.columns)
            self.assertIn(f'{cat}_weight', out.columns)

    def test_count_values(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP, drop_unmatched=False)
        # p0 (fuel): no match anywhere.
        self.assertEqual(out.loc['p0', 'groceries'], 0)
        # p1 (shop:convenience): matches groceries (1 listed pair).
        self.assertEqual(out.loc['p1', 'groceries'], 1)
        # p2 (shop:bakery): matches groceries (1 listed pair).
        self.assertEqual(out.loc['p2', 'groceries'], 1)
        # p3 (amenity:school): matches schools, not groceries.
        self.assertEqual(out.loc['p3', 'schools'], 1)
        self.assertEqual(out.loc['p3', 'groceries'], 0)
        # p4 (railway:station): matches transit_rail.
        self.assertEqual(out.loc['p4', 'transit_rail'], 1)

    def test_weight_values(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP, drop_unmatched=False)
        # p1 (convenience): weight = 0.5.
        self.assertEqual(out.loc['p1', 'groceries_weight'], 0.5)
        # p2 (bakery): weight = 0.5.
        self.assertEqual(out.loc['p2', 'groceries_weight'], 0.5)
        # p4 (railway:station): weight = 3 (integer weight handled).
        self.assertEqual(out.loc['p4', 'transit_rail_weight'], 3.0)

    def test_drop_unmatched(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP, drop_unmatched=True)
        # p0 (fuel) and p5 (restaurant) match nothing → dropped.
        self.assertNotIn('p0', out.index)
        self.assertNotIn('p5', out.index)
        # The rest stay.
        self.assertSetEqual(set(out.index), {'p1', 'p2', 'p3', 'p4'})

    def test_drop_unmatched_false_keeps_all(self):
        toy = _toy_pois()
        out = categorize_pois(toy, CATEGORY_MAP, drop_unmatched=False)
        self.assertEqual(len(out), len(toy))

    def test_missing_tag_column_silently_skipped(self):
        """If a category references a tag key the DataFrame doesn't have,
        that pair simply doesn't match — no error."""
        toy = _toy_pois().drop(columns=['railway'])
        out = categorize_pois(toy, CATEGORY_MAP, drop_unmatched=False)
        # transit_rail columns still added, but always zero.
        self.assertEqual(out['transit_rail'].sum(), 0)
        self.assertEqual(out['transit_rail_weight'].sum(), 0.0)

    def test_multi_match_within_category(self):
        """A row matching multiple (tag:value) pairs in one category gets
        count > 1 and summed weight."""
        # Make a single row that matches both shop:supermarket AND shop:bakery
        # (unusual but possible — large food hall etc.). Since we have one
        # `shop` column with a single value, simulate via a custom category.
        cm = {
            'groceries': [
                ('shop:supermarket', 1.0),
                ('amenity:supermarket', 0.5),  # second match condition
            ],
        }
        gdf = gpd.GeoDataFrame(
            {'shop': ['supermarket'], 'amenity': ['supermarket']},
            geometry=[Point(0, 0)],
            index=pd.Index(['p'], name='id'),
            crs='EPSG:4326',
        )
        out = categorize_pois(gdf, cm, drop_unmatched=False)
        self.assertEqual(out.loc['p', 'groceries'], 2)
        self.assertEqual(out.loc['p', 'groceries_weight'], 1.5)

    def test_custom_weight_suffix(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP,
                              weight_suffix='_w', drop_unmatched=False)
        self.assertIn('groceries_w', out.columns)
        self.assertNotIn('groceries_weight', out.columns)

    def test_column_name_collision_raises(self):
        """If a category name already exists as a column, raise instead of
        silently overwriting (would erase OSM tag data)."""
        toy = _toy_pois().assign(groceries='preexisting')
        with self.assertRaisesRegex(ValueError, "groceries"):
            categorize_pois(toy, CATEGORY_MAP)

    def test_input_not_mutated(self):
        toy = _toy_pois()
        cols_before = list(toy.columns)
        _ = categorize_pois(toy, CATEGORY_MAP)
        self.assertEqual(list(toy.columns), cols_before)

    def test_empty_category_map(self):
        """Empty category map = no new columns, no rows dropped."""
        toy = _toy_pois()
        out = categorize_pois(toy, {}, drop_unmatched=True)
        self.assertEqual(len(out), len(toy))
        for c in toy.columns:
            self.assertIn(c, out.columns)


if __name__ == '__main__':
    unittest.main()
