"""Tests for `aperta.geo_processing`.

Run with:
    python -m unittest tests.test_geo_processing
"""

import math
import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from aperta.geo_processing import cross_sum_within_radius


def _points(
    coords: list[tuple[float, float]], index: list, extra: dict | None = None
) -> gpd.GeoDataFrame:
    data = {"geometry": [Point(x, y) for x, y in coords]}
    if extra:
        data.update(extra)
    return gpd.GeoDataFrame(data, index=pd.Index(index, name="id"), crs="EPSG:3857")


class AggregateWithinRadiusTestCase(unittest.TestCase):
    """`cross_sum_within_radius` — cross-set buffer aggregation."""

    def test_count_within_radius(self):
        """Default behaviour: count sources within radius of each target."""
        targets = _points([(0, 0), (10, 0)], index=["T0", "T1"])
        sources = _points(
            [(1, 0), (2, 0), (3, 0), (9, 0), (11, 0)],
            index=["S0", "S1", "S2", "S3", "S4"],
        )
        # T0 at (0,0) with radius 5 → covers S0, S1, S2 (count=3).
        # T1 at (10,0) with radius 5 → covers S3, S4 (count=2).
        result = cross_sum_within_radius(targets, sources, radius=5.0)
        self.assertEqual(list(result.index), ["T0", "T1"])
        self.assertEqual(result.loc["T0"], 3.0)
        self.assertEqual(result.loc["T1"], 2.0)

    def test_no_sources_in_range_returns_zero(self):
        targets = _points([(100, 100)], index=["T"])
        sources = _points([(0, 0), (1, 1)], index=["S0", "S1"])
        result = cross_sum_within_radius(targets, sources, radius=1.0)
        self.assertEqual(result.loc["T"], 0.0)

    def test_weighted_sum(self):
        """`weight_column` switches from count to sum-of-column."""
        targets = _points([(0, 0)], index=["T"])
        sources = _points(
            [(1, 0), (2, 0), (10, 0)],
            index=["S0", "S1", "S2"],
            extra={"pop": [100.0, 50.0, 999.0]},
        )
        # T at (0,0) with radius 5 → S0 (100) + S1 (50) = 150. S2 excluded.
        result = cross_sum_within_radius(targets, sources, radius=5.0, weight_column="pop")
        self.assertEqual(result.loc["T"], 150.0)

    def test_weighted_sum_no_sources_in_range(self):
        targets = _points([(0, 0)], index=["T"])
        sources = _points(
            [(100, 0)],
            index=["S0"],
            extra={"pop": [50.0]},
        )
        result = cross_sum_within_radius(targets, sources, radius=5.0, weight_column="pop")
        self.assertEqual(result.loc["T"], 0.0)

    def test_return_density_divides_by_pi_r_squared(self):
        targets = _points([(0, 0)], index=["T"])
        sources = _points(
            [(1, 0), (2, 0), (3, 0), (4, 0)],
            index=["S0", "S1", "S2", "S3"],
        )
        # 4 sources in range; area = π·5² = 25π.
        result = cross_sum_within_radius(targets, sources, radius=5.0, return_density=True)
        self.assertAlmostEqual(result.loc["T"], 4.0 / (math.pi * 25.0))

    def test_weighted_density(self):
        targets = _points([(0, 0)], index=["T"])
        sources = _points(
            [(1, 0), (2, 0)],
            index=["S0", "S1"],
            extra={"pop": [10.0, 20.0]},
        )
        result = cross_sum_within_radius(
            targets, sources, radius=5.0, weight_column="pop", return_density=True
        )
        # (10 + 20) / (π · 25).
        self.assertAlmostEqual(result.loc["T"], 30.0 / (math.pi * 25.0))

    def test_polygon_inputs_use_centroid(self):
        """Non-point geometries are queried at their centroid."""
        from shapely.geometry import box

        # Two square cells; centroids at (0.5, 0.5) and (10.5, 0.5).
        target_polys = gpd.GeoDataFrame(
            {"geometry": [box(0, 0, 1, 1), box(10, 0, 11, 1)]},
            index=pd.Index(["A", "B"], name="cell_id"),
            crs="EPSG:3857",
        )
        sources = _points(
            [(0.6, 0.5), (0.7, 0.5), (10.5, 0.5)],
            index=["S0", "S1", "S2"],
        )
        # A's centroid (0.5, 0.5) with radius 1 → S0 + S1 = 2.
        # B's centroid (10.5, 0.5) with radius 1 → S2 = 1.
        result = cross_sum_within_radius(target_polys, sources, radius=1.0)
        self.assertEqual(result.loc["A"], 2.0)
        self.assertEqual(result.loc["B"], 1.0)

    def test_geoseries_input_works(self):
        """Pass GeoSeries (not GeoDataFrame) for targets and sources."""
        targets = gpd.GeoSeries([Point(0, 0)], index=pd.Index(["T"], name="id"), crs="EPSG:3857")
        sources = gpd.GeoSeries(
            [Point(1, 0), Point(2, 0)], index=pd.Index(["S0", "S1"], name="id"), crs="EPSG:3857"
        )
        result = cross_sum_within_radius(targets, sources, radius=5.0)
        self.assertEqual(result.loc["T"], 2.0)

    def test_weight_column_requires_geodataframe(self):
        targets = _points([(0, 0)], index=["T"])
        sources_gs = gpd.GeoSeries([Point(1, 0)], index=["S0"], crs="EPSG:3857")
        with self.assertRaisesRegex(ValueError, "GeoDataFrame"):
            cross_sum_within_radius(targets, sources_gs, radius=5.0, weight_column="pop")

    def test_weight_column_missing_raises(self):
        targets = _points([(0, 0)], index=["T"])
        sources = _points([(1, 0)], index=["S0"])  # no 'pop' column
        with self.assertRaisesRegex(ValueError, "pop"):
            cross_sum_within_radius(targets, sources, radius=5.0, weight_column="pop")

    def test_name_propagates(self):
        targets = _points([(0, 0)], index=["T"])
        sources = _points([(1, 0)], index=["S0"])
        result = cross_sum_within_radius(targets, sources, radius=5.0, name="my_density")
        self.assertEqual(result.name, "my_density")

    def test_targets_index_preserved(self):
        targets = _points([(0, 0), (1, 1), (2, 2)], index=["x", "y", "z"])
        sources = _points([(0, 0)], index=["S0"])
        result = cross_sum_within_radius(targets, sources, radius=10.0)
        self.assertEqual(list(result.index), ["x", "y", "z"])


if __name__ == "__main__":
    unittest.main()
