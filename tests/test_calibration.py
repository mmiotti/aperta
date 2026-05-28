"""Tests for `aperta.calibration.calibrate_edge_weights`.

Synthetic-graph recovery tests — build a graph + a feature with a known
per-feature coefficient, generate ground truth from the known model, fit,
verify the fitted coefficient matches.

Run with:
    python -m unittest tests.test_calibration
"""

import unittest

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from aperta.calibration import (
    calibrate_edge_weights,
    evaluate_against_counters,
    snap_counters_to_edges,
)


def _chain_graph(
    n: int = 20,
    edge_len: float = 100.0,
    speed_kph: float = 36.0,
    slow_zone_frac: float = 0.4,
    seed: int = 0,
) -> nx.MultiDiGraph:
    """Linear chain of `n` nodes with bidirectional edges between neighbours.

    Each edge has `length=edge_len` and `speed_kph=speed_kph`. A random
    `slow_zone` (0 or 1) is assigned to each edge with probability
    `slow_zone_frac`. Chosen specifically so every origin-destination pair
    has a *unique* shortest path — tie-breaking between equally-optimal
    paths is implementation-defined and would silently corrupt the
    recovered coefficients.
    """
    rng = np.random.default_rng(seed)
    g = nx.MultiDiGraph(crs="EPSG:2056")
    for i in range(n):
        g.add_node(i, x=float(i * edge_len), y=0.0)
    for i in range(n - 1):
        slow = float(rng.random() < slow_zone_frac)
        g.add_edge(i, i + 1, length=edge_len, speed_kph=speed_kph, slow_zone=slow)
        g.add_edge(i + 1, i, length=edge_len, speed_kph=speed_kph, slow_zone=slow)
    return g


def _ground_truth_from_model(
    g: nx.MultiDiGraph,
    n_trips: int = 200,
    slow_zone_coef: float = 5.0,
    noise_std: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate trips with known time = baseline_time × (1 + coef · slow_zone_avg).

    Routes each trip along its baseline (length/speed) shortest path so the
    ground-truth time exactly matches the multiplier-feature model. Adds
    small Gaussian noise so OLS has variance to fit on.
    """
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes)
    rows = []
    for _ in range(n_trips):
        o, d = rng.choice(len(nodes), 2, replace=False)
        o_node, d_node = nodes[o], nodes[d]
        path = nx.shortest_path(g, o_node, d_node, weight="length")
        edges = list(zip(path[:-1], path[1:]))
        baseline_time = sum(
            g[u][v][0]["length"] / (g[u][v][0]["speed_kph"] / 3.6) for u, v in edges
        )
        total_length = sum(g[u][v][0]["length"] for u, v in edges)
        slow_time = (
            sum(g[u][v][0]["length"] * g[u][v][0]["slow_zone"] for u, v in edges) / total_length
            if total_length > 0
            else 0
        )
        observed = baseline_time * (1 + slow_zone_coef * slow_time)
        observed += rng.normal(0, noise_std)
        rows.append(
            {
                "orig_x": g.nodes[o_node]["x"],
                "orig_y": g.nodes[o_node]["y"],
                "dest_x": g.nodes[d_node]["x"],
                "dest_y": g.nodes[d_node]["y"],
                "time_measured": observed,
                "dist_measured": total_length,
            }
        )
    return pd.DataFrame(rows)


class CalibrateEdgeWeightsTestCase(unittest.TestCase):
    """End-to-end recovery: build a graph + a feature with a known
    coefficient, generate ground truth, fit, verify the recovered
    coefficient matches the planted one.
    """

    def test_recovers_known_multiplier_coefficient(self):
        """A synthetic feature with a planted coefficient is recovered to
        within 10 % by OLS.

        Uses `n_iterations=1` deliberately: ground truth is generated on
        baseline-routed paths through the slow zone, so iteration 2 would
        re-route AROUND the slow zone (its calibrated edges are now slower),
        a model misspecification that's expected when the synthetic ground
        truth's path-choice mechanism differs from the calibrated routing.
        The single-iteration OLS-recovery test is the right scope for the
        unit test.
        """
        g = _chain_graph(n=20)
        gt = _ground_truth_from_model(g, n_trips=400, slow_zone_coef=5.0, noise_std=2.0, seed=0)
        result = calibrate_edge_weights(
            g,
            gt,
            baseline_speed_attr="speed_kph",
            multiplier_features={"slow_zone": 0.0},
            additive_route_features={},
            additive_endpoint_features={},
            constant=True,
            n_iterations=1,
            snap_max_distance=10.0,
            min_trip_distance=0.0,
            max_trip_distance=10_000.0,
            max_dist_to_line_ratio=10.0,
        )
        # α (baseline coefficient) should be near 1.0.
        alpha = result.coefficients.loc["baseline_time", "coef"]
        self.assertAlmostEqual(alpha, 1.0, places=1)
        # Recovered multiplier coefficient ≈ 5.0.
        recovered = result.coefficients.loc["slow_zone__mult", "coef"]
        self.assertAlmostEqual(recovered, 5.0, delta=0.5)
        # R² should be very high (data was generated FROM the model).
        self.assertGreater(result.r_squared, 0.99)

    def test_returns_expected_result_fields(self):
        g = _chain_graph(n=10)
        gt = _ground_truth_from_model(g, n_trips=100, seed=1)
        result = calibrate_edge_weights(
            g,
            gt,
            multiplier_features={"slow_zone": 0.0},
            n_iterations=1,
            snap_max_distance=10.0,
            min_trip_distance=0.0,
            max_trip_distance=10_000.0,
            max_dist_to_line_ratio=10.0,
        )
        # Required output shape.
        self.assertIn("baseline_time", result.coefficients.index)
        self.assertIn("slow_zone__mult", result.coefficients.index)
        self.assertEqual(len(result.predicted_times), result.n_used)
        self.assertEqual(len(result.observed_times), result.n_used)
        self.assertIn(result.edge_duration_attr, next(iter(g.edges(data=True)))[2])
        self.assertGreater(result.r_squared, 0.0)
        self.assertGreater(result.rmse, 0.0)
        self.assertEqual(len(result.iter_log), 1)

    def test_missing_ground_truth_column_raises(self):
        g = _chain_graph(n=5)
        gt = pd.DataFrame(
            {"orig_x": [0], "orig_y": [0], "dest_x": [400], "dest_y": [0]}
        )  # no time_measured
        with self.assertRaisesRegex(ValueError, "time_measured"):
            calibrate_edge_weights(g, gt, multiplier_features={"slow_zone": 0.0})

    def test_no_features_runs(self):
        """With no features, the fit is just `time ≈ α · baseline_time`."""
        g = _chain_graph(n=10)
        # Ground truth = exact baseline (no feature corrections).
        rng = np.random.default_rng(0)
        nodes = list(g.nodes)
        rows = []
        for _ in range(100):
            o, d = rng.choice(len(nodes), 2, replace=False)
            o_node, d_node = nodes[o], nodes[d]
            path = nx.shortest_path(g, o_node, d_node, weight="length")
            baseline_time = sum(
                g[u][v][0]["length"] / (g[u][v][0]["speed_kph"] / 3.6)
                for u, v in zip(path[:-1], path[1:])
            )
            rows.append(
                {
                    "orig_x": g.nodes[o_node]["x"],
                    "orig_y": g.nodes[o_node]["y"],
                    "dest_x": g.nodes[d_node]["x"],
                    "dest_y": g.nodes[d_node]["y"],
                    "time_measured": baseline_time + rng.normal(0, 1.0),
                    "dist_measured": sum(g[u][v][0]["length"] for u, v in zip(path[:-1], path[1:])),
                }
            )
        gt = pd.DataFrame(rows)
        result = calibrate_edge_weights(
            g,
            gt,
            constant=False,
            n_iterations=1,
            snap_max_distance=10.0,
            min_trip_distance=0.0,
            max_trip_distance=10_000.0,
            max_dist_to_line_ratio=10.0,
        )
        # α should be very close to 1.0.
        self.assertAlmostEqual(result.coefficients.loc["baseline_time", "coef"], 1.0, places=2)


class SnapCountersToEdgesTestCase(unittest.TestCase):
    """`snap_counters_to_edges` snaps directional counters to the right edge
    using a bearing-aware nearest-line match."""

    def _directed_two_way_graph(self) -> nx.MultiDiGraph:
        """A single 100 m east-west road, two directed edges
        (one east-bound, one west-bound) carrying opposite-direction
        traffic. Mirrors the OSMnx pattern for two-way roads."""
        g = nx.MultiDiGraph(crs="EPSG:2056")
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        line = LineString([(0.0, 0.0), (100.0, 0.0)])
        g.add_edge(1, 2, key=0, geometry=line, length=100.0)
        g.add_edge(2, 1, key=0, geometry=LineString(list(line.coords)[::-1]), length=100.0)
        return g

    def test_directional_counter_picks_correct_edge(self):
        """East-bound counter (bearing 90°) snaps to (1,2); west-bound
        counter (bearing 270°) snaps to (2,1)."""
        g = self._directed_two_way_graph()
        counters = gpd.GeoDataFrame(
            {"bearing_deg": [90.0, 270.0]},
            geometry=[Point(50.0, 1.0), Point(50.0, -1.0)],
            crs="EPSG:2056",
        )
        result = snap_counters_to_edges(counters, g, search_radius=10.0, bearing_tol_deg=20.0)
        self.assertEqual((result.loc[0, "u"], result.loc[0, "v"]), (1, 2))
        self.assertEqual((result.loc[1, "u"], result.loc[1, "v"]), (2, 1))

    def test_bearing_out_of_tolerance_no_match(self):
        """A counter pointing north (90° off from the east-west road)
        gets no match even though the road is well within search radius."""
        g = self._directed_two_way_graph()
        counters = gpd.GeoDataFrame(
            {"bearing_deg": [0.0]},  # north
            geometry=[Point(50.0, 1.0)],
            crs="EPSG:2056",
        )
        result = snap_counters_to_edges(counters, g, search_radius=10.0, bearing_tol_deg=20.0)
        self.assertTrue(pd.isna(result.loc[0, "u"]))

    def test_eligible_edges_filter(self):
        """Highway counter restricted to highway edges via callback."""
        g = self._directed_two_way_graph()
        # Tag (1,2) as motorway, (2,1) as residential — same geometry but
        # different class. Counter restricted to motorway should pick (1,2).
        g[1][2][0]["highway"] = "motorway"
        g[2][1][0]["highway"] = "residential"
        counters = gpd.GeoDataFrame(
            {"bearing_deg": [90.0]},
            geometry=[Point(50.0, 0.5)],
            crs="EPSG:2056",
        )

        def only_motorway(_counter, cands):
            return cands[cands["highway"] == "motorway"]

        result = snap_counters_to_edges(
            counters, g, search_radius=10.0, eligible_edges=only_motorway
        )
        self.assertEqual((result.loc[0, "u"], result.loc[0, "v"]), (1, 2))

    def test_missing_bearing_column_raises(self):
        g = self._directed_two_way_graph()
        counters = gpd.GeoDataFrame(geometry=[Point(50.0, 0.0)], crs="EPSG:2056")
        with self.assertRaises(ValueError):
            snap_counters_to_edges(counters, g)


class EvaluateAgainstCountersTestCase(unittest.TestCase):
    """`evaluate_against_counters` reports R², regression slope, RMSE."""

    def _counters(self, observed: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "u": [1, 2, 3, 4],
                "v": [2, 3, 4, 5],
                "k": [0, 0, 0, 0],
                "traffic_cars": observed,
            }
        )

    def test_perfect_match_returns_r2_one_slope_one(self):
        counters = self._counters([100.0, 200.0, 300.0, 400.0])
        modeled = pd.Series(
            {(1, 2, 0): 100.0, (2, 3, 0): 200.0, (3, 4, 0): 300.0, (4, 5, 0): 400.0}
        )
        result = evaluate_against_counters(modeled, counters)
        self.assertAlmostEqual(result["r2"], 1.0)
        self.assertAlmostEqual(result["slope"], 1.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertEqual(result["n_matched"], 4)

    def test_scale_invariant_r2_slope_recovered(self):
        """Modeled = 2 × observed: perfect correlation but slope = 2.
        R² should still be 1.0 (correlation is scale-invariant)."""
        counters = self._counters([100.0, 200.0, 300.0, 400.0])
        modeled = pd.Series(
            {(1, 2, 0): 200.0, (2, 3, 0): 400.0, (3, 4, 0): 600.0, (4, 5, 0): 800.0}
        )
        result = evaluate_against_counters(modeled, counters)
        self.assertAlmostEqual(result["r2"], 1.0)
        self.assertAlmostEqual(result["slope"], 2.0)
        self.assertGreater(result["rmse"], 0)

    def test_unmatched_counters_dropped(self):
        """Counters with NA u/v/k are excluded from the comparison."""
        counters = self._counters([100.0, 200.0, 300.0, 400.0])
        counters.loc[[2, 3], ["u", "v", "k"]] = pd.NA
        modeled = pd.Series({(1, 2, 0): 100.0, (2, 3, 0): 200.0})
        result = evaluate_against_counters(modeled, counters)
        self.assertEqual(result["n_matched"], 2)


if __name__ == "__main__":
    unittest.main()
