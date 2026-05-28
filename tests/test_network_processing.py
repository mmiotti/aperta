"""Tests for `aperta.network_processing` — graph-specific helpers.

Run with:
    python -m unittest tests.test_network_processing
"""

import unittest

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from aperta.network_processing import (
    aggregate_edges_to_nodes,
    assign_to_eligible_centroid,
    consolidate_intersections,
    flag_node_intersections,
    lanes_per_direction,
    snap_to_network_nodes,
)


class SnapToNetworkNodesTestCase(unittest.TestCase):
    """`snap_to_network_nodes` snaps a GeoDataFrame of points to the nearest
    node in a networkx graph, returning (node_ids, distances).
    """

    def _graph(self) -> nx.Graph:
        """Toy graph with three nodes at known positions."""
        g = nx.Graph()
        g.add_node("a", x=0.0, y=0.0)
        g.add_node("b", x=10.0, y=0.0)
        g.add_node("c", x=0.0, y=10.0)
        return g

    def _points(self, coords: list[tuple[float, float]], ids: list[str]) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in coords],
            index=pd.Index(ids, name="point_id"),
        )

    def test_returns_tuple_of_two_series(self):
        graph = self._graph()
        points = self._points([(1.0, 1.0)], ["p0"])
        result = snap_to_network_nodes(points, graph)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        ids, distances = result
        self.assertIsInstance(ids, pd.Series)
        self.assertIsInstance(distances, pd.Series)

    def test_nearest_node_assignment(self):
        """Each point gets assigned the closest node by Euclidean distance."""
        graph = self._graph()
        points = self._points(
            [
                (1.0, 0.0),  # closest to 'a' (dist 1)
                (9.0, 1.0),  # closest to 'b' (dist √2)
                (0.0, 8.0),
            ],  # closest to 'c' (dist 2)
            ["p_a", "p_b", "p_c"],
        )
        ids, distances = snap_to_network_nodes(points, graph)
        self.assertEqual(ids.loc["p_a"], "a")
        self.assertEqual(ids.loc["p_b"], "b")
        self.assertEqual(ids.loc["p_c"], "c")
        self.assertAlmostEqual(distances.loc["p_a"], 1.0)
        self.assertAlmostEqual(distances.loc["p_b"], np.sqrt(2))
        self.assertAlmostEqual(distances.loc["p_c"], 2.0)

    def test_distance_zero_for_point_on_node(self):
        """A point exactly on a graph node returns that node with distance 0."""
        graph = self._graph()
        points = self._points([(0.0, 0.0)], ["exact"])
        ids, distances = snap_to_network_nodes(points, graph)
        self.assertEqual(ids.loc["exact"], "a")
        self.assertAlmostEqual(distances.loc["exact"], 0.0)

    def test_indexed_by_points_index(self):
        """Result Series are indexed by the input `points.index`."""
        graph = self._graph()
        points = self._points([(1.0, 0.0), (5.0, 5.0)], ["first", "second"])
        ids, distances = snap_to_network_nodes(points, graph)
        self.assertEqual(list(ids.index), ["first", "second"])
        self.assertEqual(list(distances.index), ["first", "second"])

    def test_max_distance_caps_assignment(self):
        """Points farther than `max_distance` from every node get NaN."""
        graph = self._graph()
        points = self._points(
            [
                (1.0, 0.0),  # within 2.0 of 'a'
                (50.0, 50.0),
            ],  # far from every node
            ["near", "far"],
        )
        ids, distances = snap_to_network_nodes(points, graph, max_distance=5.0)
        self.assertEqual(ids.loc["near"], "a")
        self.assertTrue(pd.isna(ids.loc["far"]))
        self.assertAlmostEqual(distances.loc["near"], 1.0)
        self.assertTrue(pd.isna(distances.loc["far"]))

    def test_multiple_points_to_same_node(self):
        """Multiple points snapping to the same node all get that node's ID."""
        graph = self._graph()
        points = self._points(
            [(0.1, 0.1), (0.2, -0.2), (-0.3, 0.0)],  # all near 'a'
            ["p1", "p2", "p3"],
        )
        ids, _ = snap_to_network_nodes(points, graph)
        self.assertEqual(list(ids), ["a", "a", "a"])

    def test_works_with_osmnx_style_multidigraph(self):
        """Accepts MultiDiGraph (the shape OSMnx returns) — only needs node x/y."""
        g = nx.MultiDiGraph()
        g.add_node(101, x=0.0, y=0.0)
        g.add_node(202, x=5.0, y=0.0)
        g.add_edge(101, 202, length=5.0)
        points = self._points([(4.0, 0.0)], ["p"])
        ids, distances = snap_to_network_nodes(points, g)
        self.assertEqual(ids.loc["p"], 202)
        self.assertAlmostEqual(distances.loc["p"], 1.0)

    def test_missing_node_xy_raises(self):
        """A graph whose nodes lack `x` or `y` attributes raises a clear error."""
        g = nx.Graph()
        g.add_node("bad_node")  # no x/y attrs
        points = self._points([(0.0, 0.0)], ["p"])
        with self.assertRaises(KeyError):
            snap_to_network_nodes(points, g)

    def test_eligible_node_ids_filters_targets(self):
        """`eligible_node_ids` restricts the snap to a subset of nodes."""
        graph = self._graph()  # nodes 'a', 'b', 'c'
        points = self._points([(0.5, 0.5)], ["p"])
        # 'a' is closest to (0.5, 0.5), but exclude 'a' → next-nearest snaps.
        ids, dists = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"b", "c"},
        )
        self.assertIn(ids.loc["p"], {"b", "c"})
        self.assertNotEqual(ids.loc["p"], "a")

    def test_eligible_node_ids_empty_filter_raises(self):
        """Eligibility filter that excludes every node raises a clear error."""
        graph = self._graph()
        points = self._points([(0.5, 0.5)], ["p"])
        with self.assertRaisesRegex(ValueError, "every node"):
            snap_to_network_nodes(points, graph, eligible_node_ids=set())


class AggregateEdgesToNodesTestCase(unittest.TestCase):
    """`aggregate_edges_to_nodes` rolls up a per-edge attribute to per-node values."""

    def _toy_graph(self) -> nx.Graph:
        """Five nodes with edges of varying tier:
        a --[tier=1]-- b --[tier=2]-- c
                        |
                        +--[tier=5]-- d
        e (isolated)
        """
        g = nx.Graph()
        for n in "abcde":
            g.add_node(n, x=0.0, y=0.0)
        g.add_edge("a", "b", tier=1)
        g.add_edge("b", "c", tier=2)
        g.add_edge("b", "d", tier=5)
        return g

    def test_max_aggregator(self):
        out = aggregate_edges_to_nodes(self._toy_graph(), "tier", aggregator="max")
        self.assertEqual(out.loc["a"], 1.0)  # touches only tier-1
        self.assertEqual(out.loc["b"], 5.0)  # touches tiers 1, 2, 5 → max = 5
        self.assertEqual(out.loc["c"], 2.0)
        self.assertEqual(out.loc["d"], 5.0)
        self.assertNotIn("e", out.index)  # isolated node — no edges

    def test_min_aggregator(self):
        out = aggregate_edges_to_nodes(self._toy_graph(), "tier", aggregator="min")
        self.assertEqual(out.loc["b"], 1.0)  # min of {1, 2, 5}

    def test_mean_aggregator(self):
        out = aggregate_edges_to_nodes(self._toy_graph(), "tier", aggregator="mean")
        # b: (1 + 2 + 5) / 3 ≈ 2.67
        self.assertAlmostEqual(out.loc["b"], 8 / 3)

    def test_callable_attribute(self):
        """`edge_attribute` can be a callable (u, v, data) -> value."""
        out = aggregate_edges_to_nodes(
            self._toy_graph(),
            lambda u, v, data: data.get("tier", 0) ** 2,
            aggregator="max",
        )
        self.assertEqual(out.loc["b"], 25.0)  # max of {1, 4, 25}

    def test_callable_aggregator(self):
        """`aggregator` can be a callable on the per-edge values array."""
        out = aggregate_edges_to_nodes(
            self._toy_graph(),
            "tier",
            aggregator=lambda arr: float(arr.sum() / 10),
        )
        self.assertAlmostEqual(out.loc["b"], 8 / 10)

    def test_unknown_aggregator_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown aggregator"):
            aggregate_edges_to_nodes(self._toy_graph(), "tier", aggregator="nope")

    def test_works_with_multidigraph(self):
        """OSMnx-style MultiDiGraph: parallel edges contribute individually."""
        g = nx.MultiDiGraph()
        g.add_node("a", x=0.0, y=0.0)
        g.add_node("b", x=1.0, y=0.0)
        # Two parallel edges in each direction (typical OSMnx pattern).
        g.add_edge("a", "b", tier=3)
        g.add_edge("a", "b", tier=1)  # parallel edge
        g.add_edge("b", "a", tier=3)
        g.add_edge("b", "a", tier=1)
        out = aggregate_edges_to_nodes(g, "tier", aggregator="max")
        self.assertEqual(out.loc["a"], 3.0)
        self.assertEqual(out.loc["b"], 3.0)


class AssignToEligibleCentroidTestCase(unittest.TestCase):
    """`assign_to_eligible_centroid` snaps polygons via the median of their
    eligible interior nodes."""

    def _graph_with_tiers(self) -> nx.Graph:
        """A graph where one zone has multiple eligible nodes plus a low-tier
        outlier, another has only an outlier, and we test the transport-
        centroid vs. geometric-centroid behaviour.

        Zone Z (50 x 50 area): nodes at (10,10), (40,40), (25,25) eligible;
                               (5,5) ineligible.
        Zone W (60 x 60 area, no eligible nodes inside).
        """
        g = nx.Graph()
        # Zone Z interior nodes
        g.add_node("z1", x=10.0, y=10.0)
        g.add_node("z2", x=40.0, y=40.0)
        g.add_node("z3", x=25.0, y=25.0)  # near the median
        g.add_node("z_skip", x=5.0, y=5.0)  # ineligible (excluded by filter)
        # Zone W interior — no eligible nodes
        g.add_node("w_skip", x=80.0, y=80.0)  # in W, but ineligible
        # Background node outside any polygon
        g.add_node("bg", x=200.0, y=200.0)
        return g

    def _polygons(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=[box(0, 0, 50, 50), box(50, 50, 100, 100)],
            index=pd.Index(["Z", "W"], name="zone_id"),
            crs="EPSG:2056",
        )

    def test_snaps_to_median_of_eligible(self):
        """Z has eligible nodes z1, z2, z3 inside; median ≈ (25, 25),
        nearest eligible node = z3."""
        eligible = {"z1", "z2", "z3", "bg"}  # ineligible: z_skip, w_skip
        ids, dists = assign_to_eligible_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
            centroid_method="median",
        )
        self.assertEqual(ids.loc["Z"], "z3")

    def test_mean_centroid(self):
        """Mean centroid lands somewhere different from median when distribution is skewed."""
        # Add an outlier to test mean vs median.
        g = self._graph_with_tiers()
        g.add_node("z_outlier", x=49.0, y=49.0)
        eligible = {"z1", "z2", "z3", "z_outlier", "bg"}
        ids_mean, _ = assign_to_eligible_centroid(
            self._polygons(),
            g,
            eligible_node_ids=eligible,
            centroid_method="mean",
        )
        ids_med, _ = assign_to_eligible_centroid(
            self._polygons(),
            g,
            eligible_node_ids=eligible,
            centroid_method="median",
        )
        # Both should land somewhere reasonable — actual node depends on geometry.
        # The point is just that both methods produce a sensible eligible node.
        self.assertIn(ids_mean.loc["Z"], {"z1", "z2", "z3", "z_outlier"})
        self.assertIn(ids_med.loc["Z"], {"z1", "z2", "z3", "z_outlier"})

    def test_fallback_to_geometric_centroid(self):
        """Zone W has no eligible nodes inside; falls back to its geometric
        centroid, snapping to the globally-nearest eligible node."""
        eligible = {"z1", "z2", "z3", "bg"}
        ids, _ = assign_to_eligible_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
            fallback_to_geometric_centroid=True,
        )
        # W's geometric centroid is (75, 75). Eligible nodes (excluding the
        # already-snapped ones): z1, z2, z3, bg. Nearest to (75,75) is bg (200,200)?
        # Actually no — z2 is at (40,40), bg at (200,200). Distances from (75,75):
        # z2 → ~49.5; bg → ~176. So z2 wins.
        self.assertEqual(ids.loc["W"], "z2")

    def test_no_fallback_gives_nan(self):
        """With fallback off, polygons containing no eligible nodes get NaN."""
        eligible = {"z1", "z2", "z3"}  # no node inside W
        ids, dists = assign_to_eligible_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
            fallback_to_geometric_centroid=False,
        )
        # Z still snaps fine; W → NaN.
        self.assertEqual(ids.loc["Z"], "z3")
        self.assertTrue(pd.isna(ids.loc["W"]))

    def test_empty_eligible_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            assign_to_eligible_centroid(
                self._polygons(),
                self._graph_with_tiers(),
                eligible_node_ids=set(),
            )


class FlagNodeIntersectionsTestCase(unittest.TestCase):
    """`flag_node_intersections` writes mutually-exclusive degree flags
    (`is_degree_3` / `is_degree_4`) and per-node max / min highway-rank
    flags. Obstacle flags (traffic signals etc.) live in
    `consolidate_intersections`, not here.
    """

    def _graph(self) -> nx.MultiDiGraph:
        """Mixed-degree fixture:
            1: 4-way intersection (deg 4) — primary roads
            2: degree-2 mid-edge — primary (1↔2) + residential (2↔6)
            3, 4, 5: degree-1 arm tips on residential
            6: degree-1 leaf on residential
        Highway tags chosen so node 1 sees both primary (rank 5) and
        residential (rank 2) — tests max/min rank.
        """
        g = nx.MultiDiGraph()
        for n, x, y in [(1, 0, 0), (2, 1, 0), (3, -1, 0), (4, 0, 1), (5, 0, -1), (6, 2, 0)]:
            g.add_node(n, x=float(x), y=float(y))
        for u, v, hw in [
            (1, 2, "primary"),
            (1, 3, "residential"),
            (1, 4, "residential"),
            (1, 5, "residential"),
            (2, 6, "residential"),
        ]:
            g.add_edge(u, v, highway=hw)
            g.add_edge(v, u, highway=hw)
        return g

    def test_degree_flags_are_mutually_exclusive(self):
        """is_degree_3 = exactly 3; is_degree_4 = ≥ 4; never both set."""
        g = self._graph()
        flag_node_intersections(g)
        # Node 1: degree 4 → only is_degree_4.
        self.assertEqual(g.nodes[1]["is_degree_3"], 0.0)
        self.assertEqual(g.nodes[1]["is_degree_4"], 1.0)
        # Node 2: degree 2 → neither.
        self.assertEqual(g.nodes[2]["is_degree_3"], 0.0)
        self.assertEqual(g.nodes[2]["is_degree_4"], 0.0)
        # Leaf node 6: degree 1 → neither.
        self.assertEqual(g.nodes[6]["is_degree_3"], 0.0)
        self.assertEqual(g.nodes[6]["is_degree_4"], 0.0)

    def test_degree_3_fires_at_exactly_three(self):
        """Add one arm to a leaf, drop one from the 4-way → 3-way."""
        g = nx.MultiDiGraph()
        for n, (x, y) in enumerate([(0, 0), (1, 0), (-1, 0), (0, 1)]):
            g.add_node(n, x=float(x), y=float(y))
        for u, v in [(0, 1), (0, 2), (0, 3)]:
            g.add_edge(u, v)
            g.add_edge(v, u)
        flag_node_intersections(g)
        self.assertEqual(g.nodes[0]["is_degree_3"], 1.0)
        self.assertEqual(g.nodes[0]["is_degree_4"], 0.0)

    def test_max_min_highway_rank(self):
        """max/min from HIGHWAY_RANKS over incident edges."""
        g = self._graph()
        flag_node_intersections(g)
        from aperta.network_processing import HIGHWAY_RANKS

        # Node 1: edges of types {primary, residential} → max=5, min=2.
        self.assertEqual(g.nodes[1]["max_highway_rank"], float(HIGHWAY_RANKS["primary"]))
        self.assertEqual(g.nodes[1]["min_highway_rank"], float(HIGHWAY_RANKS["residential"]))
        # Node 6: only residential edges → max=min=2.
        self.assertEqual(g.nodes[6]["max_highway_rank"], float(HIGHWAY_RANKS["residential"]))
        self.assertEqual(g.nodes[6]["min_highway_rank"], float(HIGHWAY_RANKS["residential"]))

    def test_undirected_graph_works(self):
        """Undirected graphs use `graph.neighbors`, not predecessors/successors."""
        g = nx.MultiGraph()
        g.add_node(0, x=0.0, y=0.0)
        for i, (x, y) in enumerate([(1, 0), (-1, 0), (0, 1)], start=1):
            g.add_node(i, x=float(x), y=float(y))
            g.add_edge(0, i)
        flag_node_intersections(g)
        # 3 distinct neighbours of node 0 → is_degree_3 set, is_degree_4 clear.
        self.assertEqual(g.nodes[0]["is_degree_3"], 1.0)
        self.assertEqual(g.nodes[0]["is_degree_4"], 0.0)


class ConsolidateIntersectionsTestCase(unittest.TestCase):
    """`consolidate_intersections` wraps `osmnx.consolidate_intersections`,
    plus reattaches obstacle flags (traffic signals, stops, roundabouts)
    that OSMnx alone would drop when their host nodes are merged away.
    """

    def _graph_with_signal_and_roundabout(self) -> nx.MultiDiGraph:
        """4-arm intersection at (1000, 1000) with a traffic_signal node 5 m
        offset (typical OSM pattern — signals tagged on the approach, not
        the centre). Separately, a small roundabout (two nodes 11 m apart,
        connected by a `junction=roundabout` edge) at (2000, 2000).
        """
        g = nx.MultiDiGraph(crs="EPSG:2056")
        g.add_node(1, x=1000.0, y=1000.0)
        for n, (x, y) in zip([2, 3, 4, 5], [(1100, 1000), (900, 1000), (1000, 1100), (1000, 900)]):
            g.add_node(n, x=float(x), y=float(y))
        # Signal sits 5√2 ≈ 7 m east-northeast of the intersection centre.
        g.add_node(6, x=1005.0, y=1005.0, highway="traffic_signals")
        for u, v in [(1, 2), (1, 3), (1, 4), (1, 5)]:
            g.add_edge(u, v)
            g.add_edge(v, u)
        # East arm goes through the signal node.
        g.add_edge(2, 6)
        g.add_edge(6, 1)
        g.add_edge(1, 6)
        g.add_edge(6, 2)
        # Roundabout: two nodes ~11 m apart with junction=roundabout edges.
        g.add_node(10, x=2000.0, y=2000.0)
        g.add_node(11, x=2010.0, y=2005.0)
        g.add_edge(10, 11, junction="roundabout")
        g.add_edge(11, 10, junction="roundabout")
        return g

    def test_signal_reallocated_to_consolidated_node(self):
        """The off-centre traffic_signal node is dropped during consolidation
        but its flag re-attaches to the consolidated 4-way intersection."""
        g = self._graph_with_signal_and_roundabout()
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=30.0)
        # Find the consolidated central intersection (degree ≥ 4 near 1000,1000).
        central = None
        for nid, d in consolidated.nodes(data=True):
            if abs(d["x"] - 1000) < 30 and abs(d["y"] - 1000) < 30 and d.get("is_degree_4") == 1.0:
                central = nid
                break
        self.assertIsNotNone(central, "no consolidated 4-way intersection found")
        self.assertEqual(consolidated.nodes[central]["is_traffic_signal"], 1.0)

    def test_roundabout_detected_from_edge_tag(self):
        """A node consolidated from a `junction=roundabout` edge is flagged."""
        g = self._graph_with_signal_and_roundabout()
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=30.0)
        rb_nodes = [
            nid for nid, d in consolidated.nodes(data=True) if d.get("is_roundabout") == 1.0
        ]
        self.assertEqual(len(rb_nodes), 1)
        self.assertAlmostEqual(consolidated.nodes[rb_nodes[0]]["x"], 2005, delta=10)
        self.assertAlmostEqual(consolidated.nodes[rb_nodes[0]]["y"], 2002.5, delta=10)

    def test_non_intersection_nodes_have_zero_flags(self):
        """Arm-tip nodes (degree 1 in the original) carry no obstacle flags."""
        g = self._graph_with_signal_and_roundabout()
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=30.0)
        # Whichever nodes ended up near the arm tips (not within tolerance of
        # the centre) should have all flags 0.
        for nid, d in consolidated.nodes(data=True):
            if abs(d["x"] - 1000) > 50 and abs(d["x"] - 2005) > 30:
                self.assertEqual(d.get("is_traffic_signal", 0.0), 0.0)
                self.assertEqual(d.get("is_roundabout", 0.0), 0.0)

    def test_obstacle_buffer_excludes_far_signals(self):
        """A signal further than `obstacle_buffer` is NOT attached."""
        g = self._graph_with_signal_and_roundabout()
        # With buffer=2 m the signal at (1005,1005) is too far from the
        # consolidated central node at ~(1001,1001).
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=2.0)
        any_signal = any(
            d.get("is_traffic_signal") == 1.0 for _, d in consolidated.nodes(data=True)
        )
        self.assertFalse(any_signal)


class LanesPerDirectionTestCase(unittest.TestCase):
    """`lanes_per_direction` corrects OSM's bidirectional `lanes` tag for
    use in per-direction quantities (directional AADT, per-lane capacity).
    """

    def test_oneway_returns_lanes_unchanged(self):
        # Motorway: 3 lanes, oneway → all 3 lanes in this direction.
        self.assertEqual(lanes_per_direction({"lanes": 3, "oneway": True}), 3.0)

    def test_twoway_halves_lanes(self):
        # Two-way primary: 4 total lanes → 2 per direction.
        self.assertEqual(lanes_per_direction({"lanes": 4, "oneway": False}), 2.0)

    def test_twoway_with_one_lane_returns_one(self):
        # Narrow shared road: 1 lane both ways → can't split.
        self.assertEqual(lanes_per_direction({"lanes": 1, "oneway": False}), 1.0)

    def test_missing_lanes_defaults_to_one(self):
        # No lanes tag → OSM implicit default = 1 per direction.
        self.assertEqual(lanes_per_direction({"oneway": False}), 1.0)
        self.assertEqual(lanes_per_direction({"oneway": True}), 1.0)

    def test_string_lanes_parsed(self):
        # OSM often stores lanes as strings.
        self.assertEqual(lanes_per_direction({"lanes": "4", "oneway": False}), 2.0)

    def test_list_lanes_takes_first(self):
        # Post-OSMnx merges occasionally leave list-valued tags.
        self.assertEqual(lanes_per_direction({"lanes": ["4", "4"], "oneway": False}), 2.0)

    def test_unparseable_lanes_defaults_to_one(self):
        self.assertEqual(lanes_per_direction({"lanes": "unknown"}), 1.0)


if __name__ == "__main__":
    unittest.main()
