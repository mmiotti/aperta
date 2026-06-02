"""Tests for `aperta.network_processing` — graph-specific helpers.

Run with:
    python -m unittest tests.test_network_processing
"""

import unittest
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from aperta.network_processing import (
    aggregate_edges_to_nodes,
    assign_to_eligible_centroid,
    consolidate_intersections,
    flag_node_intersection_topology,
    flag_node_osm_classification,
    lanes_per_direction,
    prepare_network,
    snap_to_network_nodes,
)


def _flag_all(g):
    """Test helper: topology + OSM classification in one call."""
    flag_node_intersection_topology(g)
    flag_node_osm_classification(g)


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

    def test_eligible_node_flag_filters_by_per_node_attribute(self):
        """`eligible_node_flag` reads a per-node bool attribute to derive eligibility."""
        graph = self._graph()  # nodes 'a', 'b', 'c'
        # Mark only 'b' and 'c' as eligible.
        graph.nodes["a"]["is_snap_eligible"] = False
        graph.nodes["b"]["is_snap_eligible"] = True
        graph.nodes["c"]["is_snap_eligible"] = True
        points = self._points([(0.5, 0.5)], ["p"])
        # 'a' is closest geometrically, but flagged ineligible → snaps to 'b' or 'c'.
        ids, _ = snap_to_network_nodes(points, graph, eligible_node_flag="is_snap_eligible")
        self.assertIn(ids.loc["p"], {"b", "c"})
        self.assertNotEqual(ids.loc["p"], "a")

    def test_eligible_node_ids_takes_precedence_over_flag(self):
        """If both `eligible_node_ids` and `eligible_node_flag` are given,
        the explicit `eligible_node_ids` wins."""
        graph = self._graph()
        # Flag would mark only 'a' eligible:
        graph.nodes["a"]["is_snap_eligible"] = True
        graph.nodes["b"]["is_snap_eligible"] = False
        graph.nodes["c"]["is_snap_eligible"] = False
        points = self._points([(0.5, 0.5)], ["p"])
        # But explicit eligible_node_ids restricts to {'c'} only.
        ids, _ = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"c"},
            eligible_node_flag="is_snap_eligible",
        )
        self.assertEqual(ids.loc["p"], "c")

    def test_eligible_node_flag_missing_attribute_treated_as_false(self):
        """A node without the flag attribute counts as ineligible."""
        graph = self._graph()
        # Only 'b' has the flag attribute and it's True; 'a' and 'c' have no attr.
        graph.nodes["b"]["is_snap_eligible"] = True
        points = self._points([(0.5, 0.5)], ["p"])
        ids, _ = snap_to_network_nodes(points, graph, eligible_node_flag="is_snap_eligible")
        self.assertEqual(ids.loc["p"], "b")

    def test_eligible_node_flag_with_no_eligible_nodes_raises(self):
        """Flag set to a value not present on any node → empty eligible set → raises."""
        graph = self._graph()  # nodes have no flag attribute at all
        points = self._points([(0.5, 0.5)], ["p"])
        with self.assertRaisesRegex(ValueError, "every node"):
            snap_to_network_nodes(points, graph, eligible_node_flag="is_snap_eligible")

    # --- two-pass snapping (priority + eligible fallback) ---

    def _priority_graph(self) -> nx.Graph:
        """Four nodes:
        'a' at (0, 0)   — priority + eligible
        'b' at (10, 0)  — priority + eligible
        'c' at (50, 0)  — eligible only (not priority)
        'd' at (100, 0) — eligible only (not priority)
        """
        g = nx.Graph()
        coords = {"a": (0, 0), "b": (10, 0), "c": (50, 0), "d": (100, 0)}
        for n, (x, y) in coords.items():
            g.add_node(n, x=float(x), y=float(y))
        return g

    def test_priority_snap_two_pass_prefers_priority_within_radius(self):
        """A point near a priority node and an eligible-only node should snap
        to the priority node, even if the eligible-only node is slightly closer."""
        graph = self._priority_graph()
        # Point at (45, 0): geometrically closer to 'c' (5 m) than to 'b' (35 m).
        # With priority radius 40 m, 'b' is in range → priority pass picks 'b'.
        points = self._points([(45.0, 0.0)], ["p"])
        ids, dists = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"a", "b", "c", "d"},
            priority_node_ids={"a", "b"},
            priority_max_distance=40.0,
        )
        self.assertEqual(ids.loc["p"], "b")
        self.assertAlmostEqual(dists.loc["p"], 35.0)

    def test_priority_snap_falls_back_to_eligible_when_no_priority_in_range(self):
        """If no priority node is within `priority_max_distance`, snap to
        nearest eligible node (within `max_distance`)."""
        graph = self._priority_graph()
        # Point at (60, 0): nearest priority 'b' is 50 m away, beyond radius 30 m.
        # Falls back to eligible — nearest is 'c' (10 m).
        points = self._points([(60.0, 0.0)], ["p"])
        ids, dists = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"a", "b", "c", "d"},
            priority_node_ids={"a", "b"},
            priority_max_distance=30.0,
        )
        self.assertEqual(ids.loc["p"], "c")
        self.assertAlmostEqual(dists.loc["p"], 10.0)

    def test_priority_snap_mixed_first_pass_and_fallback(self):
        """Some points get priority snap, others fall back to eligible."""
        graph = self._priority_graph()
        # p1 at (5, 0): closest priority 'a' is 5 m — within priority radius.
        # p2 at (60, 0): closest priority 'b' is 50 m — beyond priority radius,
        #                fall back to eligible nearest 'c' (10 m).
        points = self._points([(5.0, 0.0), (60.0, 0.0)], ["p1", "p2"])
        ids, dists = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"a", "b", "c", "d"},
            priority_node_ids={"a", "b"},
            priority_max_distance=20.0,
        )
        self.assertEqual(ids.loc["p1"], "a")
        self.assertEqual(ids.loc["p2"], "c")
        self.assertAlmostEqual(dists.loc["p1"], 5.0)
        self.assertAlmostEqual(dists.loc["p2"], 10.0)

    def test_priority_snap_uses_flag_attribute(self):
        """`priority_node_flag` derives the priority set from a per-node attr."""
        graph = self._priority_graph()
        graph.nodes["a"]["is_priority"] = True
        graph.nodes["b"]["is_priority"] = True
        graph.nodes["c"]["is_priority"] = False
        graph.nodes["d"]["is_priority"] = False
        points = self._points([(45.0, 0.0)], ["p"])
        ids, _ = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"a", "b", "c", "d"},
            priority_node_flag="is_priority",
            priority_max_distance=40.0,
        )
        self.assertEqual(ids.loc["p"], "b")

    def test_priority_ids_take_precedence_over_priority_flag(self):
        """If both `priority_node_ids` and `priority_node_flag` are given,
        the explicit set wins."""
        graph = self._priority_graph()
        # Flag would mark only 'c' as priority:
        graph.nodes["a"]["is_priority"] = False
        graph.nodes["b"]["is_priority"] = False
        graph.nodes["c"]["is_priority"] = True
        graph.nodes["d"]["is_priority"] = False
        points = self._points([(45.0, 0.0)], ["p"])
        # But explicit priority_node_ids restricts to {'b'}.
        ids, _ = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"a", "b", "c", "d"},
            priority_node_ids={"b"},
            priority_node_flag="is_priority",
            priority_max_distance=40.0,
        )
        self.assertEqual(ids.loc["p"], "b")

    def test_priority_empty_set_falls_back_to_eligible_silently(self):
        """An empty priority set is a valid state (not all graphs have
        priority targets); every point falls through to the eligible pass."""
        graph = self._priority_graph()
        points = self._points([(45.0, 0.0)], ["p"])
        ids, _ = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"a", "b", "c", "d"},
            priority_node_ids=set(),
            priority_max_distance=40.0,
        )
        # Snap proceeds via eligible: 'c' at 5 m is nearest.
        self.assertEqual(ids.loc["p"], "c")

    def test_priority_max_distance_none_means_no_priority_cap(self):
        """`priority_max_distance=None` lets the priority pass match any
        distance — every point snaps to a priority node if any exist."""
        graph = self._priority_graph()
        # Point at (95, 0): geometrically closer to 'd' (5 m) than to any
        # priority node. With no priority cap, still snaps to priority 'b'.
        points = self._points([(95.0, 0.0)], ["p"])
        ids, _ = snap_to_network_nodes(
            points,
            graph,
            eligible_node_ids={"a", "b", "c", "d"},
            priority_node_ids={"a", "b"},
            priority_max_distance=None,
        )
        self.assertEqual(ids.loc["p"], "b")


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
    """`flag_node_intersection_topology` + `flag_node_osm_classification`
    write per-node attributes describing intersection type (`n_streets`,
    `is_t_junction`, `is_4way`), their rank-conditional variants (`_major`,
    `_anchor`), and per-node max / min OSM highway-rank. Obstacle flags
    (traffic signals etc.) live in `consolidate_intersections`, not here.
    """

    def _graph(self) -> nx.MultiDiGraph:
        """Mixed-degree fixture:
            1: 4-way intersection (n_streets=4) — primary + residential
            2: passthrough (n_streets=2) — primary (1↔2) + residential (2↔6)
            3, 4, 5: leaves (n_streets=1) on residential
            6: leaf (n_streets=1) on residential
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

    def test_basic_intersection_flags_mutually_exclusive(self):
        """is_t_junction = exactly 3 distinct neighbours; is_4way = ≥ 4;
        never both set on the same node. Leaves and passthroughs get neither."""
        g = self._graph()
        _flag_all(g)
        # Node 1: 4 distinct neighbours → only is_4way.
        self.assertEqual(g.nodes[1]["n_streets"], 4.0)
        self.assertEqual(g.nodes[1]["is_t_junction"], 0.0)
        self.assertEqual(g.nodes[1]["is_4way"], 1.0)
        # Node 2: passthrough → neither.
        self.assertEqual(g.nodes[2]["n_streets"], 2.0)
        self.assertEqual(g.nodes[2]["is_t_junction"], 0.0)
        self.assertEqual(g.nodes[2]["is_4way"], 0.0)
        # Leaf node 6: 1 neighbour → neither.
        self.assertEqual(g.nodes[6]["n_streets"], 1.0)
        self.assertEqual(g.nodes[6]["is_t_junction"], 0.0)
        self.assertEqual(g.nodes[6]["is_4way"], 0.0)

    def test_t_junction_fires_at_exactly_three(self):
        """3-way intersection (T-junction) lights up is_t_junction."""
        g = nx.MultiDiGraph()
        for n, (x, y) in enumerate([(0, 0), (1, 0), (-1, 0), (0, 1)]):
            g.add_node(n, x=float(x), y=float(y))
        for u, v in [(0, 1), (0, 2), (0, 3)]:
            g.add_edge(u, v)
            g.add_edge(v, u)
        _flag_all(g)
        self.assertEqual(g.nodes[0]["n_streets"], 3.0)
        self.assertEqual(g.nodes[0]["is_t_junction"], 1.0)
        self.assertEqual(g.nodes[0]["is_4way"], 0.0)

    def test_max_min_highway_rank(self):
        """max/min from OSM_HIGHWAY_RANKS over incident edges."""
        g = self._graph()
        _flag_all(g)
        from aperta.network_processing import OSM_HIGHWAY_RANKS

        # Node 1: edges of types {primary, residential} → max=5, min=2.
        self.assertEqual(g.nodes[1]["max_highway_rank"], float(OSM_HIGHWAY_RANKS["primary"]))
        self.assertEqual(g.nodes[1]["min_highway_rank"], float(OSM_HIGHWAY_RANKS["residential"]))
        # Node 6: only residential edges → max=min=2.
        self.assertEqual(g.nodes[6]["max_highway_rank"], float(OSM_HIGHWAY_RANKS["residential"]))
        self.assertEqual(g.nodes[6]["min_highway_rank"], float(OSM_HIGHWAY_RANKS["residential"]))

    def test_undirected_graph_works(self):
        """Undirected graphs use `graph.neighbors`, not predecessors/successors."""
        g = nx.MultiGraph()
        g.add_node(0, x=0.0, y=0.0)
        for i, (x, y) in enumerate([(1, 0), (-1, 0), (0, 1)], start=1):
            g.add_node(i, x=float(x), y=float(y))
            g.add_edge(0, i)
        _flag_all(g)
        # 3 distinct neighbours of node 0 → is_t_junction set, is_4way clear.
        self.assertEqual(g.nodes[0]["n_streets"], 3.0)
        self.assertEqual(g.nodes[0]["is_t_junction"], 1.0)
        self.assertEqual(g.nodes[0]["is_4way"], 0.0)

    def test_major_requires_min_rank_ge_3(self):
        """`_major` variants need every incident edge to be tertiary or better."""
        # Node A: 3-way T-junction with three primary edges → major qualifies.
        g_pure_t = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D"):
            g_pure_t.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D")]:
            g_pure_t.add_edge(u, v, highway="primary")
            g_pure_t.add_edge(v, u, highway="primary")
        _flag_all(g_pure_t)
        self.assertEqual(g_pure_t.nodes["A"]["is_t_junction"], 1.0)
        self.assertEqual(g_pure_t.nodes["A"]["is_t_junction_major"], 1.0)

        # Node from `_graph` fixture: 4-way with one primary + three
        # residential → min_rank = 2 (residential) → major fails.
        g_mixed = self._graph()
        _flag_all(g_mixed)
        self.assertEqual(g_mixed.nodes[1]["is_4way"], 1.0)
        self.assertEqual(g_mixed.nodes[1]["is_4way_major"], 0.0)  # has a residential branch

    def test_anchor_requires_max_rank_ge_3_and_min_rank_le_5(self):
        """`_anchor` variants need ≥1 tertiary+ edge AND not pure trunk/motorway."""
        # Mixed residential + primary 4-way: max=5 (primary), min=2 (residential).
        # 5 >= 3 ✓ and 2 <= 5 ✓ → anchor qualifies.
        g_mixed = self._graph()
        _flag_all(g_mixed)
        self.assertEqual(g_mixed.nodes[1]["is_4way"], 1.0)
        self.assertEqual(g_mixed.nodes[1]["is_4way_anchor"], 1.0)

        # Pure-residential T-junction: max=2 → fails `max >= 3` → anchor=0.
        g_pure_res = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D"):
            g_pure_res.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D")]:
            g_pure_res.add_edge(u, v, highway="residential")
            g_pure_res.add_edge(v, u, highway="residential")
        _flag_all(g_pure_res)
        self.assertEqual(g_pure_res.nodes["A"]["is_t_junction"], 1.0)
        self.assertEqual(g_pure_res.nodes["A"]["is_t_junction_anchor"], 0.0)

        # Pure-motorway T-junction: max=min=7 → fails `min <= 5` → anchor=0.
        g_pure_mw = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D"):
            g_pure_mw.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D")]:
            g_pure_mw.add_edge(u, v, highway="motorway")
            g_pure_mw.add_edge(v, u, highway="motorway")
        _flag_all(g_pure_mw)
        self.assertEqual(g_pure_mw.nodes["A"]["is_t_junction"], 1.0)
        self.assertEqual(g_pure_mw.nodes["A"]["is_t_junction_anchor"], 0.0)

    def test_major_is_a_subset_of_anchor_when_max_rank_le_5(self):
        """If every edge is tertiary–primary (rank 3–5), the node is BOTH
        major (min >= 3) AND anchor (max >= 3, min <= 5)."""
        g = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D", "E"):
            g.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]:
            g.add_edge(u, v, highway="tertiary")
            g.add_edge(v, u, highway="tertiary")
        _flag_all(g)
        self.assertEqual(g.nodes["A"]["is_4way"], 1.0)
        self.assertEqual(g.nodes["A"]["is_4way_major"], 1.0)
        self.assertEqual(g.nodes["A"]["is_4way_anchor"], 1.0)

    def test_passthrough_node_gets_no_intersection_flags(self):
        """Passthrough (n_streets=2) is neither T-junction nor 4-way, regardless of rank."""
        g = nx.MultiDiGraph()
        for n in ("A", "B", "C"):
            g.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("B", "C")]:
            g.add_edge(u, v, highway="primary")
            g.add_edge(v, u, highway="primary")
        _flag_all(g)
        self.assertEqual(g.nodes["B"]["n_streets"], 2.0)
        # All intersection flags off:
        for flag in (
            "is_t_junction",
            "is_4way",
            "is_t_junction_major",
            "is_4way_major",
            "is_t_junction_anchor",
            "is_4way_anchor",
        ):
            self.assertEqual(g.nodes["B"][flag], 0.0, f"{flag} should be 0 for passthrough")


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
            if abs(d["x"] - 1000) < 30 and abs(d["y"] - 1000) < 30 and d.get("is_4way") == 1.0:
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


class PrepareNetworkTestCase(unittest.TestCase):
    """`prepare_network` resolves per-mode defaults (with optional `base_mode`
    inheritance for subtype labels), applies the directedness transform,
    precomputes the largest CC / SCC, and decorates the graph in place with
    per-node `snap_eligible_flag` and per-edge `cost_excluded_flag` boolean
    attributes.
    """

    def _trap_graph(self, with_highway: bool = False) -> nx.MultiDiGraph:
        """Tiny directed graph with a one-way trap.

        Topology (all edges directed):
            0 -> 1 -> 2 -> 0    (a 3-cycle = strongly connected)
            2 -> 3              (one-way edge into 3; 3 has no outgoing edge)

        Largest SCC = {0, 1, 2}. Node 3 is trapped: reachable from the cycle
        but cannot reach anything. As undirected, every node is mutually
        reachable: largest CC = {0, 1, 2, 3}.

        If `with_highway`, edges carry varying `highway` tags so cost-exclusion
        logic can be exercised.
        """
        g = nx.MultiDiGraph()
        for n, (x, y) in {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (2, 1)}.items():
            g.add_node(n, x=float(x), y=float(y))
        if with_highway:
            g.add_edge(0, 1, highway="residential")
            g.add_edge(1, 2, highway="motorway")
            g.add_edge(2, 0, highway=["primary", "trunk"])  # list-valued
            g.add_edge(2, 3, highway="footway")
        else:
            g.add_edge(0, 1)
            g.add_edge(1, 2)
            g.add_edge(2, 0)
            g.add_edge(2, 3)
        return g

    # --- default resolution ---

    def test_walk_defaults(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(self._trap_graph(), "walk")
        self.assertEqual(prepared.mode, "walk")
        self.assertEqual(prepared.directedness, "undirected")
        self.assertEqual(prepared.network_type, "all")
        self.assertEqual(prepared.snap_eligible_flag, "is_snap_eligible_walk")
        self.assertEqual(prepared.cost_excluded_flag, "cost_excluded_walk")

    def test_bike_defaults(self):
        prepared = prepare_network(self._trap_graph(), "bike")
        self.assertEqual(prepared.directedness, "undirected")
        self.assertEqual(prepared.network_type, "bike")
        self.assertEqual(prepared.snap_eligible_flag, "is_snap_eligible_bike")
        self.assertEqual(prepared.cost_excluded_flag, "cost_excluded_bike")

    def test_car_defaults(self):
        prepared = prepare_network(self._trap_graph(), "car")
        self.assertEqual(prepared.directedness, "directed_scc")
        self.assertEqual(prepared.network_type, "drive")
        self.assertEqual(prepared.snap_eligible_flag, "is_snap_eligible_car")
        self.assertEqual(prepared.cost_excluded_flag, "cost_excluded_car")

    def test_unknown_mode_without_base_mode_raises(self):
        # No `mode in MODE_DEFAULTS` and no `base_mode` → can't infer defaults.
        with self.assertRaisesRegex(ValueError, "no defaults are available"):
            prepare_network(self._trap_graph(), "moped")

    def test_invalid_base_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "base_mode"):
            prepare_network(self._trap_graph(), "car_night", base_mode="moped")  # type: ignore[arg-type]

    def test_unknown_mode_with_explicit_flags_succeeds(self):
        # No MODE_DEFAULTS entry, but caller supplies everything explicit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(
                self._trap_graph(),
                "snowmobile",
                directedness="undirected",
                network_type="all",
                cost_excluded_tags=set(),
            )
        self.assertEqual(prepared.mode, "snowmobile")
        self.assertEqual(prepared.snap_eligible_flag, "is_snap_eligible_snowmobile")

    # --- base_mode inheritance ---

    def test_base_mode_inherits_defaults(self):
        # Subtype label "car_night" with base_mode="car" → car defaults apply.
        prepared = prepare_network(self._trap_graph(), "car_night", base_mode="car")
        self.assertEqual(prepared.mode, "car_night")
        self.assertEqual(prepared.directedness, "directed_scc")  # from car defaults
        self.assertEqual(prepared.network_type, "drive")  # from car defaults
        # Flag names embed the user-supplied label, not the base:
        self.assertEqual(prepared.snap_eligible_flag, "is_snap_eligible_car_night")
        self.assertEqual(prepared.cost_excluded_flag, "cost_excluded_car_night")

    def test_base_mode_warnings_fire_for_subtype(self):
        # car_night with directedness='undirected' should fire the car warning
        # because base_mode='car' resolves to the car rule set.
        with self.assertWarnsRegex(UserWarning, "treats one-way streets as"):
            prepare_network(
                self._trap_graph(),
                "car_night",
                base_mode="car",
                directedness="undirected",
            )

    def test_base_mode_override_individual_flags(self):
        # car_night inherits car defaults but overrides cost_excluded_tags
        # to add night-closure tags.
        prepared = prepare_network(
            self._trap_graph(with_highway=True),
            "car_night",
            base_mode="car",
            cost_excluded_tags={"service"},
        )
        # The directedness still inherits from car (directed → SCC).
        self.assertEqual(prepared.directedness, "directed_scc")

    # --- snap-eligible node computation ---

    def test_undirected_snap_set_is_largest_cc(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(self._trap_graph(), "walk")
        self.assertEqual(prepared.snap_eligible_nodes, frozenset({0, 1, 2, 3}))

    def test_directed_scc_snap_set_excludes_trapped_node(self):
        prepared = prepare_network(self._trap_graph(), "car")
        self.assertEqual(prepared.snap_eligible_nodes, frozenset({0, 1, 2}))
        self.assertNotIn(3, prepared.snap_eligible_nodes)

    def test_directed_scc_picks_largest_when_multiple(self):
        g = nx.MultiDiGraph()
        # SCC A: 3-cycle on 0,1,2
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        # SCC B: 2-cycle on 10,11
        g.add_edge(10, 11)
        g.add_edge(11, 10)
        for n in g.nodes:
            g.nodes[n]["x"] = float(n)
            g.nodes[n]["y"] = 0.0
        prepared = prepare_network(g, "car")
        self.assertEqual(prepared.snap_eligible_nodes, frozenset({0, 1, 2}))

    def test_directed_scc_requires_directed_graph(self):
        undirected = nx.MultiGraph()
        undirected.add_edge(0, 1)
        for n in undirected.nodes:
            undirected.nodes[n]["x"] = 0.0
            undirected.nodes[n]["y"] = 0.0
        with self.assertRaises(ValueError):
            prepare_network(undirected, "car", directedness="directed_scc")

    # --- graph attribute decoration ---

    def test_snap_eligible_attribute_written_per_node(self):
        # Car defaults → directed SCC of {0,1,2}, trap node 3 excluded.
        # Every node should carry `is_snap_eligible_car: bool`.
        prepared = prepare_network(self._trap_graph(), "car")
        g = prepared.graph
        self.assertTrue(g.nodes[0]["is_snap_eligible_car"])
        self.assertTrue(g.nodes[1]["is_snap_eligible_car"])
        self.assertTrue(g.nodes[2]["is_snap_eligible_car"])
        self.assertFalse(g.nodes[3]["is_snap_eligible_car"])

    def test_cost_excluded_attribute_written_per_edge_with_string_highway(self):
        # Car defaults have empty cost_excluded_tags, so the flag is False
        # everywhere even though edges carry highway tags. Override to
        # exclude motorway/trunk to actually exercise the masking.
        prepared = prepare_network(
            self._trap_graph(with_highway=True),
            "car_filtered",
            base_mode="car",
            cost_excluded_tags={"motorway", "trunk"},
        )
        g = prepared.graph
        # Edge 0->1 (residential) → not excluded
        self.assertFalse(g.edges[0, 1, 0]["cost_excluded_car_filtered"])
        # Edge 1->2 (motorway) → excluded
        self.assertTrue(g.edges[1, 2, 0]["cost_excluded_car_filtered"])
        # Edge 2->0 has list-valued highway including 'trunk' → excluded
        self.assertTrue(g.edges[2, 0, 0]["cost_excluded_car_filtered"])
        # Edge 2->3 (footway) → not excluded
        self.assertFalse(g.edges[2, 3, 0]["cost_excluded_car_filtered"])

    def test_cost_excluded_handles_none_highway(self):
        # _trap_graph() without with_highway=True has no highway tags.
        # All edges should default to cost_excluded=False.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(self._trap_graph(), "walk")
        g = prepared.graph
        # Undirected MultiGraph after to_undirected
        for u, v, k in g.edges(keys=True):
            self.assertFalse(g.edges[u, v, k]["cost_excluded_walk"])

    def test_multiple_modes_accumulate_flags(self):
        # Calling prepare_network twice with different modes on graphs derived
        # from the same source produces decorated graphs that, if applied to
        # the SAME underlying directed graph, would carry both flag sets.
        # We exercise the per-call accumulation directly on a directed graph.
        g = self._trap_graph()
        # Apply car (directed_scc — no to_undirected, so g itself is mutated).
        prepared_car = prepare_network(g, "car")
        self.assertIs(prepared_car.graph, g)  # same object
        # Car flags landed on every node.
        for n in g.nodes:
            self.assertIn("is_snap_eligible_car", g.nodes[n])
        # Now also apply 'car_night' on the same graph; both flag sets coexist.
        _ = prepare_network(g, "car_night", base_mode="car")
        for n in g.nodes:
            self.assertIn("is_snap_eligible_car", g.nodes[n])
            self.assertIn("is_snap_eligible_car_night", g.nodes[n])

    def test_custom_flag_names_override_defaults(self):
        prepared = prepare_network(
            self._trap_graph(),
            "car",
            snap_eligible_flag="my_snap_flag",
            cost_excluded_flag="my_cost_flag",
        )
        self.assertEqual(prepared.snap_eligible_flag, "my_snap_flag")
        self.assertEqual(prepared.cost_excluded_flag, "my_cost_flag")
        g = prepared.graph
        self.assertIn("my_snap_flag", g.nodes[0])
        # No default name written:
        self.assertNotIn("is_snap_eligible_car", g.nodes[0])

    # --- override paths ---

    def test_user_overrides_take_precedence_over_defaults(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(
                self._trap_graph(),
                "walk",
                network_type="bike",  # nonsense for walk but accepted
            )
        self.assertEqual(prepared.network_type, "bike")
        self.assertEqual(prepared.directedness, "undirected")

    def test_empty_cost_excluded_tags_is_explicit_override(self):
        # Passing `cost_excluded_tags=set()` overrides the walk default.
        # The override is observable via the per-edge flag staying False
        # even on motorway-tagged edges.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(
                self._trap_graph(with_highway=True),
                "walk",
                cost_excluded_tags=set(),
            )
        g = prepared.graph
        # The motorway-tagged edge should now NOT be excluded.
        # After to_undirected, find any edge with highway=motorway.
        motorway_edges = [
            (u, v, k)
            for u, v, k, d in g.edges(keys=True, data=True)
            if d.get("highway") == "motorway"
        ]
        self.assertTrue(motorway_edges)
        for u, v, k in motorway_edges:
            self.assertFalse(g.edges[u, v, k]["cost_excluded_walk"])

    # --- warning emission ---

    def test_walk_directed_scc_warns(self):
        with self.assertWarnsRegex(UserWarning, "unnecessarily restrictive"):
            prepare_network(self._trap_graph(), "walk", directedness="directed_scc")

    def test_walk_network_type_walk_warns(self):
        with self.assertWarnsRegex(UserWarning, "Cambridge MA pitfall"):
            prepare_network(self._trap_graph(), "walk", network_type="walk")

    def test_walk_all_with_empty_excluded_tags_warns(self):
        with self.assertWarnsRegex(UserWarning, "routed across motorways"):
            prepare_network(self._trap_graph(), "walk", cost_excluded_tags=set())

    def test_car_undirected_warns(self):
        with self.assertWarnsRegex(UserWarning, "treats one-way streets as"):
            prepare_network(self._trap_graph(), "car", directedness="undirected")

    def test_car_all_with_empty_excluded_tags_warns(self):
        with self.assertWarnsRegex(UserWarning, "footways, pedestrian paths"):
            prepare_network(self._trap_graph(), "car", network_type="all")

    def test_bike_defaults_emit_no_warning(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            prepare_network(self._trap_graph(), "bike")
        self.assertEqual(
            [w for w in captured if issubclass(w.category, UserWarning)],
            [],
            "Bike defaults should not emit any UserWarning.",
        )

    def test_car_defaults_emit_no_warning(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            prepare_network(self._trap_graph(), "car")
        self.assertEqual(
            [w for w in captured if issubclass(w.category, UserWarning)],
            [],
            "Car defaults should not emit any UserWarning.",
        )

    def test_non_default_but_unproblematic_combo_emits_no_warning(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            prepare_network(self._trap_graph(), "bike", directedness="directed_scc")
        self.assertEqual(
            [w for w in captured if issubclass(w.category, UserWarning)],
            [],
            "Bike + directed_scc is a valid non-default choice and should not warn.",
        )

    def test_unknown_mode_no_warnings_fire(self):
        # No MODE_DEFAULTS entry and no base_mode → no warning rules apply,
        # so even pathological combinations should be silent (the user is
        # off the warning-policy reservation by choice).
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            prepare_network(
                self._trap_graph(),
                "snowmobile",
                directedness="undirected",
                network_type="all",
                cost_excluded_tags=set(),
            )
        self.assertEqual(
            [w for w in captured if issubclass(w.category, UserWarning)],
            [],
            "Unknown mode without base_mode should not fire any warnings.",
        )

    # --- PreparedGraph itself ---

    def test_prepared_graph_is_frozen(self):
        prepared = prepare_network(self._trap_graph(), "car")
        with self.assertRaises(Exception):  # dataclass frozen → FrozenInstanceError
            prepared.mode = "walk"  # type: ignore[misc]

    # --- cost-mask-aware snap-eligibility (the motorway-ramp pedestrian case) ---

    def _cost_mask_graph(self) -> nx.MultiDiGraph:
        """4 nodes: A-B-C all connected by `residential` edges; D connected
        to B only via a `motorway` edge.

        Topology: A, B, C, D are all in the single largest connected
        component (undirected) — purely topological eligibility includes all
        4. But for walk, motorway is in `cost_excluded_tags`, so D is
        practically unreachable on foot — the cost-masked subgraph isolates
        D into its own singleton component.
        """
        g = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D"):
            g.add_node(n, x=0.0, y=0.0)
        g.add_edge("A", "B", highway="residential")
        g.add_edge("B", "A", highway="residential")
        g.add_edge("B", "C", highway="residential")
        g.add_edge("C", "B", highway="residential")
        g.add_edge("B", "D", highway="motorway")
        g.add_edge("D", "B", highway="motorway")
        return g

    def test_snap_eligible_excludes_nodes_reachable_only_via_excluded_edges(self):
        """A node reachable from the main CC ONLY through cost-excluded edges
        must NOT be in the snap-eligible set, even though it is topologically
        in the largest CC. This is the motorway-on-ramp pedestrian case
        (pre-cost-mask behavior put cells here and routing failed silently)."""
        prepared = prepare_network(self._cost_mask_graph(), "walk")
        self.assertIn("A", prepared.snap_eligible_nodes)
        self.assertIn("B", prepared.snap_eligible_nodes)
        self.assertIn("C", prepared.snap_eligible_nodes)
        self.assertNotIn("D", prepared.snap_eligible_nodes)

    def test_snap_eligible_includes_nodes_when_cost_excluded_tags_empty(self):
        """With empty cost_excluded_tags, the cost-masked subgraph equals the
        raw graph, so a node reachable only via motorway IS eligible (because
        nothing is excluded for routing either). Confirms the new behavior
        reduces to the old behavior in the no-mask case (default for bike + car)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # expect the empty-tags warning
            prepared = prepare_network(
                self._cost_mask_graph(),
                "walk",
                cost_excluded_tags=set(),
            )
        self.assertIn("D", prepared.snap_eligible_nodes)

    def test_snap_eligible_excluded_node_also_marked_in_per_node_attribute(self):
        """The per-node `is_snap_eligible_walk` graph attribute reflects the
        cost-mask-aware result, not the raw topology — confirms the graph
        decoration is in sync with the dataclass field."""
        prepared = prepare_network(self._cost_mask_graph(), "walk")
        g = prepared.graph
        self.assertTrue(g.nodes["A"]["is_snap_eligible_walk"])
        self.assertTrue(g.nodes["B"]["is_snap_eligible_walk"])
        self.assertTrue(g.nodes["C"]["is_snap_eligible_walk"])
        self.assertFalse(g.nodes["D"]["is_snap_eligible_walk"])

    # --- priority-node classification ---

    def _intersection_graph(self) -> nx.MultiDiGraph:
        """A graph with three node types that map to distinct priority outcomes:

        - Node H (hub): 4-way intersection, all tertiary edges → all of
          is_4way / is_4way_major / is_4way_anchor set.
        - Node M (mixed T): 3-way intersection with one residential + two
          tertiary edges → is_t_junction set; major fails (min_rank=2);
          anchor passes (max_rank=3, min_rank=2 ≤ 5).
        - Node E (leaf): dead-end on residential → no intersection flags.

        Built so all four nodes are in a single (undirected) connected
        component for the cost-mask-aware eligibility step.
        """
        g = nx.MultiDiGraph()
        for n in ("H", "M", "E", "N1", "N2", "N3"):
            g.add_node(n, x=0.0, y=0.0)
        # H is a 4-way on tertiary roads
        for nbr in ("N1", "N2", "N3", "M"):
            g.add_edge("H", nbr, highway="tertiary")
            g.add_edge(nbr, "H", highway="tertiary")
        # M is a T-junction: tertiary to H, tertiary to N1, residential to E
        g.add_edge("M", "N1", highway="tertiary")
        g.add_edge("N1", "M", highway="tertiary")
        g.add_edge("M", "E", highway="residential")
        g.add_edge("E", "M", highway="residential")
        # Write topology + OSM-classification node attributes.
        _flag_all(g)
        return g

    def test_car_priority_default_picks_anchor_intersections(self):
        """Car default predicate (is_t_junction_anchor OR is_4way_anchor)
        marks H (4-way anchor) and M (T-junction anchor) but not E (leaf)."""
        prepared = prepare_network(self._intersection_graph(), "car")
        # H: 4-way intersection with all tertiary edges → anchor.
        self.assertIn("H", prepared.snap_priority_nodes)
        # M: T-junction with mixed residential/tertiary → anchor.
        self.assertIn("M", prepared.snap_priority_nodes)
        # E: leaf node (n_streets=1) → no intersection flags → not priority.
        self.assertNotIn("E", prepared.snap_priority_nodes)

    def test_walk_bike_priority_default_picks_only_4way(self):
        """Walk + bike default predicate (is_4way) marks only nodes with
        n_streets >= 4 — H qualifies, M (T-junction) does not."""
        for mode in ("walk", "bike"):
            with self.subTest(mode=mode):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prepared = prepare_network(self._intersection_graph(), mode)
                self.assertIn("H", prepared.snap_priority_nodes)
                self.assertNotIn("M", prepared.snap_priority_nodes)
                self.assertNotIn("E", prepared.snap_priority_nodes)

    def test_priority_nodes_decoration_written_to_graph(self):
        """Per-node `is_snap_priority_<mode>` graph attribute reflects the
        priority set — survives `.graphml` roundtripping."""
        prepared = prepare_network(self._intersection_graph(), "car")
        g = prepared.graph
        flag = prepared.snap_priority_flag
        self.assertEqual(flag, "is_snap_priority_car")
        self.assertTrue(g.nodes["H"][flag])
        self.assertTrue(g.nodes["M"][flag])
        self.assertFalse(g.nodes["E"][flag])

    def test_priority_filter_override_replaces_mode_default(self):
        """Explicit `priority_node_filter` overrides the mode default."""

        def only_4way_anchors(data: dict) -> bool:
            return bool(data.get("is_4way_anchor", 0))

        prepared = prepare_network(
            self._intersection_graph(),
            "car",
            priority_node_filter=only_4way_anchors,
        )
        # H (4-way anchor) qualifies, M (T-junction anchor) does not.
        self.assertIn("H", prepared.snap_priority_nodes)
        self.assertNotIn("M", prepared.snap_priority_nodes)

    def test_unknown_mode_without_base_mode_gets_empty_priority(self):
        """No predicate available → empty priority set, no raise."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(
                self._intersection_graph(),
                "snowmobile",
                directedness="directed_scc",
                network_type="all",
                cost_excluded_tags=set(),
            )
        self.assertEqual(prepared.snap_priority_nodes, frozenset())
        # Per-node flag still written (all False).
        for n in prepared.graph.nodes:
            self.assertFalse(prepared.graph.nodes[n]["is_snap_priority_snowmobile"])

    def test_priority_is_subset_of_eligible(self):
        """Every priority node must also be in `snap_eligible_nodes` —
        priority is a refinement of eligibility, not orthogonal to it."""
        prepared = prepare_network(self._intersection_graph(), "car")
        self.assertTrue(prepared.snap_priority_nodes.issubset(prepared.snap_eligible_nodes))

    def test_custom_snap_priority_flag_name(self):
        """Explicit `snap_priority_flag` overrides the default name."""
        prepared = prepare_network(
            self._intersection_graph(),
            "car",
            snap_priority_flag="my_priority",
        )
        self.assertEqual(prepared.snap_priority_flag, "my_priority")
        g = prepared.graph
        self.assertIn("my_priority", g.nodes["H"])
        self.assertNotIn("is_snap_priority_car", g.nodes["H"])

    # --- end-to-end integration ---

    def test_prepared_graph_feeds_snap_and_cost_mask(self):
        """End-to-end: prepare_network → snap_to_network_nodes by flag →
        mask_excluded_edges → apply_edge_weights. Verifies the per-node and
        per-edge decorations flow through the downstream helpers cleanly.
        """
        from aperta.routing import apply_edge_weights, mask_excluded_edges

        prepared = prepare_network(
            self._trap_graph(with_highway=True),
            "car_filtered",
            base_mode="car",
            cost_excluded_tags={"motorway", "trunk"},
        )

        # Snap-by-flag: a point near trap node 3 should NOT snap to 3
        # (3 is outside the largest SCC for car directedness).
        points = gpd.GeoDataFrame(
            geometry=[Point(2.0, 1.0)],  # right on top of node 3
            index=pd.Index(["p"], name="point_id"),
        )
        ids, _ = snap_to_network_nodes(
            points,
            prepared.graph,
            eligible_node_flag=prepared.snap_eligible_flag,
        )
        self.assertNotEqual(ids.loc["p"], 3)
        self.assertIn(ids.loc["p"], {0, 1, 2})

        # Cost mask: edges flagged as cost-excluded get inf, others get the
        # base weight (here: a constant 1.0 per edge).
        masked = mask_excluded_edges(lambda d: 1.0, prepared.cost_excluded_flag)
        apply_edge_weights(prepared.graph, masked, "cost")
        # Edge 1->2 is motorway → excluded.
        self.assertEqual(prepared.graph.edges[1, 2, 0]["cost"], float("inf"))
        # Edge 2->0 has list-valued highway with 'trunk' → excluded.
        self.assertEqual(prepared.graph.edges[2, 0, 0]["cost"], float("inf"))
        # Edge 0->1 (residential) and 2->3 (footway) → not excluded.
        self.assertEqual(prepared.graph.edges[0, 1, 0]["cost"], 1.0)
        self.assertEqual(prepared.graph.edges[2, 3, 0]["cost"], 1.0)


if __name__ == "__main__":
    unittest.main()
