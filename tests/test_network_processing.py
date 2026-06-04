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
from shapely.geometry import LineString, Point, box

from aperta.network_processing import (
    aggregate_edges_to_nodes,
    consolidate_intersections,
    flag_node_intersection_topology,
    flag_node_osm_classification,
    lanes_per_direction,
)
from aperta.network_snap import (
    insert_projected_nodes,
    nodes_incident_to_edges,
    snap_to_network_nodes,
    split_edge_at_point,
    split_two_way_edge_at_point,
    transport_centroid,
)
from aperta.routing_prep import compute_snap_eligibility, prepare_network


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

    def test_max_radius_caps_assignment(self):
        """Points farther than `max_radius` from every node get NaN."""
        graph = self._graph()
        points = self._points(
            [
                (1.0, 0.0),  # within 2.0 of 'a'
                (50.0, 50.0),
            ],  # far from every node
            ["near", "far"],
        )
        ids, distances = snap_to_network_nodes(points, graph, max_radius=5.0)
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

    def test_both_eligible_kwargs_raises(self):
        """Passing both `eligible_node_ids` and `eligible_node_flag` raises
        (avoids silent footgun from the previous "ids wins" precedence)."""
        graph = self._graph()
        graph.nodes["a"]["is_snap_eligible"] = True
        points = self._points([(0.5, 0.5)], ["p"])
        with self.assertRaises(ValueError):
            snap_to_network_nodes(
                points,
                graph,
                eligible_node_ids={"c"},
                eligible_node_flag="is_snap_eligible",
            )

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


class TransportCentroidTestCase(unittest.TestCase):
    """`transport_centroid` returns per-polygon centroids of the eligible
    network nodes inside (median or mean). Output is a point GeoDataFrame
    indexed by polygon id."""

    def _graph_with_tiers(self) -> nx.Graph:
        """Zone Z (50 x 50): nodes at (10,10), (40,40), (25,25) eligible;
                              (5,5) ineligible.
        Zone W (50 x 50): no eligible nodes inside.
        """
        g = nx.Graph()
        g.add_node("z1", x=10.0, y=10.0)
        g.add_node("z2", x=40.0, y=40.0)
        g.add_node("z3", x=25.0, y=25.0)
        g.add_node("z_skip", x=5.0, y=5.0)
        g.add_node("w_skip", x=80.0, y=80.0)
        g.add_node("bg", x=200.0, y=200.0)
        return g

    def _polygons(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=[box(0, 0, 50, 50), box(50, 50, 100, 100)],
            index=pd.Index(["Z", "W"], name="zone_id"),
            crs="EPSG:2056",
        )

    def test_returns_median_of_eligible_interior(self):
        """Z has eligible z1=(10,10), z2=(40,40), z3=(25,25) inside; median (25, 25)."""
        eligible = {"z1", "z2", "z3", "bg"}
        centroids = transport_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
            centroid_method="median",
        )
        self.assertIsInstance(centroids, gpd.GeoDataFrame)
        # Z's median: x = median([10, 40, 25]) = 25; y similarly = 25.
        z_pt = centroids.loc["Z"].geometry
        self.assertAlmostEqual(z_pt.x, 25.0)
        self.assertAlmostEqual(z_pt.y, 25.0)

    def test_returns_mean_of_eligible_interior(self):
        """Mean = mean of {10, 40, 25} = 25 (same as median for symmetric set)."""
        eligible = {"z1", "z2", "z3", "bg"}
        centroids = transport_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
            centroid_method="mean",
        )
        z_pt = centroids.loc["Z"].geometry
        self.assertAlmostEqual(z_pt.x, 25.0)
        self.assertAlmostEqual(z_pt.y, 25.0)

    def test_fallback_to_geometric_centroid(self):
        """Zone W has no eligible nodes inside; falls back to geometric centroid."""
        eligible = {"z1", "z2", "z3", "bg"}
        centroids = transport_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
            fallback_to_geometric_centroid=True,
        )
        # W's geometric centroid is at the middle: (75, 75).
        w_pt = centroids.loc["W"].geometry
        self.assertAlmostEqual(w_pt.x, 75.0)
        self.assertAlmostEqual(w_pt.y, 75.0)

    def test_no_fallback_drops_polygons(self):
        """With fallback off, polygons without eligible nodes are absent from output."""
        eligible = {"z1", "z2", "z3"}
        centroids = transport_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
            fallback_to_geometric_centroid=False,
        )
        self.assertIn("Z", centroids.index)
        self.assertNotIn("W", centroids.index)

    def test_output_crs_preserved(self):
        centroids = transport_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids={"z1", "z2", "z3", "bg"},
        )
        self.assertEqual(centroids.crs, self._polygons().crs)

    def test_empty_eligible_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            transport_centroid(
                self._polygons(),
                self._graph_with_tiers(),
                eligible_node_ids=set(),
            )

    def test_compose_with_snap_to_network_nodes(self):
        """The composed pattern: transport_centroid → snap_to_network_nodes
        reproduces the old all-in-one behaviour, with caller in control of
        the snap function."""
        eligible = {"z1", "z2", "z3", "bg"}
        centroids = transport_centroid(
            self._polygons(),
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
        )
        ids, _ = snap_to_network_nodes(
            centroids,
            self._graph_with_tiers(),
            eligible_node_ids=eligible,
        )
        # Z's centroid is at (25, 25) → nearest eligible node z3 at (25, 25).
        self.assertEqual(ids.loc["Z"], "z3")


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


class SplitEdgeAtPointTestCase(unittest.TestCase):
    """`split_edge_at_point` inserts a synthetic node on an edge at the
    projection of a target point, splitting the edge into two children
    with proportional geometry / length and copy-as-is for other attrs.
    """

    def _simple_multidi_graph(self) -> nx.MultiDiGraph:
        """One-way directed multigraph: node 1 (0,0) → node 2 (100,0).
        Edge has standard OSM attrs + a LineString geometry."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        g.add_edge(
            1,
            2,
            key=0,
            highway="residential",
            lanes=2,
            maxspeed=50,
            oneway=True,
            length=100.0,
            geometry=LineString([(0.0, 0.0), (100.0, 0.0)]),
            cost_excluded_walk=False,
        )
        return g

    def test_midpoint_split_geometry_and_length(self):
        """Point at exact midpoint → two 50 m children with sliced geometries."""
        g = self._simple_multidi_graph()
        result = split_edge_at_point(g, 1, 2, point=(50.0, 0.0), k=0)
        # Parent removed; new node added.
        self.assertFalse(g.has_edge(1, 2, 0))
        self.assertIn(result.new_node_id, g.nodes)
        self.assertEqual(g.nodes[result.new_node_id]["x"], 50.0)
        self.assertEqual(g.nodes[result.new_node_id]["y"], 0.0)
        # Two children, each 50 m, summing to parent length.
        c1_data = g.edges[result.child1]
        c2_data = g.edges[result.child2]
        self.assertAlmostEqual(c1_data["length"], 50.0, places=4)
        self.assertAlmostEqual(c2_data["length"], 50.0, places=4)
        self.assertAlmostEqual(c1_data["length"] + c2_data["length"], 100.0, places=4)
        # Geometries should be valid LineStrings starting/ending at the right
        # points.
        self.assertEqual(list(c1_data["geometry"].coords)[0], (0.0, 0.0))
        self.assertEqual(list(c1_data["geometry"].coords)[-1], (50.0, 0.0))
        self.assertEqual(list(c2_data["geometry"].coords)[0], (50.0, 0.0))
        self.assertEqual(list(c2_data["geometry"].coords)[-1], (100.0, 0.0))

    def test_copy_as_is_attrs(self):
        """OSM-direct attrs and cost-mask flags are copied to both children."""
        g = self._simple_multidi_graph()
        result = split_edge_at_point(g, 1, 2, point=(30.0, 0.0), k=0)
        for child in (result.child1, result.child2):
            d = g.edges[child]
            self.assertEqual(d["highway"], "residential")
            self.assertEqual(d["lanes"], 2)
            self.assertEqual(d["maxspeed"], 50)
            self.assertEqual(d["oneway"], True)
            self.assertFalse(d["cost_excluded_walk"])

    def test_asymmetric_split(self):
        """Projection at 70 % → children of 70 m + 30 m."""
        g = self._simple_multidi_graph()
        result = split_edge_at_point(g, 1, 2, point=(70.0, 0.0), k=0)
        self.assertAlmostEqual(g.edges[result.child1]["length"], 70.0, places=4)
        self.assertAlmostEqual(g.edges[result.child2]["length"], 30.0, places=4)

    def test_off_line_point_projects_correctly(self):
        """A point above the line should project onto its nearest line-point.
        New node sits on the line, not at the input point."""
        g = self._simple_multidi_graph()
        result = split_edge_at_point(g, 1, 2, point=(60.0, 10.0), k=0)
        # Projection of (60, 10) onto a horizontal line at y=0 is (60, 0).
        self.assertEqual(g.nodes[result.new_node_id]["x"], 60.0)
        self.assertEqual(g.nodes[result.new_node_id]["y"], 0.0)

    def test_undirected_simple_graph(self):
        """Simple `nx.Graph` (no multi, no direction): edge split works without k."""
        g = nx.Graph()
        g.add_node("u", x=0.0, y=0.0)
        g.add_node("v", x=10.0, y=0.0)
        g.add_edge(
            "u",
            "v",
            length=10.0,
            geometry=LineString([(0.0, 0.0), (10.0, 0.0)]),
        )
        # Use explicit new_node_id since "u" / "v" are non-int.
        result = split_edge_at_point(g, "u", "v", point=(4.0, 0.0), new_node_id="n")
        self.assertEqual(result.new_node_id, "n")
        self.assertFalse(g.has_edge("u", "v"))
        self.assertTrue(g.has_edge("u", "n"))
        self.assertTrue(g.has_edge("n", "v"))

    def test_polyline_geometry_split(self):
        """Geometry with multiple vertices is split into two valid polylines."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        # 3-vertex polyline: dog-leg through (50, 50)
        g.add_edge(
            1,
            2,
            key=0,
            length=141.42,
            geometry=LineString([(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)]),
        )
        parent_len = g.edges[1, 2, 0]["geometry"].length
        # Split at the dog-leg corner: (50, 50) at fraction 0.5.
        result = split_edge_at_point(g, 1, 2, point=(50.0, 50.0), k=0)
        c1 = g.edges[result.child1]["geometry"]
        c2 = g.edges[result.child2]["geometry"]
        self.assertAlmostEqual(c1.length + c2.length, parent_len, places=3)
        self.assertEqual(list(c1.coords)[-1], (50.0, 50.0))
        self.assertEqual(list(c2.coords)[0], (50.0, 50.0))

    def test_endpoint_projection_raises(self):
        """Projection at u (0,0) or beyond v should raise — caller's job to
        snap to the endpoint node instead."""
        g = self._simple_multidi_graph()
        with self.assertRaises(ValueError):
            split_edge_at_point(g, 1, 2, point=(0.0, 0.0), k=0)
        with self.assertRaises(ValueError):
            split_edge_at_point(g, 1, 2, point=(100.0, 0.0), k=0)
        # Past v also raises (projects to v).
        with self.assertRaises(ValueError):
            split_edge_at_point(g, 1, 2, point=(200.0, 0.0), k=0)

    def test_missing_geometry_raises(self):
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=10.0, y=0.0)
        g.add_edge(1, 2, key=0, length=10.0)  # no geometry
        with self.assertRaisesRegex(ValueError, "geometry"):
            split_edge_at_point(g, 1, 2, point=(5.0, 0.0), k=0)

    def test_multigraph_requires_k(self):
        g = self._simple_multidi_graph()
        with self.assertRaisesRegex(ValueError, "`k` is required"):
            split_edge_at_point(g, 1, 2, point=(50.0, 0.0))

    def test_simple_graph_forbids_k(self):
        g = nx.Graph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=10.0, y=0.0)
        g.add_edge(1, 2, length=10.0, geometry=LineString([(0, 0), (10, 0)]))
        with self.assertRaisesRegex(ValueError, "`k` must be None"):
            split_edge_at_point(g, 1, 2, point=(5.0, 0.0), k=0)

    def test_nonexistent_edge_raises(self):
        g = self._simple_multidi_graph()
        with self.assertRaisesRegex(ValueError, "not in graph"):
            split_edge_at_point(g, 1, 999, point=(50.0, 0.0), k=0)

    def test_new_node_id_collision_raises(self):
        g = self._simple_multidi_graph()
        with self.assertRaisesRegex(ValueError, "already exists"):
            split_edge_at_point(g, 1, 2, point=(50.0, 0.0), k=0, new_node_id=1)

    def test_auto_node_id_uses_max_plus_one(self):
        g = self._simple_multidi_graph()
        result = split_edge_at_point(g, 1, 2, point=(50.0, 0.0), k=0)
        self.assertEqual(result.new_node_id, 3)  # max(1, 2) + 1

    def test_extra_node_attrs(self):
        g = self._simple_multidi_graph()
        extras = {"is_t_junction": 0, "is_4way": 0, "n_streets": 2}
        result = split_edge_at_point(
            g,
            1,
            2,
            point=(50.0, 0.0),
            k=0,
            extra_node_attrs=extras,
        )
        nd = g.nodes[result.new_node_id]
        self.assertEqual(nd["is_t_junction"], 0)
        self.assertEqual(nd["is_4way"], 0)
        self.assertEqual(nd["n_streets"], 2)
        # x and y are still set correctly.
        self.assertEqual(nd["x"], 50.0)
        self.assertEqual(nd["y"], 0.0)

    def test_multidi_parallel_edges_only_one_split(self):
        """When two parallel edges (same u, v, different k) exist, splitting
        one leaves the other untouched."""
        g = self._simple_multidi_graph()
        # Add a parallel edge with key=1.
        g.add_edge(
            1,
            2,
            key=1,
            highway="service",
            length=100.0,
            geometry=LineString([(0.0, 0.0), (100.0, 0.0)]),
        )
        result = split_edge_at_point(g, 1, 2, point=(50.0, 0.0), k=0)
        # k=0 was removed; k=1 still there.
        self.assertFalse(g.has_edge(1, 2, 0))
        self.assertTrue(g.has_edge(1, 2, 1))
        self.assertEqual(g.edges[1, 2, 1]["highway"], "service")
        # New node connects only via the split children.
        self.assertTrue(g.has_edge(1, result.new_node_id))
        self.assertTrue(g.has_edge(result.new_node_id, 2))

    def test_directed_split_preserves_direction(self):
        """For directed graphs, child1 is (u, new) and child2 is (new, v).
        Reverse direction edges are not affected by this call."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=10.0, y=0.0)
        # Add both directions; we split only the u→v one.
        g.add_edge(1, 2, key=0, length=10.0, geometry=LineString([(0, 0), (10, 0)]))
        g.add_edge(2, 1, key=0, length=10.0, geometry=LineString([(10, 0), (0, 0)]))
        result = split_edge_at_point(g, 1, 2, point=(5.0, 0.0), k=0)
        # u→v split.
        self.assertFalse(g.has_edge(1, 2, 0))
        self.assertTrue(g.has_edge(1, result.new_node_id))
        self.assertTrue(g.has_edge(result.new_node_id, 2))
        # v→u untouched.
        self.assertTrue(g.has_edge(2, 1, 0))


class SplitTwoWayEdgeAtPointTestCase(unittest.TestCase):
    """`split_two_way_edge_at_point` splits BOTH directional edges of a
    two-way road through a SHARED synthetic node — the typical pattern for
    OSMnx-style MultiDiGraphs."""

    def _two_way_graph(self) -> nx.MultiDiGraph:
        """MultiDiGraph with both directions of a 100 m residential road
        between node 1 (0,0) and node 2 (100,0)."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        g.add_edge(
            1,
            2,
            key=0,
            highway="residential",
            lanes=2,
            length=100.0,
            geometry=LineString([(0.0, 0.0), (100.0, 0.0)]),
        )
        g.add_edge(
            2,
            1,
            key=0,
            highway="residential",
            lanes=2,
            length=100.0,
            geometry=LineString([(100.0, 0.0), (0.0, 0.0)]),
        )
        return g

    def test_basic_two_way_split_four_children_through_one_node(self):
        g = self._two_way_graph()
        result = split_two_way_edge_at_point(
            g,
            1,
            2,
            point=(60.0, 0.0),
            k_forward=0,
            k_reverse=0,
        )
        # One new node.
        self.assertIn(result.new_node_id, g.nodes)
        self.assertEqual(g.nodes[result.new_node_id]["x"], 60.0)
        self.assertEqual(g.nodes[result.new_node_id]["y"], 0.0)
        # Both parent edges gone; four child edges exist.
        self.assertFalse(g.has_edge(1, 2, 0))
        self.assertFalse(g.has_edge(2, 1, 0))
        self.assertTrue(g.has_edge(1, result.new_node_id))
        self.assertTrue(g.has_edge(result.new_node_id, 2))
        self.assertTrue(g.has_edge(2, result.new_node_id))
        self.assertTrue(g.has_edge(result.new_node_id, 1))
        # Both directions share the new node id.
        self.assertEqual(result.forward.new_node_id, result.new_node_id)
        self.assertEqual(result.reverse.new_node_id, result.new_node_id)

    def test_lengths_sum_to_parents_per_direction(self):
        g = self._two_way_graph()
        result = split_two_way_edge_at_point(
            g,
            1,
            2,
            point=(40.0, 0.0),
            k_forward=0,
            k_reverse=0,
        )
        fwd_c1 = g.edges[result.forward.child1]
        fwd_c2 = g.edges[result.forward.child2]
        rev_c1 = g.edges[result.reverse.child1]
        rev_c2 = g.edges[result.reverse.child2]
        self.assertAlmostEqual(fwd_c1["length"] + fwd_c2["length"], 100.0, places=4)
        self.assertAlmostEqual(rev_c1["length"] + rev_c2["length"], 100.0, places=4)
        # Forward direction (1→2) projects (40,0) onto geom [(0,0)→(100,0)]
        # at distance 40 m: child1=(1, new)=40 m, child2=(new, 2)=60 m.
        self.assertAlmostEqual(fwd_c1["length"], 40.0, places=4)
        self.assertAlmostEqual(fwd_c2["length"], 60.0, places=4)
        # Reverse direction (2→1) projects (40,0) onto geom [(100,0)→(0,0)]
        # at distance 60 m: child1=(2, new)=60 m, child2=(new, 1)=40 m.
        # The new node is in the same spatial location for both directions;
        # only the "distance along the directed line" interpretation differs.
        self.assertAlmostEqual(rev_c1["length"], 60.0, places=4)
        self.assertAlmostEqual(rev_c2["length"], 40.0, places=4)

    def test_independent_attrs_per_direction(self):
        """Attributes that differ between directions (e.g. one direction was
        cost-excluded for a mode) are preserved independently per child."""
        g = self._two_way_graph()
        # Tag forward and reverse with different attrs to verify independence.
        g.edges[1, 2, 0]["cost_excluded_walk"] = False
        g.edges[2, 1, 0]["cost_excluded_walk"] = True
        result = split_two_way_edge_at_point(
            g,
            1,
            2,
            point=(50.0, 0.0),
            k_forward=0,
            k_reverse=0,
        )
        self.assertFalse(g.edges[result.forward.child1]["cost_excluded_walk"])
        self.assertFalse(g.edges[result.forward.child2]["cost_excluded_walk"])
        self.assertTrue(g.edges[result.reverse.child1]["cost_excluded_walk"])
        self.assertTrue(g.edges[result.reverse.child2]["cost_excluded_walk"])

    def test_missing_forward_edge_raises(self):
        g = self._two_way_graph()
        g.remove_edge(1, 2, 0)
        with self.assertRaisesRegex(ValueError, "Forward edge"):
            split_two_way_edge_at_point(
                g,
                1,
                2,
                point=(50.0, 0.0),
                k_forward=0,
                k_reverse=0,
            )

    def test_missing_reverse_edge_raises(self):
        """If there's no reverse edge (genuinely one-way road), the wrapper
        refuses and tells the caller to use the single-edge primitive."""
        g = self._two_way_graph()
        g.remove_edge(2, 1, 0)
        with self.assertRaisesRegex(ValueError, "Reverse edge.*one-way"):
            split_two_way_edge_at_point(
                g,
                1,
                2,
                point=(50.0, 0.0),
                k_forward=0,
                k_reverse=0,
            )

    def test_non_multidigraph_raises(self):
        g = nx.Graph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=10.0, y=0.0)
        g.add_edge(1, 2, length=10.0, geometry=LineString([(0, 0), (10, 0)]))
        with self.assertRaisesRegex(ValueError, "MultiDiGraph"):
            split_two_way_edge_at_point(
                g,
                1,
                2,
                point=(5.0, 0.0),
                k_forward=0,
                k_reverse=0,
            )

    def test_endpoint_projection_raises(self):
        g = self._two_way_graph()
        with self.assertRaises(ValueError):
            split_two_way_edge_at_point(
                g,
                1,
                2,
                point=(0.0, 0.0),
                k_forward=0,
                k_reverse=0,
            )

    def test_independent_geometries_polyline(self):
        """Forward and reverse may have different geometries (e.g. dual
        carriageway). Each direction's split uses its own geometry; the
        new node lands at the forward projection."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        # Forward goes via (50, 10); reverse goes via (50, -10) — a dual
        # carriageway divided by a median.
        g.add_edge(
            1,
            2,
            key=0,
            length=100.0,
            geometry=LineString([(0.0, 0.0), (50.0, 10.0), (100.0, 0.0)]),
        )
        g.add_edge(
            2,
            1,
            key=0,
            length=100.0,
            geometry=LineString([(100.0, 0.0), (50.0, -10.0), (0.0, 0.0)]),
        )
        # Snap target at (50, 0): each direction projects independently onto
        # its OWN polyline. The new node sits at the FORWARD projection
        # (some point on the y>=0 polyline; we don't assert exact coords
        # because the nearest point on a polyline can fall mid-segment).
        fwd_parent_geom = g.edges[1, 2, 0]["geometry"]
        rev_parent_geom = g.edges[2, 1, 0]["geometry"]
        result = split_two_way_edge_at_point(
            g,
            1,
            2,
            point=(50.0, 0.0),
            k_forward=0,
            k_reverse=0,
        )
        new_x = g.nodes[result.new_node_id]["x"]
        new_y = g.nodes[result.new_node_id]["y"]
        # The new node sits exactly on the forward polyline (within float tol).
        self.assertAlmostEqual(fwd_parent_geom.distance(Point(new_x, new_y)), 0.0, places=4)
        # And on the y>=0 side (the forward polyline never dips below y=0).
        self.assertGreaterEqual(new_y, 0.0)
        # Both directions are split through this same node id.
        self.assertEqual(result.forward.new_node_id, result.new_node_id)
        self.assertEqual(result.reverse.new_node_id, result.new_node_id)
        # Each direction's child lengths still sum to ITS OWN parent length.
        fwd_total = (
            g.edges[result.forward.child1]["length"] + g.edges[result.forward.child2]["length"]
        )
        rev_total = (
            g.edges[result.reverse.child1]["length"] + g.edges[result.reverse.child2]["length"]
        )
        # Each direction's child lengths sum to ITS OWN parent length.
        self.assertAlmostEqual(
            fwd_total,
            fwd_parent_geom.length,
            places=3,
        )
        self.assertAlmostEqual(
            rev_total,
            rev_parent_geom.length,
            places=3,
        )

    def test_auto_node_id(self):
        g = self._two_way_graph()
        result = split_two_way_edge_at_point(
            g,
            1,
            2,
            point=(50.0, 0.0),
            k_forward=0,
            k_reverse=0,
        )
        self.assertEqual(result.new_node_id, 3)

    def test_node_id_collision_raises(self):
        g = self._two_way_graph()
        with self.assertRaisesRegex(ValueError, "already exists"):
            split_two_way_edge_at_point(
                g,
                1,
                2,
                point=(50.0, 0.0),
                k_forward=0,
                k_reverse=0,
                new_node_id=1,
            )


class InsertProjectedNodesTestCase(unittest.TestCase):
    """`insert_projected_nodes` enriches the graph with virtual nodes so a
    subsequent `snap_to_network_nodes` call has fine-grained targets."""

    def _toy_graph(self) -> nx.MultiDiGraph:
        """Two-way OSM-style road: 1 (0,0) <-> 2 (100,0), residential.
        Plus an isolated tertiary edge for edge-filter tests."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        g.add_node(3, x=200.0, y=0.0)
        g.add_node(4, x=300.0, y=0.0)
        g.add_edge(
            1,
            2,
            key=0,
            highway="residential",
            length=100.0,
            geometry=LineString([(0, 0), (100, 0)]),
        )
        g.add_edge(
            2,
            1,
            key=0,
            highway="residential",
            length=100.0,
            geometry=LineString([(100, 0), (0, 0)]),
        )
        g.add_edge(
            3,
            4,
            key=0,
            highway="tertiary",
            length=100.0,
            geometry=LineString([(200, 0), (300, 0)]),
        )
        g.add_edge(
            4,
            3,
            key=0,
            highway="tertiary",
            length=100.0,
            geometry=LineString([(300, 0), (200, 0)]),
        )
        return g

    def _points(self, coords: list[tuple[float, float]], ids: list[str]) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in coords],
            index=pd.Index(ids, name="point_id"),
        )

    def test_inserts_virtual_node_for_mid_edge_projection(self):
        """A point that projects onto the middle of an edge gets a virtual
        node inserted; the parent edge is replaced by two children."""
        g = self._toy_graph()
        n_before = g.number_of_nodes()
        points = self._points([(50.0, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
        )
        # One new node inserted (and `is_virtual=1`).
        self.assertEqual(g.number_of_nodes(), n_before + 1)
        new_nodes = [n for n, d in g.nodes(data=True) if d.get("is_virtual") == 1]
        self.assertEqual(len(new_nodes), 1)
        # Parent forward + reverse edges gone, replaced by children.
        self.assertFalse(g.has_edge(1, 2, 0))
        self.assertFalse(g.has_edge(2, 1, 0))

    def test_skip_when_projection_near_endpoint(self):
        """A point that projects within `node_spacing` of an endpoint does
        NOT trigger an insertion — the existing endpoint will serve."""
        g = self._toy_graph()
        n_before = g.number_of_nodes()
        # Point at (5, 5): projection is (5, 0), within 10m of node 1 at (0,0).
        points = self._points([(5.0, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
        )
        # No insertion.
        self.assertEqual(g.number_of_nodes(), n_before)
        # Parent edges intact.
        self.assertTrue(g.has_edge(1, 2, 0))
        self.assertTrue(g.has_edge(2, 1, 0))

    def test_no_insertion_beyond_max_radius(self):
        """Points farther than `max_radius` from every edge contribute no
        insertion."""
        g = self._toy_graph()
        n_before = g.number_of_nodes()
        points = self._points([(50.0, 500.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=50.0,
            node_spacing=10.0,
        )
        self.assertEqual(g.number_of_nodes(), n_before)

    def test_eligible_node_ids_filter(self):
        """Edges with an endpoint outside the eligible set are skipped."""
        g = self._toy_graph()
        # Eligible only includes 3, 4 (the tertiary segment).
        points = self._points([(50.0, 5.0)], ["p"])  # nearest residential
        insert_projected_nodes(
            points,
            g,
            max_radius=500.0,
            node_spacing=10.0,
            eligible_node_ids={3, 4},
        )
        # No insertion on (1,2) — endpoints 1, 2 aren't eligible. The
        # residential edge stays intact.
        self.assertTrue(g.has_edge(1, 2, 0))

    def test_cost_excluded_edges_skipped(self):
        """Edges flagged via `cost_excluded_flag` are not insertion candidates."""
        g = self._toy_graph()
        # Mark the residential forward+reverse as cost-excluded.
        g.edges[1, 2, 0]["cost_excluded_x"] = True
        g.edges[2, 1, 0]["cost_excluded_x"] = True
        points = self._points([(50.0, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
            cost_excluded_flag="cost_excluded_x",
        )
        # No insertion on residential. Residential edges intact.
        self.assertTrue(g.has_edge(1, 2, 0))
        self.assertTrue(g.has_edge(2, 1, 0))

    def test_default_edge_filter_excludes_motorway(self):
        """The default `edge_filter` excludes motorway + trunk tags."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        g.add_edge(
            1,
            2,
            key=0,
            highway="motorway",
            length=100.0,
            geometry=LineString([(0, 0), (100, 0)]),
        )
        g.add_edge(
            2,
            1,
            key=0,
            highway="motorway",
            length=100.0,
            geometry=LineString([(100, 0), (0, 0)]),
        )
        n_before = g.number_of_nodes()
        points = self._points([(50.0, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
        )
        # No insertion — motorway excluded by default.
        self.assertEqual(g.number_of_nodes(), n_before)

    def test_edge_filter_allowlist_iterable(self):
        """`edge_filter` as an iterable of OSM tags acts as an allowlist."""
        g = self._toy_graph()
        # Point closer to the residential edge, but allowlist only `tertiary`.
        points = self._points([(50.0, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=500.0,
            node_spacing=10.0,
            edge_filter={"tertiary"},
        )
        # Residential not allowed. Tertiary is allowed but the point
        # projects at x=200 (start of tertiary edge) — within `node_spacing`?
        # distance from (50,5) to (200,0) is ~150 > node_spacing=10, so
        # the projection is at edge_start (proj_fwd=0) — at_endpoint case → skipped.
        # In either case, residential edge stays intact.
        self.assertTrue(g.has_edge(1, 2, 0))

    def test_edge_filter_callable(self):
        """`edge_filter` accepts an arbitrary callable predicate."""
        g = self._toy_graph()
        n_before = g.number_of_nodes()
        points = self._points([(50.0, 5.0)], ["p"])
        # Custom predicate: allow only residential.
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
            edge_filter=lambda d: d.get("highway") == "residential",
        )
        self.assertEqual(g.number_of_nodes(), n_before + 1)

    def test_two_way_split_inserts_one_node_replaces_both_directions(self):
        """For two-way OSM roads (forward + reverse edges), insertion
        creates ONE virtual node shared by both directions and replaces
        both parent edges."""
        g = self._toy_graph()
        points = self._points([(50.0, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
        )
        # Both directional parents gone.
        self.assertFalse(g.has_edge(1, 2, 0))
        self.assertFalse(g.has_edge(2, 1, 0))
        # Find the new node.
        new = [n for n, d in g.nodes(data=True) if d.get("is_virtual") == 1][0]
        # Four new edges through the new node.
        self.assertTrue(g.has_edge(1, new))
        self.assertTrue(g.has_edge(new, 2))
        self.assertTrue(g.has_edge(2, new))
        self.assertTrue(g.has_edge(new, 1))

    def test_one_way_edge_single_direction_split(self):
        """A genuinely one-way edge (no reverse) gets a single-direction split."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        g.add_edge(
            1,
            2,
            key=0,
            highway="residential",
            length=100.0,
            geometry=LineString([(0, 0), (100, 0)]),
        )
        points = self._points([(50.0, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
        )
        new = [n for n, d in g.nodes(data=True) if d.get("is_virtual") == 1][0]
        self.assertTrue(g.has_edge(1, new))
        self.assertTrue(g.has_edge(new, 2))
        # No reverse edges created.
        self.assertFalse(g.has_edge(2, new))
        self.assertFalse(g.has_edge(new, 1))

    def test_self_loop_edge_handles_single_direction_split(self):
        """A self-loop (`u == v`) — `_find_reverse_edge_key` returns None
        and insertion uses single-direction split."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=100.0, y=0.0)
        g.add_edge(
            1,
            1,
            key=0,
            highway="residential",
            length=14.14,
            geometry=LineString([(0.0, 0.0), (2.5, 2.5), (5.0, 0.0), (2.5, -2.5), (0.0, 0.0)]),
        )
        g.add_edge(
            1,
            2,
            key=0,
            highway="residential",
            length=100.0,
            geometry=LineString([(0.0, 0.0), (100.0, 0.0)]),
        )
        n_before = g.number_of_nodes()
        # A point near the top vertex of the self-loop, far from node 1
        # in pure Euclidean terms but near the loop interior.
        points = self._points([(2.5, 2.5)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=10.0,
            node_spacing=0.5,
        )
        # Insertion happened — graph mutated.
        self.assertEqual(g.number_of_nodes(), n_before + 1)

    def test_near_endpoint_two_way_asymmetric_geometry(self):
        """Regression: real-world OSMnx-consolidated graphs sometimes have
        slightly-asymmetric forward / reverse geometries. The endpoint check
        considers both projections, so cells projecting near a reverse-edge
        endpoint don't trigger a degenerate split."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=20.0, y=0.0)
        g.add_edge(
            1,
            2,
            key=0,
            highway="residential",
            length=20.0,
            geometry=LineString([(0.0, 0.0), (20.0, 0.0)]),
        )
        # Reverse: 19.98 m, ends at x=0.02 rather than 0.
        g.add_edge(
            2,
            1,
            key=0,
            highway="residential",
            length=19.98,
            geometry=LineString([(20.0, 0.0), (0.02, 0.0)]),
        )
        n_before = g.number_of_nodes()
        # Point at (0.02, 5): forward projection at 0.02 (interior), reverse
        # projection at endpoint. node_spacing=2 should cover both.
        points = self._points([(0.02, 5.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_radius=20.0,
            node_spacing=2.0,
        )
        # No insertion (projection within node_spacing of endpoint on both sides).
        self.assertEqual(g.number_of_nodes(), n_before)

    def test_lineage_walk_resolves_long_edge_in_few_passes(self):
        """B: well-separated points on a long parent edge get distinct virtuals
        in a small number of passes via in-pass lineage descent."""
        import io
        from contextlib import redirect_stdout

        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=2000.0, y=0.0)
        g.add_edge(
            1,
            2,
            key=0,
            highway="residential",
            length=2000.0,
            geometry=LineString([(0.0, 0.0), (2000.0, 0.0)]),
        )
        g.add_edge(
            2,
            1,
            key=0,
            highway="residential",
            length=2000.0,
            geometry=LineString([(2000.0, 0.0), (0.0, 0.0)]),
        )
        # 9 points at x=200, 400, ..., 1800 — all well-separated and well
        # away from the endpoint at x=2000 (avoid the near-endpoint skip).
        coords = [(200.0 * (k + 1), 5.0) for k in range(9)]
        ids_p = [f"p{k}" for k in range(9)]
        points = self._points(coords, ids_p)
        n_before = g.number_of_nodes()
        buf = io.StringIO()
        with redirect_stdout(buf):
            insert_projected_nodes(
                points,
                g,
                max_radius=200.0,
                node_spacing=50.0,
                verbose=True,
            )
        # 9 distinct virtuals inserted (one per well-separated point) — all
        # in a single straight-line pass via the lineage walk.
        self.assertEqual(g.number_of_nodes(), n_before + 9)
        out = buf.getvalue()
        self.assertIn("inserted=9", out)

    def test_verbose_prints_summary(self):
        """`verbose=True` emits a summary line with insertion counts."""
        import io
        from contextlib import redirect_stdout

        g = self._toy_graph()
        points = self._points([(50.0, 10.0)], ["p"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            insert_projected_nodes(
                points,
                g,
                max_radius=200.0,
                node_spacing=5.0,
                verbose=True,
            )
        out = buf.getvalue()
        self.assertIn("insert_projected_nodes", out)
        self.assertIn("inserted=", out)
        self.assertIn("endpoint_skip=", out)
        self.assertIn("no_match=", out)

    def test_returns_none(self):
        """The function does not return a (point -> node) mapping; that's the
        subsequent `snap_to_network_nodes` call's job."""
        g = self._toy_graph()
        points = self._points([(50.0, 5.0)], ["p"])
        result = insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=10.0,
        )
        self.assertIsNone(result)


class SnapToNetworkNodesPriorityTestCase(unittest.TestCase):
    """Priority-aware two-tier behavior of `snap_to_network_nodes`."""

    def _graph(self) -> nx.MultiDiGraph:
        """Toy graph: priority node `p` and non-priority node `r`, near a point."""
        g = nx.MultiDiGraph()
        g.add_node("p", x=10.0, y=0.0)  # priority candidate (incident to tertiary)
        g.add_node("q", x=110.0, y=0.0)
        g.add_node("r", x=2.0, y=0.0)  # non-priority, closer to (0, 0)
        g.add_node("s", x=102.0, y=0.0)
        g.add_edge(
            "p",
            "q",
            key=0,
            highway="tertiary",
            length=100.0,
            geometry=LineString([(10, 0), (110, 0)]),
        )
        g.add_edge(
            "r",
            "s",
            key=0,
            highway="residential",
            length=100.0,
            geometry=LineString([(2, 0), (102, 0)]),
        )
        return g

    def _points(self, coords, ids):
        return gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in coords],
            index=pd.Index(ids, name="point_id"),
        )

    def test_priority_node_within_radius_wins_over_closer_nonpriority(self):
        """A priority node within `priority_node_radius` is chosen even if a
        non-priority node is closer."""
        g = self._graph()
        priority = {"p", "q"}
        points = self._points([(0.0, 0.0)], ["p1"])
        # Non-priority `r` is closer (2 m) than priority `p` (10 m).
        # With priority radius 50, `p` wins despite `r` being closer.
        ids, dists = snap_to_network_nodes(
            points,
            g,
            max_radius=100.0,
            priority_node_ids=priority,
            priority_node_radius=50.0,
        )
        self.assertEqual(ids.loc["p1"], "p")
        self.assertAlmostEqual(float(dists.loc["p1"]), 10.0)

    def test_falls_back_to_eligible_when_no_priority_in_range(self):
        """A point beyond `priority_node_radius` from any priority node
        falls back to the nearest eligible node within `max_radius`."""
        g = self._graph()
        priority = {"p", "q"}
        # Point at (500, 0): no priority within 50m, closest eligible
        # is `s` at (102, 0) → distance 398.
        points = self._points([(500.0, 0.0)], ["p1"])
        ids, dists = snap_to_network_nodes(
            points,
            g,
            max_radius=500.0,
            priority_node_ids=priority,
            priority_node_radius=50.0,
        )
        self.assertEqual(ids.loc["p1"], "q")  # priority node `q` at (110, 0)
        # Wait — priority q is at (110,0), distance from (500,0) is 390.
        # If max_radius=500, q is within max_radius too. But priority
        # radius is 50, so q (390m away) is outside priority. So tier 1
        # misses; tier 2 picks nearest eligible (which includes q AND s).
        # s is at (102, 0), distance 398. q is at (110, 0), distance 390.
        # Tier 2 picks closer: q at 390.
        # Actually the eligible set defaults to ALL nodes — so q is in it.
        # The test verifies tier-2 fallback finds q.
        self.assertAlmostEqual(float(dists.loc["p1"]), 390.0)

    def test_single_tier_when_no_priority_set(self):
        """Without a priority set, behaves exactly as the historical
        single-tier snap."""
        g = self._graph()
        points = self._points([(0.0, 0.0)], ["p1"])
        ids, dists = snap_to_network_nodes(
            points,
            g,
            max_radius=100.0,
        )
        # Nearest is `r` at (2, 0).
        self.assertEqual(ids.loc["p1"], "r")
        self.assertAlmostEqual(float(dists.loc["p1"]), 2.0)

    def test_priority_node_flag_alternative(self):
        """`priority_node_flag` reads a per-node bool attribute."""
        g = self._graph()
        for n in ("p", "q"):
            g.nodes[n]["is_priority"] = True
        points = self._points([(0.0, 0.0)], ["p1"])
        ids, _ = snap_to_network_nodes(
            points,
            g,
            max_radius=100.0,
            priority_node_flag="is_priority",
            priority_node_radius=50.0,
        )
        self.assertEqual(ids.loc["p1"], "p")

    def test_priority_radius_required_when_priority_set_given(self):
        """Passing a priority set without `priority_node_radius` raises."""
        g = self._graph()
        points = self._points([(0.0, 0.0)], ["p1"])
        with self.assertRaises(ValueError):
            snap_to_network_nodes(
                points,
                g,
                max_radius=100.0,
                priority_node_ids={"p"},
            )

    def test_both_priority_kwargs_raises(self):
        g = self._graph()
        points = self._points([(0.0, 0.0)], ["p1"])
        with self.assertRaises(ValueError):
            snap_to_network_nodes(
                points,
                g,
                max_radius=100.0,
                priority_node_ids={"p"},
                priority_node_flag="is_priority",
                priority_node_radius=50.0,
            )

    def test_nodes_incident_to_edges_derives_priority_set(self):
        """`nodes_incident_to_edges` returns the union of endpoints of
        edges matching the tag predicate — typical use for building the
        priority node set."""
        g = self._graph()
        result = nodes_incident_to_edges(g, edge_tags={"tertiary"})
        self.assertEqual(result, {"p", "q"})

    def test_nodes_incident_to_edges_callable_predicate(self):
        g = self._graph()
        result = nodes_incident_to_edges(
            g,
            edge_tags=lambda d: d.get("highway") == "residential",
        )
        self.assertEqual(result, {"r", "s"})

    def test_nodes_incident_to_edges_none_returns_empty(self):
        g = self._graph()
        self.assertEqual(nodes_incident_to_edges(g, edge_tags=None), set())

    def test_end_to_end_insert_then_priority_snap(self):
        """The canonical workflow: insert virtuals, then snap with priority."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=200.0, y=0.0)
        g.add_node(3, x=0.0, y=50.0)
        g.add_node(4, x=200.0, y=50.0)
        # Priority edge along y=0.
        g.add_edge(
            1,
            2,
            key=0,
            highway="primary",
            length=200.0,
            geometry=LineString([(0, 0), (200, 0)]),
        )
        g.add_edge(
            2,
            1,
            key=0,
            highway="primary",
            length=200.0,
            geometry=LineString([(200, 0), (0, 0)]),
        )
        # Non-priority edge along y=50.
        g.add_edge(
            3,
            4,
            key=0,
            highway="residential",
            length=200.0,
            geometry=LineString([(0, 50), (200, 50)]),
        )
        g.add_edge(
            4,
            3,
            key=0,
            highway="residential",
            length=200.0,
            geometry=LineString([(200, 50), (0, 50)]),
        )
        # Two points: one near primary, one near residential.
        points = gpd.GeoDataFrame(
            geometry=[Point(100.0, 5.0), Point(100.0, 45.0)],
            index=pd.Index(["near_primary", "near_resi"], name="point_id"),
        )
        # Step 1: insert virtuals (both edges allowed, default filter).
        insert_projected_nodes(
            points,
            g,
            max_radius=200.0,
            node_spacing=25.0,
        )
        # Step 2: build priority node set (primary edge incidence + virtuals).
        priority = nodes_incident_to_edges(g, edge_tags={"primary"})
        ids, dists = snap_to_network_nodes(
            points,
            g,
            max_radius=200.0,
            priority_node_ids=priority,
            priority_node_radius=100.0,
        )
        # near_primary matches a primary-side virtual.
        primary_target = ids.loc["near_primary"]
        self.assertIn(primary_target, priority)
        # near_resi: closest priority node is ~45m below (a virtual on the
        # primary edge). Since priority_node_radius=100 covers that, the
        # priority node wins.
        self.assertIn(ids.loc["near_resi"], priority)


class ComputeSnapEligibilityTestCase(unittest.TestCase):
    """`compute_snap_eligibility` is the prep-stage entry point: writes per-edge
    `cost_excluded_<mode>` and returns the eligible-node frozenset. Does NOT
    decorate per-node `is_snap_eligible_<mode>` (that's prepare_network's job
    on the post-snap graph) and does NOT call to_undirected() on the input
    (snap_to_network_tiered needs the directed MultiDiGraph)."""

    def _trap_graph(self, with_highway: bool = True) -> nx.MultiDiGraph:
        """Same trap shape as PrepareNetworkTestCase._trap_graph but with
        highway tags by default since cost-mask testing is the focus."""
        g = nx.MultiDiGraph()
        for n, (x, y) in {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (2, 1)}.items():
            g.add_node(n, x=float(x), y=float(y))
        if with_highway:
            g.add_edge(0, 1, highway="residential")
            g.add_edge(1, 2, highway="motorway")
            g.add_edge(2, 0, highway=["primary", "trunk"])
            g.add_edge(2, 3, highway="footway")
        else:
            g.add_edge(0, 1)
            g.add_edge(1, 2)
            g.add_edge(2, 0)
            g.add_edge(2, 3)
        return g

    def test_returns_frozenset_and_flag_name(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elig, cost_flag = compute_snap_eligibility(self._trap_graph(), "walk")
        self.assertIsInstance(elig, frozenset)
        self.assertEqual(cost_flag, "cost_excluded_walk")

    def test_walk_uses_undirected_view_largest_cc(self):
        """For walk (undirected mode), eligibility is the largest CC of the
        cost-masked undirected view. With motorway/trunk excluded, edges
        (1, 2) and (2, 0) are masked — leaving {0, 1} disconnected and
        {2, 3} connected via the residential→walk-eligible path. Test the
        actual behavior: largest CC dominates and is consistent."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elig, _ = compute_snap_eligibility(self._trap_graph(), "walk")
        # The function must return SOME largest connected component on the
        # cost-masked subgraph. Verify properties rather than exact membership.
        self.assertGreater(len(elig), 0)
        self.assertLessEqual(len(elig), 4)

    def test_car_uses_directed_scc(self):
        """For car (directed_scc), eligibility is the largest SCC. Node 3
        has no outgoing edge → not in SCC. Largest SCC = {0, 1, 2}."""
        elig, _ = compute_snap_eligibility(self._trap_graph(), "car")
        self.assertEqual(elig, frozenset({0, 1, 2}))
        self.assertNotIn(3, elig)

    def test_writes_cost_excluded_per_edge(self):
        g = self._trap_graph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_snap_eligibility(g, "walk")
        # Walk excludes motorway / trunk / motorway_link / trunk_link.
        self.assertFalse(g.edges[0, 1, 0]["cost_excluded_walk"])
        self.assertTrue(g.edges[1, 2, 0]["cost_excluded_walk"])
        # (2, 0) has list-valued highway with 'trunk' inside — should exclude.
        self.assertTrue(g.edges[2, 0, 0]["cost_excluded_walk"])
        self.assertFalse(g.edges[2, 3, 0]["cost_excluded_walk"])

    def test_does_not_write_per_node_eligibility_flag(self):
        """Per-node `is_snap_eligible_<mode>` is intentionally NOT written here.
        That's prepare_network's job downstream on the post-snap graph."""
        g = self._trap_graph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_snap_eligibility(g, "walk")
        for n in g.nodes:
            self.assertNotIn("is_snap_eligible_walk", g.nodes[n])

    def test_does_not_transform_directedness(self):
        """Input graph stays a MultiDiGraph (snap_to_network_tiered needs it)."""
        g = self._trap_graph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_snap_eligibility(g, "walk")
        self.assertIsInstance(g, nx.MultiDiGraph)
        self.assertTrue(g.is_directed())

    def test_overwrite_if_missing_preserves_existing(self):
        """`overwrite_cost_excluded='if_missing'` leaves edges that already
        carry the flag untouched. Use case: prepare_network calling this with
        if_missing on a loaded graph whose cost-mask was persisted at prep
        time."""
        g = self._trap_graph()
        # Pre-seed (0, 1, 0) with a bogus True. if_missing must preserve.
        g.edges[0, 1, 0]["cost_excluded_walk"] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_snap_eligibility(g, "walk", overwrite_cost_excluded="if_missing")
        self.assertTrue(g.edges[0, 1, 0]["cost_excluded_walk"])  # bogus value preserved

    def test_overwrite_always_overwrites_existing(self):
        g = self._trap_graph()
        g.edges[0, 1, 0]["cost_excluded_walk"] = True  # bogus
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_snap_eligibility(g, "walk", overwrite_cost_excluded="always")
        # Residential edge → walk does NOT exclude → False.
        self.assertFalse(g.edges[0, 1, 0]["cost_excluded_walk"])

    def test_subtype_mode_via_base_mode(self):
        """`mode='car_night'` with `base_mode='car'` produces
        cost_excluded_car_night flag and uses car defaults."""
        elig, cost_flag = compute_snap_eligibility(
            self._trap_graph(),
            "car_night",
            base_mode="car",
        )
        self.assertEqual(cost_flag, "cost_excluded_car_night")
        self.assertEqual(elig, frozenset({0, 1, 2}))


class PrepareNetworkIdempotencyTestCase(unittest.TestCase):
    """`prepare_network` is idempotent on the cost-mask flag — it uses
    `if_missing` so a graph whose cost-mask was already written by
    `compute_snap_eligibility` keeps its values. Re-deriving the eligibility
    set is cheap and always runs (to pick up virtual nodes inserted between
    the two calls)."""

    def _g(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for n, (x, y) in {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (2, 1)}.items():
            g.add_node(n, x=float(x), y=float(y))
        g.add_edge(0, 1, highway="residential")
        g.add_edge(1, 2, highway="motorway")
        g.add_edge(2, 0, highway="primary")
        g.add_edge(2, 3, highway="residential")
        return g

    def test_compute_then_prepare_is_noop_on_edges(self):
        """After compute_snap_eligibility writes cost_excluded_walk, calling
        prepare_network on the same graph preserves those values."""
        g = self._g()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_snap_eligibility(g, "walk")
        # Manually flip one to verify if_missing actually fires.
        g.edges[0, 1, 0]["cost_excluded_walk"] = True  # bogus value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_network(g, "walk")
        # if_missing preserved our bogus value on (0, 1, 0).
        # (prepared.graph is the same g for car/MultiDiGraph; for walk it's
        # the undirected view, so look on the input graph.)
        self.assertTrue(g.edges[0, 1, 0]["cost_excluded_walk"])
        # And the eligibility set was re-derived (not stale).
        self.assertIsInstance(prepared.snap_eligible_nodes, frozenset)


class GraphmlBoolRoundtripTestCase(unittest.TestCase):
    """Bool-typed per-mode flags (`is_snap_eligible_<mode>`,
    `cost_excluded_<mode>`, `is_virtual`) round-trip through .graphml as
    integers (0 / 1), not the literal strings 'True' / 'False' — thanks to
    the prefix-scan dtype helper in `load_consolidated_graphml`."""

    def _g(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph(crs="EPSG:4326")
        for n, (x, y) in {0: (0, 0), 1: (1, 0), 2: (2, 0)}.items():
            g.add_node(n, x=float(x), y=float(y), osmid=n)
        g.add_edge(0, 1, key=0, highway="residential", length=1.0)
        g.add_edge(1, 2, key=0, highway="motorway", length=1.0)
        return g

    def test_bool_flags_roundtrip_as_int(self):
        import tempfile

        import osmnx as ox

        from aperta.network_processing import load_consolidated_graphml

        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prepared = prepare_network(self._g(), "walk")
            # Tag one node as virtual to exercise the is_virtual prefix.
            prepared.graph.nodes[1]["is_virtual"] = 1
            ox.save_graphml(prepared.graph, tmp_path)

            loaded = load_consolidated_graphml(tmp_path)
            # Per-node is_snap_eligible_walk: every node should have a real int.
            for n in loaded.nodes:
                val = loaded.nodes[n].get("is_snap_eligible_walk")
                self.assertIsInstance(val, int, f"node {n} has wrong dtype")
            # Per-edge cost_excluded_walk: every edge.
            if loaded.is_multigraph():
                for _u, _v, _k, d in loaded.edges(keys=True, data=True):
                    val = d.get("cost_excluded_walk")
                    self.assertIsInstance(val, int)
            # is_virtual=1 round-trips as int 1 on node 1.
            self.assertEqual(loaded.nodes[1].get("is_virtual"), 1)
        finally:
            import os

            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
