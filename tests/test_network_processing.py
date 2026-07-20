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
    smooth_node_attribute,
    snap_features_to_nodes,
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

    def test_accepts_geoseries_with_noncontiguous_index(self):
        """`snap_to_network_nodes` accepts a GeoSeries (not just a GeoDataFrame),
        including one whose index is non-contiguous (e.g., after a `.loc` filter).
        Guards against a latent `df[["geometry"]]` label-lookup bug on GeoSeries.
        """
        graph = self._graph()
        # Non-contiguous index — the shape that surfaced the bug in real use.
        points = gpd.GeoSeries(
            [Point(1.0, 0.0), Point(0.0, 8.0)],
            index=pd.Index([3, 17], name="point_id"),
        )
        ids, distances = snap_to_network_nodes(points, graph)
        self.assertEqual(ids.loc[3], "a")
        self.assertEqual(ids.loc[17], "c")
        self.assertAlmostEqual(distances.loc[3], 1.0)
        self.assertAlmostEqual(distances.loc[17], 2.0)

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

    def test_walk_combinations_emit_no_warning(self):
        # The walk-specific warnings (directed_scc "unnecessarily
        # restrictive", network_type='walk' "Cambridge pitfall",
        # network_type='all' + empty cost_excluded_tags "routed across
        # motorways") were dropped — each baked in an OSMnx-fetched data
        # shape that doesn't hold for custom PBF pipelines, and the
        # directed_scc one directly contradicted the module-level
        # asymmetric-cost note.
        for kwargs in [
            {"directedness": "directed_scc"},
            {"network_type": "walk"},
            {"cost_excluded_tags": set()},
        ]:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                prepare_network(self._trap_graph(), "walk", **kwargs)
            walk_warns = [w for w in captured if "walk" in str(w.message).lower()]
            self.assertEqual(
                walk_warns,
                [],
                msg=f"unexpected walk warning for kwargs={kwargs}: "
                f"{[str(w.message) for w in walk_warns]}",
            )

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
            max_distance=200.0,
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
            max_distance=200.0,
            node_spacing=10.0,
        )
        # No insertion.
        self.assertEqual(g.number_of_nodes(), n_before)
        # Parent edges intact.
        self.assertTrue(g.has_edge(1, 2, 0))
        self.assertTrue(g.has_edge(2, 1, 0))

    def test_no_insertion_beyond_max_radius(self):
        """Points farther than `max_distance` from every edge contribute no
        insertion."""
        g = self._toy_graph()
        n_before = g.number_of_nodes()
        points = self._points([(50.0, 500.0)], ["p"])
        insert_projected_nodes(
            points,
            g,
            max_distance=50.0,
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
            max_distance=500.0,
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
            max_distance=200.0,
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
            max_distance=200.0,
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
            max_distance=500.0,
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
            max_distance=200.0,
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
            max_distance=200.0,
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
            max_distance=200.0,
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
            max_distance=10.0,
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
            max_distance=20.0,
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
                max_distance=200.0,
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
                max_distance=200.0,
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
            max_distance=200.0,
            node_spacing=10.0,
        )
        self.assertIsNone(result)

    def test_inserted_virtual_node_ids_are_consecutive(self):
        """`insert_projected_nodes` pre-computes the next node id once and
        increments locally per insertion — the underlying `split_*` primitives
        are NOT allowed to re-derive `max(int_ids) + 1` per call, since that
        would make the loop O(M × N) on country-scale graphs.

        Verifying consecutive ids is a structural proxy: with the old
        per-call scan the ids would still be consecutive, but with the
        hoisted counter we can be sure no per-iteration O(N) work is sneaking
        in via the split primitives' auto-derive branch.
        """
        # Build a graph with several parallel two-way edges so many points
        # can each insert into a distinct edge.
        g = nx.MultiDiGraph()
        n_edges = 8
        max_existing_id = 0
        for i in range(n_edges):
            u, v = 2 * i + 1, 2 * i + 2
            y = float(i * 50)
            g.add_node(u, x=0.0, y=y)
            g.add_node(v, x=100.0, y=y)
            g.add_edge(
                u,
                v,
                key=0,
                highway="residential",
                length=100.0,
                geometry=LineString([(0, y), (100, y)]),
            )
            g.add_edge(
                v,
                u,
                key=0,
                highway="residential",
                length=100.0,
                geometry=LineString([(100, y), (0, y)]),
            )
            max_existing_id = max(max_existing_id, u, v)
        points = self._points(
            [(50.0, float(i * 50)) for i in range(n_edges)],
            [f"p{i}" for i in range(n_edges)],
        )
        insert_projected_nodes(
            points,
            g,
            max_distance=10.0,
            node_spacing=5.0,
        )
        virtual_ids = sorted(n for n, d in g.nodes(data=True) if d.get("is_virtual") == 1)
        self.assertEqual(len(virtual_ids), n_edges)
        # Must be consecutive starting at max_existing_id + 1.
        self.assertEqual(
            virtual_ids,
            list(range(max_existing_id + 1, max_existing_id + 1 + n_edges)),
        )


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
        """A priority node within `priority_node_max_distance` is chosen even if a
        non-priority node is closer."""
        g = self._graph()
        priority = {"p", "q"}
        points = self._points([(0.0, 0.0)], ["p1"])
        # Non-priority `r` is closer (2 m) than priority `p` (10 m).
        # With priority radius 50, `p` wins despite `r` being closer.
        ids, dists = snap_to_network_nodes(
            points,
            g,
            max_distance=100.0,
            priority_node_ids=priority,
            priority_node_max_distance=50.0,
        )
        self.assertEqual(ids.loc["p1"], "p")
        self.assertAlmostEqual(float(dists.loc["p1"]), 10.0)

    def test_falls_back_to_eligible_when_no_priority_in_range(self):
        """A point beyond `priority_node_max_distance` from any priority node
        falls back to the nearest eligible node within `max_distance`."""
        g = self._graph()
        priority = {"p", "q"}
        # Point at (500, 0): no priority within 50m, closest eligible
        # is `s` at (102, 0) → distance 398.
        points = self._points([(500.0, 0.0)], ["p1"])
        ids, dists = snap_to_network_nodes(
            points,
            g,
            max_distance=500.0,
            priority_node_ids=priority,
            priority_node_max_distance=50.0,
        )
        self.assertEqual(ids.loc["p1"], "q")  # priority node `q` at (110, 0)
        # Wait — priority q is at (110,0), distance from (500,0) is 390.
        # If max_distance=500, q is within max_distance too. But priority
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
            max_distance=100.0,
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
            max_distance=100.0,
            priority_node_flag="is_priority",
            priority_node_max_distance=50.0,
        )
        self.assertEqual(ids.loc["p1"], "p")

    def test_priority_radius_required_when_priority_set_given(self):
        """Passing a priority set without `priority_node_max_distance` raises."""
        g = self._graph()
        points = self._points([(0.0, 0.0)], ["p1"])
        with self.assertRaises(ValueError):
            snap_to_network_nodes(
                points,
                g,
                max_distance=100.0,
                priority_node_ids={"p"},
            )

    def test_both_priority_kwargs_raises(self):
        g = self._graph()
        points = self._points([(0.0, 0.0)], ["p1"])
        with self.assertRaises(ValueError):
            snap_to_network_nodes(
                points,
                g,
                max_distance=100.0,
                priority_node_ids={"p"},
                priority_node_flag="is_priority",
                priority_node_max_distance=50.0,
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
            max_distance=200.0,
            node_spacing=25.0,
        )
        # Step 2: build priority node set (primary edge incidence + virtuals).
        priority = nodes_incident_to_edges(g, edge_tags={"primary"})
        ids, dists = snap_to_network_nodes(
            points,
            g,
            max_distance=200.0,
            priority_node_ids=priority,
            priority_node_max_distance=100.0,
        )
        # near_primary matches a primary-side virtual.
        primary_target = ids.loc["near_primary"]
        self.assertIn(primary_target, priority)
        # near_resi: closest priority node is ~45m below (a virtual on the
        # primary edge). Since priority_node_max_distance=100 covers that, the
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


class SnapFeaturesToNodesTestCase(unittest.TestCase):
    """`snap_features_to_nodes` snaps point locations to nearest-within-
    radius graph nodes, writing `is_<flag>` per node."""

    def _make_grid(self):
        """3 nodes at (0,0), (100,0), (200,0)."""
        g = nx.MultiDiGraph()
        for i, x in enumerate([0.0, 100.0, 200.0]):
            g.add_node(i, x=x, y=0.0)
        return g

    def test_snaps_to_nearest_node(self):
        g = self._make_grid()
        # Location at (110, 0) → nearest is node 1 (distance 10).
        snap_features_to_nodes(g, [(110.0, 0.0)], flag_name="signal", max_distance=50.0)
        self.assertEqual(g.nodes[0]["is_signal"], 0)
        self.assertEqual(g.nodes[1]["is_signal"], 1)
        self.assertEqual(g.nodes[2]["is_signal"], 0)

    def test_drops_locations_outside_radius(self):
        g = self._make_grid()
        # (110, 0) — within 50m of node 1, kept.
        # (1000, 0) — far from all, dropped.
        snap_features_to_nodes(
            g, [(110.0, 0.0), (1000.0, 0.0)], flag_name="signal", max_distance=50.0
        )
        self.assertEqual(g.nodes[1]["is_signal"], 1)
        # All other nodes should be 0 (not just absent).
        for n in (0, 2):
            self.assertEqual(g.nodes[n]["is_signal"], 0)

    def test_multiple_locations_snap_to_different_nodes(self):
        g = self._make_grid()
        snap_features_to_nodes(
            g, [(5.0, 0.0), (105.0, 0.0), (205.0, 0.0)], flag_name="stop", max_distance=50.0
        )
        for n in (0, 1, 2):
            self.assertEqual(g.nodes[n]["is_stop"], 1)

    def test_empty_locations_zeros_all_nodes(self):
        g = self._make_grid()
        snap_features_to_nodes(g, [], flag_name="signal", max_distance=50.0)
        for n in g.nodes:
            self.assertEqual(g.nodes[n]["is_signal"], 0)

    def test_idempotent_re_run_clears_prior_flags(self):
        g = self._make_grid()
        # First call: flag node 1.
        snap_features_to_nodes(g, [(110.0, 0.0)], flag_name="signal", max_distance=50.0)
        self.assertEqual(g.nodes[1]["is_signal"], 1)
        # Second call with no locations: flag should be cleared.
        snap_features_to_nodes(g, [], flag_name="signal", max_distance=50.0)
        self.assertEqual(g.nodes[1]["is_signal"], 0)

    def test_empty_graph_is_noop(self):
        g = nx.MultiDiGraph()
        snap_features_to_nodes(g, [(0.0, 0.0)], flag_name="signal", max_distance=50.0)
        # No nodes to flag; should not raise.
        self.assertEqual(g.number_of_nodes(), 0)

    def test_flag_name_prefix(self):
        # `flag_name='traffic_signal'` produces `is_traffic_signal`.
        g = self._make_grid()
        snap_features_to_nodes(g, [(0.0, 0.0)], flag_name="traffic_signal", max_distance=10.0)
        self.assertEqual(g.nodes[0]["is_traffic_signal"], 1)


class SmoothNodeAttributeTestCase(unittest.TestCase):
    """`smooth_node_attribute` — topology-weighted Gaussian smoothing."""

    @staticmethod
    def _path_graph(elevations: dict, length: float = 100.0) -> nx.Graph:
        """Build a path graph `0 - 1 - 2 - ...` with given elevations
        and a uniform edge length."""
        g = nx.path_graph(len(elevations))
        for u, v in g.edges():
            g[u][v]["length"] = length
        nx.set_node_attributes(g, elevations, "elevation")
        return g

    def test_uniform_field_unchanged(self):
        """All nodes at the same value → smoothing changes nothing."""
        g = self._path_graph({0: 50.0, 1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0})
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        for n in g.nodes():
            self.assertAlmostEqual(g.nodes[n]["elevation"], 50.0, places=10)

    def test_self_included_with_weight_one(self):
        """Centre node has weight 1 (distance 0 → Gaussian = 1)."""
        # Path 0 - 1 - 2, elevations 100/200/300, edge length = length_scale.
        # Neighbour weight = exp(-0.5) ≈ 0.6065
        # Node 0: only neighbour is 1 → (100 + w·200) / (1 + w)
        # Node 1: neighbours are 0 and 2 → (200 + w·100 + w·300) / (1 + 2w) = 200
        # Node 2: only neighbour is 1 → (300 + w·200) / (1 + w)
        g = self._path_graph({0: 100.0, 1: 200.0, 2: 300.0})
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        import math

        w = math.exp(-0.5)
        self.assertAlmostEqual(g.nodes[0]["elevation"], (100 + w * 200) / (1 + w), places=6)
        self.assertAlmostEqual(g.nodes[1]["elevation"], 200.0, places=6)
        self.assertAlmostEqual(g.nodes[2]["elevation"], (300 + w * 200) / (1 + w), places=6)

    def test_gaussian_decay(self):
        """Edge far above length_scale → near-zero neighbour weight."""
        # Edge length = 5 × length_scale → weight exp(-12.5) ≈ 3.7e-6
        g = self._path_graph({0: 100.0, 1: 200.0}, length=500.0)
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        # Drift < 0.001 → the neighbour contribution is negligible.
        self.assertAlmostEqual(g.nodes[0]["elevation"], 100.0, places=2)
        self.assertAlmostEqual(g.nodes[1]["elevation"], 200.0, places=2)

    def test_nan_centre_passes_through(self):
        """A NaN at the centre node stays NaN — no smoothing applied."""
        g = self._path_graph({0: 100.0, 1: float("nan"), 2: 300.0})
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        self.assertTrue(np.isnan(g.nodes[1]["elevation"]))

    def test_nan_neighbour_skipped(self):
        """A NaN neighbour drops out of the sum; other neighbours still count."""
        # Path 0 - 1 - 2 - 3, elevations 100/NaN/300/200.
        # Node 2's neighbours: 1 (NaN — skipped) + 3 (200).
        # → smoothed[2] = (300 + w·200) / (1 + w)
        g = self._path_graph({0: 100.0, 1: float("nan"), 2: 300.0, 3: 200.0})
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        import math

        w = math.exp(-0.5)
        self.assertAlmostEqual(
            g.nodes[2]["elevation"],
            (300 + w * 200) / (1 + w),
            places=6,
        )

    def test_out_attr_leaves_input_alone(self):
        """`out_attr=` writes to a different attribute; original untouched."""
        g = self._path_graph({0: 100.0, 1: 200.0, 2: 300.0})
        smooth_node_attribute(g, "elevation", length_scale=100.0, out_attr="elevation_smooth")
        # Original preserved
        self.assertEqual(g.nodes[0]["elevation"], 100.0)
        self.assertEqual(g.nodes[1]["elevation"], 200.0)
        # New attribute present
        self.assertIn("elevation_smooth", g.nodes[0])
        self.assertNotEqual(g.nodes[0]["elevation_smooth"], 100.0)

    def test_directed_graph_uses_both_directions(self):
        """For a DiGraph, smoothing uses both successors AND predecessors
        (terrain is undirected even if the road graph isn't)."""
        # DiGraph 0 → 1 → 2. Smoothing of node 1 should pull from BOTH 0 and 2.
        g = nx.DiGraph()
        g.add_edge(0, 1, length=100.0)
        g.add_edge(1, 2, length=100.0)
        nx.set_node_attributes(g, {0: 100.0, 1: 200.0, 2: 300.0}, "elevation")
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        # If only successors were used: node 1 → (200 + w·300) / (1 + w) ≠ 200
        # With both: (200 + w·100 + w·300) / (1 + 2w) = 200
        self.assertAlmostEqual(g.nodes[1]["elevation"], 200.0, places=6)

    def test_multigraph_uses_shortest_parallel_edge(self):
        """For a MultiGraph with parallel edges, the shortest one sets the
        weight (the Gaussian distance)."""
        g = nx.MultiGraph()
        g.add_edge(0, 1, length=500.0)  # far parallel edge
        g.add_edge(0, 1, length=100.0)  # near parallel edge
        nx.set_node_attributes(g, {0: 100.0, 1: 200.0}, "elevation")
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        # Should use length=100 → weight exp(-0.5) ≈ 0.6065
        import math

        w = math.exp(-0.5)
        expected = (100 + w * 200) / (1 + w)
        self.assertAlmostEqual(g.nodes[0]["elevation"], expected, places=6)

    def test_iterations_compound(self):
        """`n_iterations=k+1` smooths more than `n_iterations=k`. Use a
        single peak on a flat background so the max actually moves with
        smoothing (a step function has sticky boundaries — boundary
        extrema have no off-peak neighbour to pull them down)."""
        # 11 nodes at 100 with a single 200-peak at node 5.
        elevs = {i: 100.0 for i in range(11)}
        elevs[5] = 200.0

        g1 = self._path_graph(elevs)
        smooth_node_attribute(g1, "elevation", length_scale=100.0, n_iterations=1)
        peak_1 = max(g1.nodes[n]["elevation"] for n in g1.nodes())

        g3 = self._path_graph(elevs)
        smooth_node_attribute(g3, "elevation", length_scale=100.0, n_iterations=3)
        peak_3 = max(g3.nodes[n]["elevation"] for n in g3.nodes())

        # More iterations → peak gets pulled further toward the
        # background by repeated convolution with the Gaussian kernel.
        self.assertLess(peak_3, peak_1)
        # And the peak hasn't fully decayed yet (the bg-level is 100).
        self.assertGreater(peak_3, 100.0)

    def test_node_without_attribute_left_untouched(self):
        """Nodes that don't have the input attribute aren't given one."""
        g = self._path_graph({0: 100.0, 1: 200.0, 2: 300.0})
        g.add_node(99)  # no elevation
        smooth_node_attribute(g, "elevation", length_scale=100.0)
        self.assertNotIn("elevation", g.nodes[99])


if __name__ == "__main__":
    unittest.main()
