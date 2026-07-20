"""Snap-target resolution on transport networks.

This module separates two questions that the previous tiered design conflated:

- **Where should new graph nodes be introduced so that snap targets are
  available where points need them?** Answered by `insert_projected_nodes`
  (mutates the graph by inserting virtual nodes onto eligible edges).
- **Given a graph (enriched with new nodes or not), which node does each point
  match to?** Answered by `snap_to_network_nodes` (read-only; optionally
  two-tier with a priority node set for "prefer main-road nodes" semantics).

The typical workflow on a study region is one or more `insert_projected_nodes`
calls (e.g., one for cells with a wide edge filter, one for zone centroids
restricted to main-road tags) followed by `snap_to_network_nodes` per point
layer that needs a `(point → node_id)` mapping. The enriched graph is reusable
across many layers; the snap function never mutates it.

Other primitives in this module:

- `split_edge_at_point` and `split_two_way_edge_at_point` — the edge-split
  primitives used by `insert_projected_nodes`. Exposed for advanced callers that
  want to drive insertions directly.
- `transport_centroid` — polygon-aware helper that produces a representative
  point per polygon (the median of contained eligible nodes) before snapping.
- `nodes_incident_to_edges` — derive a node set (typically the "priority" node
  set for `snap_to_network_nodes`) from an edge tag predicate.

Eligibility / priority node sets that this module's functions consume are
typically produced by `aperta.routing_prep` — `compute_snap_eligibility`
(prep-stage, returns just the eligible frozenset + cost-mask flag name)
or `prepare_network` (consumer-stage, returns a full `PreparedGraph` with
the same eligible set plus per-node `is_snap_eligible_<mode>` decorations).
Priority sets are typically derived from edge tags via the local
`nodes_incident_to_edges` helper. This module does not depend on
`routing_prep` — its functions accept node sets (or per-node
flag-attribute names) as plain arguments.
"""

from collections import namedtuple
from typing import Callable, Iterable, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

# -----------------------------------------------------------------------------
# Edge-split primitives
# -----------------------------------------------------------------------------

# Result of `split_edge_at_point`: the new node id + the two child-edge keys.
# For multigraphs the child key is `(u, v, k)`; for simple graphs `(u, v)`.
SplitEdgeResult = namedtuple("SplitEdgeResult", "new_node_id child1 child2")


def split_edge_at_point(
    graph: nx.Graph,
    u,
    v,
    point: tuple[float, float] | Point,
    *,
    k=None,
    new_node_id=None,
    extra_node_attrs: dict | None = None,
) -> SplitEdgeResult:
    """Insert a synthetic node on edge `(u, v[, k])` at the projection of
    `point` onto the edge geometry; split the parent edge into two children.

    **Single-direction split.** For two-way OSM roads (both `(u, v, k_fwd)`
    and `(v, u, k_rev)` exist), use `split_two_way_edge_at_point` instead —
    it handles symmetric routing through one shared virtual node. Use this
    function directly only for genuinely one-way roads or when asymmetric
    routing is intended.

    Load-bearing primitive of `insert_projected_nodes`: a point with no
    eligible node within `node_spacing` of its projection on the nearest
    candidate edge gets a virtual node here, on that edge, so the next
    `snap_to_network_nodes` call can match it.

    Per-edge attribute split semantics:

    - `geometry`: sliced into two LineStrings via shapely `substring` at the
      projected fractional position. Both children inherit valid LineStrings.
    - `length`: recomputed as `float(child_geom.length)` on each child
      (matches the `consolidate_intersections` convention).
    - All other per-edge attributes: **copied verbatim to both children.**
      Correct for OSM-direct properties (`highway`, `lanes`, `oneway`,
      `junction`, `cycleway*`, `maxspeed`) and for the per-edge cost-mask
      flag (`cost_excluded_<mode>`), which depend only on the road class
      (uniform along the parent). Attributes that DO require recomputation
      after split (`elev_gain` / `elev_loss`, endpoint-mean-propagated
      intersection flags, `density_norm` on edges) are expected to be
      (re)derived AFTER insertion by the post-insertion enrichment passes
      (cf. snap-across-barrier 7-step workflow).

    Per-node attributes on the new synthetic node:

    - `x`, `y`: from the projected point on the parent geometry (so the
      new node sits exactly on the road).
    - Any caller-supplied `extra_node_attrs`. Topology / classification
      flags (`n_streets`, `is_t_junction`, `is_4way`, the `_major` /
      `_anchor` variants, `max/min_highway_rank`) are NOT computed here;
      they belong to `flag_node_intersection_topology` /
      `flag_node_osm_classification`, run on the augmented graph by the
      post-insertion `prepare_network` pass.

    Args:
        graph: networkx `Graph` / `MultiGraph` / `DiGraph` / `MultiDiGraph`.
            For multigraphs, `k` is required; for simple graphs `k` must be
            None. Directional edges (`DiGraph` / `MultiDiGraph`) are split
            preserving direction: child 1 is `(u, new)`, child 2 is
            `(new, v)`. A reverse edge `(v, u)` is NOT touched; the caller
            must split it separately if a two-way road is intended.
        u, v: endpoints of the parent edge.
        point: the snap-target point. Projected onto the parent geometry to
            find the split fraction; the projection (not the input point)
            becomes the new node's `(x, y)`. Accepts `(x, y)` tuple or
            shapely `Point`.
        k: edge key for multigraphs.
        new_node_id: optional explicit id for the synthetic node. If None,
            auto-derived as `max(int_node_ids) + 1` (raises if the graph
            has no int-typed node ids — pass `new_node_id` explicitly).
        extra_node_attrs: optional dict of additional attributes to set on
            the new node beyond `x` / `y`.

    Returns:
        `SplitEdgeResult(new_node_id, child1_edge_key, child2_edge_key)`.

    Raises:
        ValueError: if the edge doesn't exist, `k` is mis-supplied for the
            graph type, the parent edge has no `geometry`, the projection
            falls at or beyond an endpoint (caller should snap to that
            endpoint instead), or `new_node_id` collides with an existing
            node.
    """
    is_multi = graph.is_multigraph()
    if is_multi and k is None:
        raise ValueError("`k` is required when `graph` is a multigraph.")
    if not is_multi and k is not None:
        raise ValueError("`k` must be None when `graph` is not a multigraph.")

    if is_multi:
        if not graph.has_edge(u, v, k):
            raise ValueError(f"Edge ({u!r}, {v!r}, {k!r}) not in graph.")
        parent_data = dict(graph.edges[u, v, k])
    else:
        if not graph.has_edge(u, v):
            raise ValueError(f"Edge ({u!r}, {v!r}) not in graph.")
        parent_data = dict(graph.edges[u, v])

    parent_geom = parent_data.get("geometry")
    if parent_geom is None:
        raise ValueError(
            f"Edge ({u!r}, {v!r}{', ' + repr(k) if k is not None else ''}) has no "
            f"`geometry` attribute. Consolidated OSMnx graphs satisfy this by "
            f"construction; for raw graphs, run `osmnx.utils_graph.add_edge_geometry` "
            f"or pass through `consolidate_intersections` first."
        )
    if not isinstance(parent_geom, LineString):
        raise ValueError(
            f"Edge ({u!r}, {v!r}) geometry is {type(parent_geom).__name__}, expected LineString."
        )

    # Project the target onto the parent geometry to find the split fraction.
    if not isinstance(point, Point):
        point = Point(point[0], point[1])
    proj_dist = parent_geom.project(point)
    edge_length = parent_geom.length
    # Refuse degenerate splits at or past the endpoints. The right behaviour
    # in that case is "snap directly to the endpoint node" — but that's a
    # caller decision, not this primitive's concern.
    if proj_dist <= 0.0 or proj_dist >= edge_length:
        raise ValueError(
            f"Projection of point ({point.x}, {point.y}) onto edge geometry falls "
            f"at or beyond an endpoint (proj_dist={proj_dist:.4f}, "
            f"edge_length={edge_length:.4f}). Caller should snap to the nearest "
            f"endpoint node instead of inserting a virtual node."
        )
    proj_point = parent_geom.interpolate(proj_dist)

    if new_node_id is None:
        int_ids = [n for n in graph.nodes if isinstance(n, (int, np.integer))]
        if not int_ids:
            raise ValueError(
                "Cannot auto-derive `new_node_id`: graph has no int-typed node ids. "
                "Pass `new_node_id` explicitly."
            )
        new_node_id = int(max(int_ids)) + 1
    elif new_node_id in graph.nodes:
        raise ValueError(f"`new_node_id={new_node_id!r}` already exists in graph.")

    node_attrs = {"x": float(proj_point.x), "y": float(proj_point.y)}
    if extra_node_attrs:
        node_attrs.update(extra_node_attrs)
    graph.add_node(new_node_id, **node_attrs)

    return _apply_edge_split(graph, u, v, k, new_node_id, proj_dist, parent_geom)


def _apply_edge_split(
    graph: nx.Graph,
    u,
    v,
    k,
    new_node_id,
    proj_dist: float,
    parent_geom: LineString,
) -> SplitEdgeResult:
    """Internal: replace edge `(u, v[, k])` with two children that pass through
    `new_node_id` at `proj_dist` along `parent_geom`. Assumes:
      - `new_node_id` is already in the graph (caller created it)
      - `proj_dist` is in (0, parent_geom.length) — already validated
      - parent edge exists and has `parent_geom` as its geometry

    Used by both `split_edge_at_point` (creates node, splits one edge) and
    `split_two_way_edge_at_point` (creates node once, splits both directions
    through it).
    """
    is_multi = graph.is_multigraph()
    if is_multi:
        parent_data = dict(graph.edges[u, v, k])
    else:
        parent_data = dict(graph.edges[u, v])

    edge_length = parent_geom.length
    child1_geom = substring(parent_geom, 0.0, proj_dist)
    child2_geom = substring(parent_geom, proj_dist, edge_length)

    def _child_attrs(child_geom):
        attrs = {kk: vv for kk, vv in parent_data.items() if kk not in ("geometry", "length")}
        attrs["geometry"] = child_geom
        attrs["length"] = float(child_geom.length)
        return attrs

    if is_multi:
        graph.remove_edge(u, v, k)
        k1 = graph.add_edge(u, new_node_id, **_child_attrs(child1_geom))
        k2 = graph.add_edge(new_node_id, v, **_child_attrs(child2_geom))
        return SplitEdgeResult(new_node_id, (u, new_node_id, k1), (new_node_id, v, k2))
    graph.remove_edge(u, v)
    graph.add_edge(u, new_node_id, **_child_attrs(child1_geom))
    graph.add_edge(new_node_id, v, **_child_attrs(child2_geom))
    return SplitEdgeResult(new_node_id, (u, new_node_id), (new_node_id, v))


# Result of `split_two_way_edge_at_point`: new node + per-direction children.
SplitTwoWayEdgeResult = namedtuple("SplitTwoWayEdgeResult", "new_node_id forward reverse")


def split_two_way_edge_at_point(
    graph: nx.MultiDiGraph,
    u,
    v,
    point: tuple[float, float] | Point,
    *,
    k_forward,
    k_reverse,
    new_node_id=None,
    extra_node_attrs: dict | None = None,
) -> SplitTwoWayEdgeResult:
    """Symmetrically split BOTH directional edges `(u, v, k_forward)` and
    `(v, u, k_reverse)` through a single shared synthetic node. Use this for
    two-way roads (typical OSMnx case) where the snap-target should be
    reachable from both directions.

    After the split: four new edges exist — `(u, new)`, `(new, v)`,
    `(v, new)`, `(new, u)` — and the original two are gone. The shared
    synthetic node implicitly allows a U-turn through it (`A → new → A`),
    which is a property of any node in the graph; shortest-path routing
    won't generate U-turns at a virtual node unless it shortens the route
    (it doesn't), so the practical impact is negligible. Use
    `split_edge_at_point` directly if you want an asymmetric split (e.g.,
    for a genuinely one-way road with no reverse edge to also split).

    The two directional edges are typically near-mirror geometries of one
    another, but not required to be exact mirrors — each direction's
    projection is computed independently against its own geometry. If they
    disagree noticeably on where the projection lands, the new node sits at
    the FORWARD direction's projection (the reverse direction's split still
    uses its own projection along its own geometry, so the two children
    summing to the parent's length is preserved on each direction
    separately).

    Args:
        graph: `nx.MultiDiGraph` (the typical OSMnx graph type).
        u, v: endpoints of the forward edge. The reverse edge is queried as
            `(v, u, k_reverse)`.
        point: snap-target point, used as the projection target on both
            geometries. The forward direction's projection determines the
            new node's `(x, y)`.
        k_forward: edge key of the forward edge `(u, v)`.
        k_reverse: edge key of the reverse edge `(v, u)`.
        new_node_id: optional explicit id for the synthetic node (else
            auto-derived as `max(int_node_ids) + 1`).
        extra_node_attrs: optional additional attributes for the new node.

    Returns:
        `SplitTwoWayEdgeResult(new_node_id, forward, reverse)` where
        `forward` and `reverse` are each a `SplitEdgeResult` describing the
        two children of that direction.

    Raises:
        ValueError: if `graph` isn't a directed multigraph, either edge is
            missing or lacks geometry, projection on either direction falls
            at/beyond an endpoint, or `new_node_id` collides with an
            existing node.
    """
    if not (graph.is_multigraph() and graph.is_directed()):
        raise ValueError(
            "`split_two_way_edge_at_point` requires a directed MultiDiGraph. "
            "For undirected graphs, call `split_edge_at_point` directly."
        )
    if not graph.has_edge(u, v, k_forward):
        raise ValueError(f"Forward edge ({u!r}, {v!r}, {k_forward!r}) not in graph.")
    if not graph.has_edge(v, u, k_reverse):
        raise ValueError(
            f"Reverse edge ({v!r}, {u!r}, {k_reverse!r}) not in graph. "
            f"If this road is genuinely one-way, use `split_edge_at_point` instead."
        )

    forward_data = graph.edges[u, v, k_forward]
    reverse_data = graph.edges[v, u, k_reverse]
    forward_geom = forward_data.get("geometry")
    reverse_geom = reverse_data.get("geometry")
    if forward_geom is None:
        raise ValueError(f"Forward edge ({u!r}, {v!r}, {k_forward!r}) has no `geometry`.")
    if reverse_geom is None:
        raise ValueError(f"Reverse edge ({v!r}, {u!r}, {k_reverse!r}) has no `geometry`.")
    if not isinstance(forward_geom, LineString) or not isinstance(reverse_geom, LineString):
        raise ValueError("Both directions' geometry must be LineString.")

    if not isinstance(point, Point):
        point = Point(point[0], point[1])
    proj_fwd = forward_geom.project(point)
    proj_rev = reverse_geom.project(point)
    if proj_fwd <= 0.0 or proj_fwd >= forward_geom.length:
        raise ValueError(
            f"Projection of point ({point.x}, {point.y}) onto forward edge falls "
            f"at or beyond an endpoint (proj_dist={proj_fwd:.4f}, "
            f"edge_length={forward_geom.length:.4f}). Snap to the nearest endpoint "
            f"node instead."
        )
    if proj_rev <= 0.0 or proj_rev >= reverse_geom.length:
        raise ValueError(
            f"Projection of point ({point.x}, {point.y}) onto reverse edge falls "
            f"at or beyond an endpoint (proj_dist={proj_rev:.4f}, "
            f"edge_length={reverse_geom.length:.4f}). Snap to the nearest endpoint "
            f"node instead."
        )

    # New node location: forward direction's projection.
    proj_point = forward_geom.interpolate(proj_fwd)

    if new_node_id is None:
        int_ids = [n for n in graph.nodes if isinstance(n, (int, np.integer))]
        if not int_ids:
            raise ValueError(
                "Cannot auto-derive `new_node_id`: graph has no int-typed node ids. "
                "Pass `new_node_id` explicitly."
            )
        new_node_id = int(max(int_ids)) + 1
    elif new_node_id in graph.nodes:
        raise ValueError(f"`new_node_id={new_node_id!r}` already exists in graph.")

    node_attrs = {"x": float(proj_point.x), "y": float(proj_point.y)}
    if extra_node_attrs:
        node_attrs.update(extra_node_attrs)
    graph.add_node(new_node_id, **node_attrs)

    forward_result = _apply_edge_split(graph, u, v, k_forward, new_node_id, proj_fwd, forward_geom)
    reverse_result = _apply_edge_split(graph, v, u, k_reverse, new_node_id, proj_rev, reverse_geom)
    return SplitTwoWayEdgeResult(new_node_id, forward_result, reverse_result)


# -----------------------------------------------------------------------------
# Node snap
# -----------------------------------------------------------------------------


def _snap_to_subset(
    points: gpd.GeoDataFrame | gpd.GeoSeries,
    graph: nx.Graph,
    node_ids: list,
    *,
    max_distance: float | None,
) -> tuple[pd.Series, pd.Series]:
    """Helper: snap each point to its nearest node in `node_ids`. Returns
    all-NaN series if `node_ids` is empty (rather than raising)."""
    from aperta import geo_mapping  # local import to avoid module-load cycle

    if not node_ids:
        nan_ids = pd.Series([pd.NA] * len(points), index=points.index, dtype=object)
        nan_dists = pd.Series([float("nan")] * len(points), index=points.index, dtype=float)
        return nan_ids, nan_dists
    node_x = [graph.nodes[n]["x"] for n in node_ids]
    node_y = [graph.nodes[n]["y"] for n in node_ids]
    nodes_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(node_x, node_y),
        index=pd.Index(node_ids, name="node_id"),
        crs=points.crs,
    )
    return geo_mapping.map_points_to_points(points, nodes_gdf, max_distance=max_distance)


def snap_to_network_nodes(
    points: gpd.GeoDataFrame | gpd.GeoSeries,
    graph: nx.Graph,
    *,
    max_distance: float | None = None,
    eligible_node_ids: set | list | pd.Index | None = None,
    eligible_node_flag: str | None = None,
    priority_node_ids: set | list | pd.Index | None = None,
    priority_node_flag: str | None = None,
    priority_node_max_distance: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Snap each row in `points` to its nearest node in `graph`.

    Two-tier when a priority node set is given:

    1. **Tier 1 — priority node snap**: if a priority set is provided
       (via `priority_node_ids` or `priority_node_flag`), snap to the
       nearest priority node within `priority_node_max_distance`. The priority
       set is typically the nodes incident to one or more high-class
       edges (e.g., main roads); the helper `nodes_incident_to_edges`
       derives such a set from an edge predicate.
    2. **Tier 2 — eligible node snap (fallback)**: for unmatched points,
       snap to the nearest eligible node within `max_distance`. Points
       beyond `max_distance` from every eligible node return NaN.

    When no priority set is given (default), the function collapses to a
    single-pass eligible-node snap within `max_distance` — bit-identical to
    the historical single-tier behavior.

    This function does NOT mutate `graph` and does NOT insert nodes.
    For node introduction (virtual-node insertion onto edges), call
    `insert_projected_nodes` first to enrich the graph; then call this
    function for the actual snap. The two-step pattern lets the enriched
    graph be reused for many subsequent point-set snaps cheaply.

    Network nodes must carry `x` and `y` attributes (aperta convention).
    Point CRS and the graph coordinates must already agree — no reprojection.

    Args:
        points: GeoDataFrame or GeoSeries of points to snap. Output is
            indexed by `points.index`.
        graph: NetworkX (or compatible) graph with `x` / `y` node attributes.
        max_distance: optional cap on tier-2 (fallback) snap distance. `None`
            means no cap. Points farther than this from every eligible node
            return NaN.
        eligible_node_ids: optional restriction on the tier-2 (fallback)
            candidate set. `None` considers all graph nodes. Mutually
            exclusive with `eligible_node_flag`.
        eligible_node_flag: per-node bool attribute name as an alternative
            to `eligible_node_ids` (e.g., `snap_eligible_<mode>` from
            `prepare_network`).
        priority_node_ids: optional priority set for tier-1 snap. If given,
            points within `priority_node_max_distance` of a priority node snap
            to that node; unmatched points fall through to tier 2. Mutually
            exclusive with `priority_node_flag`.
        priority_node_flag: per-node bool attribute name as an alternative
            to `priority_node_ids`.
        priority_node_max_distance: tier-1 search radius. Required when a priority
            set is given; ignored otherwise. Typically equal to or larger
            than the spacing used by `insert_projected_nodes` so that virtual
            nodes inserted on priority edges are found.

    Returns:
        Tuple `(node_ids, distances)`:
            - `node_ids`: `pd.Series` of nearest-node IDs, indexed by `points.index`.
            - `distances`: `pd.Series` of distances (in CRS units), indexed by
              `points.index`.
    """
    if priority_node_ids is not None and priority_node_flag is not None:
        raise ValueError("Pass either `priority_node_ids` or `priority_node_flag`, not both.")
    if eligible_node_ids is not None and eligible_node_flag is not None:
        raise ValueError("Pass either `eligible_node_ids` or `eligible_node_flag`, not both.")

    # Resolve eligible set (tier-2 candidates).
    if eligible_node_ids is None and eligible_node_flag is not None:
        eligible_node_ids = [
            n for n, data in graph.nodes(data=True) if data.get(eligible_node_flag)
        ]
    if eligible_node_ids is None:
        eligible_node_list = list(graph.nodes)
    else:
        eligible_set = set(eligible_node_ids)
        eligible_node_list = [n for n in graph.nodes if n in eligible_set]
        if not eligible_node_list:
            raise ValueError(
                "Eligibility filter excluded every node in the graph "
                "(`eligible_node_ids` or `eligible_node_flag`). Cannot snap to "
                "an empty set of targets."
            )

    # Resolve priority set (tier-1 candidates).
    if priority_node_ids is None and priority_node_flag is not None:
        priority_node_ids = [
            n for n, data in graph.nodes(data=True) if data.get(priority_node_flag)
        ]
    priority_active = priority_node_ids is not None
    if priority_active and priority_node_max_distance is None:
        raise ValueError("`priority_node_max_distance` is required when a priority set is given.")
    if priority_active:
        # Hoist set construction OUT of the comprehension — leaving it inline
        # rebuilds the set per iteration, making this O(|eligible| × |priority|).
        # At country scale that's ~3M × ~50k = ~150 B ops; hours.
        priority_set = set(priority_node_ids)  # type: ignore[arg-type]
        priority_node_list = [n for n in eligible_node_list if n in priority_set]
    else:
        priority_node_list = []

    # Single-pass case — no priority set, just tier-2.
    if not priority_active:
        return _snap_to_subset(points, graph, eligible_node_list, max_distance=max_distance)

    # Two-pass case — tier 1 priority then tier 2 eligible for the unmatched.
    p_ids, p_dists = _snap_to_subset(
        points, graph, priority_node_list, max_distance=priority_node_max_distance
    )
    matched_mask = p_ids.notna()
    if matched_mask.all():
        return p_ids, p_dists

    unmatched = points.loc[~matched_mask]
    e_ids, e_dists = _snap_to_subset(
        unmatched, graph, eligible_node_list, max_distance=max_distance
    )

    # Combine: priority hits win; eligible fills the rest.
    out_ids = p_ids.copy()
    out_dists = p_dists.copy()
    out_ids.loc[~matched_mask] = e_ids
    out_dists.loc[~matched_mask] = e_dists
    return out_ids, out_dists


def nodes_incident_to_edges(
    graph: nx.Graph,
    edge_tags: Iterable[str] | Callable[[dict], bool] | None,
) -> set:
    """Return the set of node IDs incident to at least one edge matching
    `edge_tags`.

    Common use: derive the priority-node set for the two-tier
    `snap_to_network_nodes`. Pass the same tag set used to define
    "priority edges" upstream (e.g., `{'primary', 'secondary',
    'tertiary'}` for car main-road analyses); the returned set is the
    union of endpoints of all such edges in the graph.

    Args:
        graph: NetworkX graph. Multigraphs are walked with `keys=True`.
        edge_tags: either an iterable of OSM `highway` tag strings (matched
            against the edge's `highway` attribute, tolerating list-valued
            tags) OR a callable `(edge_data) -> bool`. `None` returns the
            empty set.

    Returns:
        Set of node IDs.
    """
    predicate = _resolve_priority_predicate(edge_tags)
    if predicate is None:
        return set()
    out: set = set()
    if graph.is_multigraph():
        for u, v, _k, d in graph.edges(keys=True, data=True):
            if predicate(d):
                out.add(u)
                out.add(v)
    else:
        for u, v, d in graph.edges(data=True):
            if predicate(d):
                out.add(u)
                out.add(v)
    return out


# -----------------------------------------------------------------------------
# Virtual-node insertion (point-driven edge splitting; companion to the node snap)
# -----------------------------------------------------------------------------


def _resolve_priority_predicate(
    priority_edge_tags: Iterable[str] | Callable[[dict], bool] | None,
) -> Callable[[dict], bool] | None:
    """Normalise the `priority_edge_tags` arg to a per-edge predicate or None.

    Accepts either a callable (used as-is) or an iterable of OSM `highway`
    tag strings (wrapped in a predicate that checks the edge's `highway`
    attribute, tolerating list-valued tags).
    """
    if priority_edge_tags is None:
        return None
    if callable(priority_edge_tags):
        return priority_edge_tags
    tag_set = frozenset(priority_edge_tags)
    if not tag_set:
        return None

    def _osm_tag_predicate(d: dict) -> bool:
        tag = d.get("highway")
        if isinstance(tag, list):
            return any(t in tag_set for t in tag)
        return tag in tag_set

    return _osm_tag_predicate


# Highway classes EXCLUDED by `insert_projected_nodes`'s default edge filter.
# Snapping a cell / building centroid to a motorway shoulder is rarely intended;
# if a caller wants to insert virtuals on these classes, they pass an explicit
# `edge_filter`. Linked variants are included so e.g. `motorway_link` ramps
# don't slip through.
DEFAULT_EXCLUDED_HIGHWAY_TAGS = frozenset({"motorway", "motorway_link", "trunk", "trunk_link"})


def _default_edge_filter_predicate(d: dict) -> bool:
    """Default edge filter for `insert_projected_nodes`: allow all edges
    except those tagged with one of `DEFAULT_EXCLUDED_HIGHWAY_TAGS`."""
    tag = d.get("highway")
    if isinstance(tag, list):
        return not any(t in DEFAULT_EXCLUDED_HIGHWAY_TAGS for t in tag)
    return tag not in DEFAULT_EXCLUDED_HIGHWAY_TAGS


def _resolve_edge_filter(
    edge_filter: Iterable[str] | Callable[[dict], bool] | None,
) -> Callable[[dict], bool]:
    """Normalise the `edge_filter` arg for `insert_projected_nodes`.

    - `None` → default predicate (exclude motorway / trunk).
    - `Iterable[str]` → allowlist predicate on the `highway` tag
      (empty iterable = "nothing allowed", an explicit override).
    - `Callable` → returned as-is.
    """
    if edge_filter is None:
        return _default_edge_filter_predicate
    if callable(edge_filter):
        return edge_filter
    tag_set = frozenset(edge_filter)

    def _allowlist_predicate(d: dict) -> bool:
        tag = d.get("highway")
        if isinstance(tag, list):
            return any(t in tag_set for t in tag)
        return tag in tag_set

    return _allowlist_predicate


def _build_edge_strtree(
    graph: nx.Graph,
    eligible_node_set: set,
    cost_excluded_flag: str | None,
    predicate: Callable[[dict], bool] | None,
):
    """Build a shapely STRtree over edges of `graph` filtered to:
      - both endpoints in `eligible_node_set`
      - if `cost_excluded_flag`, edge's flag is not True
      - if `predicate`, edge's data satisfies it

    Returns (strtree, edge_keys, edge_geometries) where edge_keys[i] is the
    `(u, v, k)` (multigraph) or `(u, v)` (simple) tuple for geometry i. Both
    `edge_keys` and `edge_geometries` are aligned to the STRtree's internal
    index. Returns (None, [], []) if no edges survive the filter.
    """
    from shapely.strtree import STRtree

    is_multi = graph.is_multigraph()
    edge_keys: list = []
    edge_geoms: list = []
    if is_multi:
        edges_iter = graph.edges(keys=True, data=True)
        for u, v, k, d in edges_iter:
            if u not in eligible_node_set or v not in eligible_node_set:
                continue
            if cost_excluded_flag is not None and d.get(cost_excluded_flag, False):
                continue
            if predicate is not None and not predicate(d):
                continue
            geom = d.get("geometry")
            if not isinstance(geom, LineString):
                continue
            edge_keys.append((u, v, k))
            edge_geoms.append(geom)
    else:
        edges_iter = graph.edges(data=True)
        for u, v, d in edges_iter:
            if u not in eligible_node_set or v not in eligible_node_set:
                continue
            if cost_excluded_flag is not None and d.get(cost_excluded_flag, False):
                continue
            if predicate is not None and not predicate(d):
                continue
            geom = d.get("geometry")
            if not isinstance(geom, LineString):
                continue
            edge_keys.append((u, v))
            edge_geoms.append(geom)

    if not edge_geoms:
        return None, [], []
    return STRtree(edge_geoms), edge_keys, edge_geoms


def _find_nearest_edge(
    strtree,
    edge_keys: list,
    edge_geoms: list,
    point: Point,
    max_distance: float | None,
):
    """Query `strtree` for the nearest edge to `point`. Returns
    `(edge_key, geom, dist)` if the nearest edge is within `max_distance`
    (or `max_distance is None`), else `None`.
    """
    if strtree is None or not edge_geoms:
        return None
    # shapely 2.x: query_nearest returns array of indices; with max_distance,
    # returns empty if nothing within.
    if max_distance is not None:
        idxs = strtree.query_nearest(point, max_distance=max_distance)
    else:
        idxs = strtree.query_nearest(point)
    if len(idxs) == 0:
        return None
    # Multiple candidates can tie at the same distance; pick the one with
    # lowest geometric distance (deterministic tiebreaker via index).
    best_idx = int(idxs[0])
    best_dist = float(edge_geoms[best_idx].distance(point))
    for idx in idxs[1:]:
        d = float(edge_geoms[int(idx)].distance(point))
        if d < best_dist:
            best_dist = d
            best_idx = int(idx)
    return edge_keys[best_idx], edge_geoms[best_idx], best_dist


def _find_reverse_edge_key(graph: nx.Graph, u, v):
    """For an OSMnx-style MultiDiGraph, find a `k_reverse` such that
    `(v, u, k_reverse)` exists. Returns the smallest such key (typically 0),
    or None if no reverse edge exists. Simple/undirected graphs always return
    None — they don't have a distinct reverse direction.

    **Self-loops** (`u == v`) return None: a loop edge IS its own reverse
    in the adjacency-dict sense, so `split_two_way_edge_at_point` would
    try to split the same edge twice and corrupt the graph. Snap falls
    back to single-direction split for self-loops.
    """
    if not (graph.is_multigraph() and graph.is_directed()):
        return None
    if u == v:
        return None
    if not graph.has_edge(v, u):
        return None
    keys = list(graph.succ[v][u].keys())
    if not keys:
        return None
    return min(keys)


def insert_projected_nodes(
    points: gpd.GeoDataFrame,
    graph: nx.Graph,
    *,
    max_distance: float | None,
    node_spacing: float = 50.0,
    eligible_node_ids: set | list | pd.Index | None = None,
    eligible_node_flag: str | None = None,
    cost_excluded_flag: str | None = None,
    edge_filter: Iterable[str] | Callable[[dict], bool] | None = None,
    verbose: bool = False,
) -> None:
    """Enrich `graph` with virtual nodes so every point in `points` has a
    graph node within snap distance, for use by a subsequent
    `snap_to_network_nodes` call.

    For each point in `points`, finds the nearest filter-eligible edge
    within `max_distance`. Project the point onto that edge:

    - If the projection lands within `node_spacing` of either endpoint
      of the edge, do NOTHING — the endpoint will serve as the snap
      target when `snap_to_network_nodes` runs on the enriched graph.
    - Otherwise, insert a virtual node at the projection and split the
      edge. The new node carries `is_virtual=1`. Child edges inherit
      the parent's attributes (including `highway`, so a virtual on a
      `primary` road shows up as priority for downstream tools that
      derive priority-node sets from edge tags).

    Points beyond `max_distance` from any filter-eligible edge contribute
    no insertion.

    This function MUTATES `graph` in place and returns nothing. The
    (point → node) mapping comes from the subsequent
    `snap_to_network_nodes` call; persisting the snap result is the
    caller's responsibility.

    **Composability.** The function can be called repeatedly on the same
    graph with different `edge_filter` values, accumulating virtuals
    suited to different needs:

        # Cells: densify any non-highway edge near each cell.
        insert_projected_nodes(
            cells_gdf, graph,
            max_distance=200, node_spacing=75,
        )
        # Zones on main roads only.
        insert_projected_nodes(
            zones_gdf, graph,
            max_distance=1000, node_spacing=250,
            edge_filter={'primary', 'secondary', 'tertiary'},
        )

    Existing virtual nodes from earlier calls are respected: the
    per-edge endpoint check on `node_spacing` means a new projection
    that falls near an already-inserted virtual yields no further
    insertion (the existing virtual becomes the snap target).

    **Single-pass via in-memory lineage walk.** Processing is a single
    straight pass through `points`. The STRtree is built once at the start
    and never rebuilt. When a point's nearest-edge query returns a parent
    that's already been split earlier in this call, an in-memory
    child-lineage walk re-projects onto the appropriate descendant child —
    no STRtree rebuild required. The walk asserts an invariant (every
    freshly-split parent has a paired child_lineage entry from the same
    insertion block) and raises if violated.

    Edge eligibility: an edge is a candidate for insertion if both
    endpoints are in the eligible node set, it isn't cost-excluded
    (per `cost_excluded_flag`, if supplied), and it passes `edge_filter`.

    Args:
        points: GeoDataFrame of points whose projections may trigger
            insertions.
        graph: NetworkX graph with `x`/`y` node attributes; edges must
            carry `geometry` LineStrings (consolidated OSMnx graphs
            satisfy this).
        max_distance: max distance from a point to its nearest candidate
            edge. Points farther than this contribute no insertion.
            `None` = unbounded.
        node_spacing: minimum spacing between an insertion and an
            existing endpoint of the chosen edge. If the projection
            lands within this radius of either endpoint, no insertion
            happens (the endpoint will serve). Default 50.0.
        eligible_node_ids: restrict candidate edges to those with both
            endpoints in this set. `None` (default) allows all nodes.
            Mutually exclusive with `eligible_node_flag`.
        eligible_node_flag: alternative to `eligible_node_ids` — name
            of a per-node bool attribute marking eligible endpoints
            (e.g., `snap_eligible_<mode>` from `prepare_network`).
        cost_excluded_flag: name of a per-edge bool attribute marking
            cost-excluded edges; if set, those edges are skipped.
        edge_filter: which edges may receive a virtual node.
            - `None` (default): exclude motorway / trunk classes (see
              `DEFAULT_EXCLUDED_HIGHWAY_TAGS`).
            - `Iterable[str]`: allowlist of OSM `highway` tag strings
              (e.g., `{'primary', 'secondary', 'tertiary'}` to only
              densify main roads).
            - `Callable[[dict], bool]`: arbitrary predicate on edge data.
        verbose: if True, print one summary line with insertion / skip /
            no-match counts. Useful on large networks for sanity-checking
            that the filter and radius parameters are tuned reasonably.

    Returns:
        None. Mutates `graph` in place.
    """
    # ---- Resolve eligible node set ------------------------------------------
    if eligible_node_ids is not None and eligible_node_flag is not None:
        raise ValueError("Pass either `eligible_node_ids` or `eligible_node_flag`, not both.")
    if eligible_node_ids is None and eligible_node_flag is not None:
        eligible_node_ids = [
            n for n, data in graph.nodes(data=True) if data.get(eligible_node_flag)
        ]
    if eligible_node_ids is None:
        eligible_node_set: set = set(graph.nodes)
    else:
        eligible_node_set = set(eligible_node_ids)
        if not eligible_node_set:
            raise ValueError("Eligibility filter excluded every node in the graph. Cannot insert.")

    # ---- Resolve edge filter -------------------------------------------------
    edge_predicate = _resolve_edge_filter(edge_filter)

    # ---- Build STRtree of filter-eligible edges -----------------------------
    elig_strtree, elig_keys, elig_geoms = _build_edge_strtree(
        graph, eligible_node_set, cost_excluded_flag, predicate=edge_predicate
    )

    def _edge_geom(edge_key):
        """Look up the current geometry of an edge by its key tuple."""
        if len(edge_key) == 3:
            return graph.edges[edge_key[0], edge_key[1], edge_key[2]]["geometry"]
        return graph.edges[edge_key[0], edge_key[1]]["geometry"]

    # Tracks parents split during THIS call and their immediate children.
    # The lineage walk uses child_lineage to descend through stale ancestors
    # without rebuilding the STRtree. INVARIANT: every key added to
    # freshly_split is added to child_lineage in the same insertion block.
    freshly_split: set = set()
    child_lineage: dict = {}

    def _walk_lineage(edge_key, edge_geom, point):
        """Walk down child_lineage from edge_key (which IS in freshly_split),
        picking the closest child at each level, until landing on a
        non-stale descendant. Returns (edge_key, edge_geom, edge_dist).

        Raises RuntimeError on invariant violations: a stale key with no
        recorded children, or > 50 hops (geometric subdivision should bottom
        out in O(log(L / node_spacing)) hops well before that).
        """
        cur_key, cur_geom = edge_key, edge_geom
        for _ in range(50):
            if cur_key not in freshly_split:
                return cur_key, cur_geom, float(cur_geom.distance(point))
            children = child_lineage.get(cur_key)
            if not children:
                raise RuntimeError(
                    f"insert_projected_nodes lineage walk hit a stale parent "
                    f"{cur_key!r} with no recorded children — every "
                    f"freshly_split entry should be paired with a child_lineage "
                    f"entry at insertion time."
                )
            best_key, best_geom, best_d = None, None, float("inf")
            for ck, cg in children:
                d = float(cg.distance(point))
                if d < best_d:
                    best_d = d
                    best_key, best_geom = ck, cg
            cur_key, cur_geom = best_key, best_geom
        raise RuntimeError(
            f"insert_projected_nodes lineage walk exceeded 50 hops descending "
            f"from {edge_key!r} — geometric subdivision should bottom out well "
            f"before this. Probable infinite loop in child_lineage."
        )

    # Pre-compute next node id once. The split_* primitives would otherwise
    # re-scan `graph.nodes` per call to derive `max(int_ids) + 1`, making the
    # full loop O(M × N) — hours on country-scale graphs. We hoist the scan,
    # then pass `new_node_id` explicitly each iteration and increment locally.
    int_node_ids = [n for n in graph.nodes if isinstance(n, (int, np.integer))]
    if not int_node_ids:
        raise ValueError("Cannot auto-derive new node ids: graph has no int-typed node ids.")
    next_node_id = int(max(int_node_ids)) + 1

    # Single-pass insertion. By construction (every freshly_split entry has a
    # paired child_lineage entry from the same insertion block), the lineage
    # walk always lands on a non-stale descendant; no point gets deferred.
    # `_walk_lineage` raises if the invariant is ever violated.
    n_inserted = 0
    n_endpoint_skip = 0
    n_no_match = 0

    for pt_id in points.index:
        pt_geom = points.loc[pt_id].geometry
        point = Point(pt_geom.x, pt_geom.y)

        # Find nearest filter-eligible edge.
        hit = _find_nearest_edge(elig_strtree, elig_keys, elig_geoms, point, max_distance)
        if hit is None:
            n_no_match += 1
            continue

        edge_key, edge_geom, _ = hit

        # Stale-edge lineage walk: re-project onto descendant child.
        if edge_key in freshly_split:
            edge_key, edge_geom, _ = _walk_lineage(edge_key, edge_geom, point)

        # Pre-compute reverse edge for two-way handling.
        k_rev = None
        rev_geom: LineString | None = None
        if len(edge_key) == 3:
            u_pre, v_pre, _ = edge_key
            k_rev_candidate = _find_reverse_edge_key(graph, u_pre, v_pre)
            if k_rev_candidate is not None:
                rev_geom_candidate = graph.edges[v_pre, u_pre, k_rev_candidate].get("geometry")
                if isinstance(rev_geom_candidate, LineString):
                    k_rev = k_rev_candidate
                    rev_geom = rev_geom_candidate

        # Per-edge endpoint check: if the projection is within node_spacing
        # of either endpoint, do nothing — the existing endpoint will serve
        # as the snap target downstream. This is what enforces "one node
        # per ~node_spacing along each edge".
        proj_fwd = edge_geom.project(point)
        edge_len = edge_geom.length
        near_endpoint = proj_fwd <= node_spacing or proj_fwd >= edge_len - node_spacing
        if not near_endpoint and rev_geom is not None:
            proj_rev = rev_geom.project(point)
            if proj_rev <= node_spacing or proj_rev >= rev_geom.length - node_spacing:
                near_endpoint = True
        if near_endpoint:
            n_endpoint_skip += 1
            continue

        # Insert a virtual node — two-way if applicable, else single-direction.
        # `new_node_id=next_node_id` skips the split primitive's internal
        # max-int scan; we bump our local counter after each successful insert.
        if len(edge_key) == 3:
            u, v, k_fwd = edge_key
            if k_rev is not None and rev_geom is not None:
                result = split_two_way_edge_at_point(
                    graph,
                    u,
                    v,
                    point,
                    k_forward=k_fwd,
                    k_reverse=k_rev,
                    new_node_id=next_node_id,
                    extra_node_attrs={"is_virtual": 1},
                )
                new_node = result.new_node_id
                rev_parent_key = (v, u, k_rev)
                freshly_split.add(rev_parent_key)
                fwd_c1, fwd_c2 = result.forward.child1, result.forward.child2
                rev_c1, rev_c2 = result.reverse.child1, result.reverse.child2
                child_lineage[edge_key] = [
                    (fwd_c1, _edge_geom(fwd_c1)),
                    (fwd_c2, _edge_geom(fwd_c2)),
                ]
                child_lineage[rev_parent_key] = [
                    (rev_c1, _edge_geom(rev_c1)),
                    (rev_c2, _edge_geom(rev_c2)),
                ]
            else:
                result_single = split_edge_at_point(
                    graph,
                    u,
                    v,
                    point,
                    k=k_fwd,
                    new_node_id=next_node_id,
                    extra_node_attrs={"is_virtual": 1},
                )
                new_node = result_single.new_node_id
                c1, c2 = result_single.child1, result_single.child2
                child_lineage[edge_key] = [
                    (c1, _edge_geom(c1)),
                    (c2, _edge_geom(c2)),
                ]
        else:
            u, v = edge_key
            result_single = split_edge_at_point(
                graph,
                u,
                v,
                point,
                new_node_id=next_node_id,
                extra_node_attrs={"is_virtual": 1},
            )
            new_node = result_single.new_node_id
            c1, c2 = result_single.child1, result_single.child2
            child_lineage[edge_key] = [
                (c1, _edge_geom(c1)),
                (c2, _edge_geom(c2)),
            ]

        next_node_id += 1
        freshly_split.add(edge_key)
        # Add the synthetic node to the eligible set so subsequent within-
        # call queries treat it as a valid endpoint (matters if a future
        # invocation rebuilds the STRtree; currently this is single-pass
        # so the set isn't re-read, but kept for invariant clarity).
        eligible_node_set.add(new_node)
        n_inserted += 1

    if verbose:
        print(
            f"  insert_projected_nodes: inserted={n_inserted:,}  "
            f"endpoint_skip={n_endpoint_skip:,}  "
            f"no_match={n_no_match:,}  "
            f"(of {len(points):,} points)"
        )


def transport_centroid(
    polygons: gpd.GeoDataFrame,
    graph: nx.Graph,
    eligible_node_ids: set | list | pd.Index,
    *,
    centroid_method: Literal["median", "mean"] = "median",
    fallback_to_geometric_centroid: bool = True,
) -> gpd.GeoDataFrame:
    """Per-polygon centroid of the eligible network nodes that fall inside it.

    For each polygon, find the eligible nodes whose location is within the
    polygon and return the median (or mean) of their `(x, y)` coordinates as
    a single point. Output is a `GeoDataFrame` of points indexed by polygon
    ID — ready to feed into `insert_projected_nodes` (to enrich the network
    if needed) followed by `snap_to_network_nodes` (to get the polygon-to-
    node mapping).

    Why centroid-then-snap instead of centroid-only or sjoin-only: zones
    (especially uniformly-tiled units like H3 hexes) often have geometric
    centroids that land on an arbitrary minor node — a service road, a
    dead-end, or worse. The median / mean of the eligible nodes within the
    polygon produces a "gravitational centre" of the polygon's
    transportation grid, biased toward where the routable network actually
    is. Snapping from there (separately, via `snap_to_network_nodes`)
    reliably lands on a more representative node.

    Polygons with no eligible node inside fall back to their geometric
    centroid if `fallback_to_geometric_centroid=True` (default). Otherwise
    they're dropped from the output.

    Args:
        polygons: GeoDataFrame of polygons. Output preserves
            `polygons.index` and `polygons.crs`.
        graph: NetworkX graph with `x` / `y` node attributes.
        eligible_node_ids: set of nodes considered when computing the
            centroid. Typically the snap-eligible set from
            `prepare_network` (`prepared.snap_eligible_nodes`).
        centroid_method: `'median'` (default) or `'mean'`. Median is more
            robust against outlier nodes (e.g., a single highway-on-ramp
            node included by accident).
        fallback_to_geometric_centroid: when True, polygons with no
            eligible node inside fall back to their geometric centroid.
            When False, such polygons are dropped from the output (the
            returned GeoDataFrame's index is a subset of
            `polygons.index`).

    Returns:
        `gpd.GeoDataFrame` of `Point` geometries, indexed by
        `polygons.index` (or a subset when `fallback_to_geometric_centroid=False`).
    """
    eligible_set = set(eligible_node_ids)
    if not eligible_set:
        raise ValueError("`eligible_node_ids` must be non-empty.")

    elig_ids = [n for n in graph.nodes if n in eligible_set]
    if not elig_ids:
        raise ValueError(
            "No eligible nodes present in the graph "
            "(every id in `eligible_node_ids` is missing from `graph.nodes`)."
        )
    elig_x = [graph.nodes[n]["x"] for n in elig_ids]
    elig_y = [graph.nodes[n]["y"] for n in elig_ids]
    eligible_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(elig_x, elig_y),
        index=pd.Index(elig_ids, name="node_id"),
        crs=polygons.crs,
    )

    # Spatial join: which eligible nodes fall in which polygon.
    joined = gpd.sjoin(
        eligible_gdf[["geometry"]],
        polygons[["geometry"]],
        how="inner",
        predicate="within",
    )
    # Group by the polygon-side index column (named after polygons.index.name,
    # or 'index_right' if anonymous).
    poly_id_col = polygons.index.name if polygons.index.name is not None else "index_right"

    # Per-polygon transport centroid: median or mean of constituent node coords.
    transport_xy: dict = {}
    if poly_id_col in joined.columns:
        for poly_id, sub in joined.groupby(poly_id_col):
            x = sub.geometry.x.to_numpy()
            y = sub.geometry.y.to_numpy()
            if centroid_method == "median":
                transport_xy[poly_id] = (float(np.median(x)), float(np.median(y)))
            elif centroid_method == "mean":
                transport_xy[poly_id] = (float(np.mean(x)), float(np.mean(y)))
            else:
                raise ValueError(
                    f"`centroid_method` must be 'median' or 'mean', got {centroid_method!r}."
                )

    # Fallback for polygons with no eligible node inside.
    missing_ids = polygons.index.difference(pd.Index(list(transport_xy.keys())))
    if len(missing_ids) > 0 and fallback_to_geometric_centroid:
        for poly_id in missing_ids:
            centroid = polygons.loc[poly_id, "geometry"].centroid
            transport_xy[poly_id] = (float(centroid.x), float(centroid.y))

    # Build the output GeoDataFrame preserving polygons.index order (for the
    # subset of polygons that ended up with a centroid).
    ordered_ids = [p for p in polygons.index if p in transport_xy]
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [transport_xy[p][0] for p in ordered_ids],
            [transport_xy[p][1] for p in ordered_ids],
        ),
        index=pd.Index(ordered_ids, name=polygons.index.name),
        crs=polygons.crs,
    )
