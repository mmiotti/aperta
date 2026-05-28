"""Graph-construction and -manipulation helpers for transport networks.

Aperta operates on `networkx.Graph` (and its multi/directed variants) as its
canonical graph type. This module supplies the operations on those graphs
that don't fit under `routing` (shortest-path queries) or `osm_helpers`
(OSM-specific download / parsing):

- **Intersection consolidation**: `consolidate_intersections` wraps
  `osmnx.consolidate_intersections` but preserves intersection-attribute
  nodes (traffic signals, stop signs, roundabouts) that the OSMnx default
  drops, which matters for any route-level analysis that counts those
  features (Section §3.3 of the toolkit paper).
- **Node snapping**: `snap_to_network_nodes` and `assign_to_eligible_centroid`
  map non-graph points (cell centroids, addresses) onto the nearest graph
  node, with optional filtering to a subset of eligible nodes.
- **Edge / node attribute helpers**: aggregate node attributes onto edges
  (`aggregate_nodes_to_edges`), aggregate edge attributes onto nodes
  (`aggregate_edges_to_nodes`), and write attribute values through to a
  graph in a tolerant way (`set_nx_edge_attributes_filled`).
- **Edge betweenness sampling**: `get_nested_edge_betweenness` runs the
  per-origin Dijkstra + path-walking accumulator used by the traffic-flow
  estimation pipeline in `traffic_flows.py`.
"""

from collections import defaultdict
from typing import Callable, Literal, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from aperta.errors import DataError

# OSM highway-type ranking used by `collapse_highway_lists_by_rank` and
# `flag_node_intersections` (for max/min per-node highway rank). Higher
# value = more major road. Anything not listed (or `None`) is treated
# as rank -1 ("not a real motor-vehicle road").
HIGHWAY_RANKS: dict[str, int] = {
    "motorway": 7,
    "motorway_link": 7,
    "trunk": 6,
    "trunk_link": 6,
    "primary": 5,
    "primary_link": 5,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 3,
    "tertiary_link": 3,
    "residential": 2,
    "road": 2,
    "living_street": 1,
    "pedestrian": 1,
    "unclassified": -1,
    "service": -1,
    "busway": -1,
    "cycleway": -1,
    "footway": -1,
    "path": -1,
    "track": -1,
    "steps": -1,
    "crossing": -1,
    "disused": -1,
}


def _highway_rank(value) -> int:
    """Rank lookup tolerant of strings, lists (OSMnx-merged), and None."""
    if value is None:
        return -1
    if isinstance(value, list):
        return max((HIGHWAY_RANKS.get(v, -1) for v in value), default=-1)
    return HIGHWAY_RANKS.get(value, -1)


def collapse_highway_lists_by_rank(graph: nx.Graph) -> None:
    """Mutate `graph` in place: collapse list-valued edge `highway` to a single
    string (the highest-rank value via `HIGHWAY_RANKS`).

    After `osmnx.consolidate_intersections`, edges built from multiple source
    edges have `highway` as a *list* of strings. Most downstream code expects
    a single string and silently picks the first element (e.g.
    `osmnx.add_edge_speeds` does this internally), which is not principled
    when the merged edges differ in road class. This helper picks the most
    *major* value instead (motorway > trunk > primary > … > unclassified).

    Unknown highway names map to rank `-1`; when a list contains only unknowns
    the resulting collapsed value is the unknown with the highest dict-order.
    Edges without a `highway` attribute are left alone.

    Auto-called from inside `consolidate_intersections`; callable standalone
    for graphs consolidated by external tooling.
    """
    if graph.is_multigraph():
        edges_data = (d for _, _, _, d in graph.edges(keys=True, data=True))
    else:
        edges_data = (d for _, _, d in graph.edges(data=True))
    for d in edges_data:
        hw = d.get("highway")
        if not isinstance(hw, list):
            continue
        ranks = [HIGHWAY_RANKS.get(v_, -1) for v_ in hw]
        d["highway"] = hw[ranks.index(max(ranks))]


def set_nx_edge_attributes_filled(
    g: nx.MultiGraph, attr: dict | pd.Series, attr_name: str, fill_value=0, strict: bool = False
):
    """Set per-edge attribute `attr_name` on `g`, filling missing edges with `fill_value`.

    `nx.set_edge_attributes` silently leaves edges absent from the input mapping
    without the attribute, which is a footgun for downstream code that expects
    the attribute to be present on every edge. This wrapper writes `fill_value`
    instead.

    Args:
        g: a MultiGraph (uses `(u, v, k)` edge keys).
        attr: edge → value mapping, keyed by `(u, v, k)` tuples.
        attr_name: edge attribute name to write.
        fill_value: value to assign to edges missing from `attr`. Default 0.
        strict: if True, raise `DataError` when `attr` is missing any of the
            graph's edges. Default False (silently fill).

    Returns:
        `g`, mutated in place.
    """
    if strict:
        _idx = pd.Series(index=list(g.edges(keys=True)))
        n = len(_idx.index.difference(pd.Series(attr).index))
        if n > 0:
            raise DataError("Incomplete data: {n:,} edges are missing in `attr'.")
    _data = {k: attr.get(k, fill_value) for k in g.edges(keys=True)}
    nx.set_edge_attributes(g, _data, attr_name)
    return g


def get_nested_edge_betweenness(
    g: nx.Graph,
    nested_node_sample: dict,
    weights: str | None = None,
    *,
    cutoff: float | None = None,
) -> pd.Series:
    """Edge usage counts from a nested (origin → sampled-destinations) sample.

    For each origin in `nested_node_sample`, runs a single-source Dijkstra
    on `g` (via `scipy.sparse.csgraph.dijkstra` with `return_predecessors`),
    walks the predecessor chain from each sampled destination back to the
    origin, and adds 1 to every edge on the path. The result is the
    weighted sum over all sampled OD pairs — a "traffic-stress"-style edge
    usage count, not classical Brandes' betweenness.

    Repeated destinations in the per-origin sample naturally count multiple
    times (each occurrence adds 1 to its path's edges), so weight comes
    from the upstream sampling step's destination distribution.

    Args:
        g: networkx graph (any variant). MultiGraph parallel edges with the
            same `(u, v)` collapse to the min-`weight` edge for routing,
            and the chosen key is the one credited in the output.
        nested_node_sample: `{origin_node -> array_of_dest_nodes}`, typically
            from `traffic_flows.nested_node_sample`. Origins are unique;
            duplicate destinations within an origin's array are fine.
        weights: edge attribute name to use as the per-edge cost (e.g.
            `'duration_s'`). Required — there's no "all edges weight 1"
            default since traffic-flow sampling always needs real costs.
        cutoff: optional network-distance cutoff in weight units. Passed to
            `csg.dijkstra(limit=cutoff)` to truncate each per-origin search
            once destinations beyond the cutoff are unreachable anyway. Set
            this to the upstream sampling radius (typically `r_zones` from
            `od_pairs.get_pairs`) — destinations sampled within that radius
            are guaranteed reachable within `cutoff`, and the truncation
            gives a large speed-up on country-scale graphs. Default `None`
            = no cutoff.

    Returns:
        `pd.Series` indexed by edge ID — `(u, v)` for plain graphs, `(u, v, k)`
        for multigraphs — with the accumulated edge usage count.
    """
    # Local import to keep scipy.sparse out of the module load path.
    import scipy.sparse.csgraph as csg

    from aperta.routing import _graph_to_csr

    if weights is None:
        raise ValueError("`weights` is required: traffic-flow sampling needs a real edge cost.")
    is_multi = g.is_multigraph()
    csr, nx_to_seq, seq_to_nx, parallel_keys = _graph_to_csr(g, weights, return_parallel_keys=True)
    limit = cutoff if cutoff is not None else np.inf

    out: dict = defaultdict(float)
    for orig_nx, dest_nodes in nested_node_sample.items():
        if orig_nx not in nx_to_seq:
            continue
        orig_seq = nx_to_seq[orig_nx]
        _, pred = csg.dijkstra(csr, indices=[orig_seq], limit=limit, return_predecessors=True)
        pred_row = pred[0]
        for dest_nx in dest_nodes:
            v_seq = nx_to_seq.get(dest_nx)
            if v_seq is None or v_seq == orig_seq:
                continue
            # Walk predecessors back to the origin; accumulate 1 per edge.
            while v_seq != orig_seq:
                u_seq = pred_row[v_seq]
                if u_seq < 0:
                    break  # unreachable / beyond cutoff
                edge_key: tuple
                if is_multi:
                    k = parallel_keys.get((int(u_seq), int(v_seq)))
                    edge_key = (seq_to_nx[int(u_seq)], seq_to_nx[int(v_seq)], k)
                else:
                    edge_key = (seq_to_nx[int(u_seq)], seq_to_nx[int(v_seq)])
                out[edge_key] += 1
                v_seq = u_seq
    return pd.Series(out)


def _add_to_edge_info(node_row, collected_edge_information, cols, node_edge_relations):
    """Fan a node's feature values out onto each edge it touches."""
    if isinstance(node_edge_relations, str):
        edge_ids = node_row[node_edge_relations].split(",")
    elif isinstance(node_edge_relations, nx.Graph):
        edge_ids = list(node_edge_relations.edges(node_row.name, keys=True))
    else:
        raise TypeError("node_edge_relations must be a str or nx.Graph.")
    for edge_id in edge_ids:
        if edge_id not in collected_edge_information:
            collected_edge_information[edge_id] = {col: [] for col in cols}
        for col in cols:
            collected_edge_information[edge_id][col].append(node_row[col])
    return collected_edge_information


def add_node_features_to_edges(
    df_nodes: pd.DataFrame, cols: list[str], node_edge_relations: str | nx.Graph, agg_func: str
) -> pd.DataFrame:
    """Aggregate node-level features onto the edges they touch (sum or mean).

    Args:
        df_nodes: list of nodes, supplied as a DataFrame.
        cols: list of columns in df_nodes to be mapped to edges.
        node_edge_relations: if str, must list the edges belonging to each node in column
            'node_edge_relations' in df_nodes, separated by a comma (,). Otherwise, supply an
            nx.Graph where the ID of each node corresponds to the index in df_nodes.
        agg_func: how to aggregate values from different nodes onto a single edge. 'sum' or 'mean',
            or 'median'.
    """

    collected_edge_information: dict = {}
    df_nodes.apply(
        lambda row: _add_to_edge_info(row, collected_edge_information, cols, node_edge_relations),
        axis=1,
    )
    for k, d in collected_edge_information.items():
        for col, values in d.items():
            if agg_func == "sum":
                collected_edge_information[k][col] = sum(values)
            elif agg_func == "mean":
                collected_edge_information[k][col] = float(np.average(values))
            elif agg_func == "median":
                collected_edge_information[k][col] = float(np.median(values))
            else:
                raise NotImplementedError(f"agg_func `{agg_func}` is not implemented.")
    return pd.DataFrame.from_dict(collected_edge_information, orient="index")


def _mean_numeric(values: list):
    """Mean over values coercible to float; first value as fallback if none coerce.

    OSM `lanes` / `maxspeed` come through as strings (sometimes numeric like
    `'50'`, sometimes with units / labels like `'50 mph'` or `'RU:urban'`).
    Coercible values are averaged; non-coercible are skipped. If nothing
    parses, returns the first raw value (preserves a sensible default rather
    than producing `NaN`).
    """
    nums: list[float] = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if nums:
        return sum(nums) / len(nums)
    return values[0] if values else None


# Default edge-attribute aggregators applied to LIST-VALUED edge attrs
# post-consolidation. `lanes` and `maxspeed` get numeric-mean so merged
# edges expose single values; non-merged edges keep whatever the source
# had (typically a single string from OSM).
#
# `length` deliberately not here: OSMnx 2.x sums it across merged edges,
# but the merged edge has a single geometry whose actual length is
# *smaller* than that sum (parallel paths collapse to one). We recompute
# `length` from `geometry.length` post-consolidation in metric units.
DEFAULT_EDGE_ATTR_AGGS = {
    "lanes": _mean_numeric,
    "maxspeed": _mean_numeric,
}


def _parse_lanes(raw) -> float | None:
    """OSM `lanes` is messy — string, list, missing. Returns float or None."""
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def lanes_per_direction(edge_data: dict) -> float:
    """Per-direction lane count for a directed edge.

    OSM's `lanes` tag is the **total** lane count across both directions on
    two-way roads, and OSMnx inherits the same value on both directional
    edges. Any per-direction quantity (directional AADT, per-lane capacity)
    is therefore off by ~2× on two-way segments without correction — and
    biased *unequally* between mostly-one-way road classes (motorways) and
    mostly-two-way ones (primary / secondary), which a single coefficient
    can't absorb.

    Rules:
      - `oneway=True`: all lanes are in this direction → return lanes.
      - `lanes` missing: OSM implicit default (1 per direction) → return 1.
      - `lanes ≤ 1`: can't split a single lane → return 1.
      - otherwise: return lanes / 2.

    Pure function over `edge_data` — caller decides whether to write the
    result back as an edge attribute. `consolidate_intersections` calls
    this for every consolidated edge and stores the result as
    `lanes_per_direction`.
    """
    lanes = _parse_lanes(edge_data.get("lanes"))
    oneway = bool(edge_data.get("oneway", False))
    if lanes is None:
        return 1.0
    if oneway or lanes <= 1:
        return max(1.0, lanes)
    return lanes / 2.0


# Edge attributes dropped post-consolidation by `consolidate_intersections`
# (callers can override). `name` is the main offender: it lists across
# merged edges, costs disk space in `.graphml`, and isn't used anywhere
# in aperta.
DEFAULT_DROP_EDGE_ATTRS = ["name"]


def _int_via_float(value) -> int:
    """`int(float(v))` — tolerates both `'0'`/`'1'` and `'0.0'`/`'1.0'` strings.

    Plain `int()` raises on float-formatted strings (`int('0.0')` → ValueError).
    Used as the cast for graphml-loaded `is_*` flags so older saves (where
    these were written as floats) and newer saves (ints) both round-trip
    cleanly.
    """
    return int(float(value))


# Per-node attribute dtypes that `consolidate_intersections` writes as
# ints. OSMnx's own `default_node_dtypes` only knows about its built-in
# attrs (elevation, x, y, osmid, street_count, lat, lon), so without this
# constant our custom `is_*` and `*_highway_rank` flags round-trip as
# strings — and `int('0.0')` would raise downstream. The values are
# integer indicators (0 / 1 for the `is_*` flags, -1…7 for highway rank),
# so int round-trips cleanly (`'0'` / `'1'`) and `is_roundabout == 1`
# works as written. Pass to `ox.load_graphml` via the `node_dtypes`
# kwarg, or use `load_consolidated_graphml` below.
CONSOLIDATED_NODE_DTYPES: dict[str, Callable] = {
    "is_degree_3": _int_via_float,
    "is_degree_4": _int_via_float,
    "max_highway_rank": _int_via_float,
    "min_highway_rank": _int_via_float,
    # Per-obstacle `is_<name>` flags are dynamic; the default set used by
    # `consolidate_intersections` is included below. Callers that pass
    # custom `obstacle_node_tags` should extend this dict accordingly.
    "is_traffic_signal": _int_via_float,
    "is_stop": _int_via_float,
    "is_yield": _int_via_float,
    "is_roundabout": _int_via_float,
}

# Per-edge attribute dtypes that `consolidate_intersections` writes that
# aren't covered by OSMnx's `default_edge_dtypes`. Without this,
# `lanes_per_direction` round-trips as a string and arithmetic breaks
# downstream.
CONSOLIDATED_EDGE_DTYPES: dict[str, Callable] = {
    "lanes_per_direction": float,
}


def load_consolidated_graphml(
    filepath, *, node_dtypes: dict | None = None, edge_dtypes: dict | None = None, **kwargs
):
    """Load a graphml saved by `consolidate_intersections`, casting our
    custom `is_*` / `*_highway_rank` attrs back to float.

    Thin wrapper around `osmnx.load_graphml` that merges in
    `CONSOLIDATED_NODE_DTYPES`. OSMnx only auto-casts attrs in its
    `default_node_dtypes`; without this our custom per-node flags arrive
    as strings (`'0.0'` / `'1.0'`), which silently breaks arithmetic
    downstream.

    Args:
        filepath: path to a `.graphml` produced by `consolidate_intersections`.
        node_dtypes: optional override merged on top of
            `CONSOLIDATED_NODE_DTYPES` (caller's values win).
        edge_dtypes: optional override merged on top of
            `CONSOLIDATED_EDGE_DTYPES` (caller's values win). OSMnx's
            built-in defaults cover the OSM-native attrs; this adds the
            aperta-derived ones (currently `lanes_per_direction`).
        **kwargs: forwarded to `osmnx.load_graphml`.

    Returns:
        `nx.MultiDiGraph`.
    """
    import osmnx as ox

    merged_node = {**CONSOLIDATED_NODE_DTYPES, **(node_dtypes or {})}
    merged_edge = {**CONSOLIDATED_EDGE_DTYPES, **(edge_dtypes or {})}
    return ox.load_graphml(filepath, node_dtypes=merged_node, edge_dtypes=merged_edge, **kwargs)


def extract_obstacle_locations(
    graph: nx.Graph,
    *,
    obstacle_node_tags: dict[str, tuple[str, str]] | None = None,
    detect_roundabouts: bool = True,
) -> tuple[dict[str, list[tuple[float, float]]], list[tuple[float, float]]]:
    """Pull obstacle + roundabout `(x, y)` locations from a raw OSMnx graph.

    Companion to `consolidate_intersections`. Returns the two structures the
    consolidator needs (`obstacle_xy`, `roundabout_xy`) so callers can
    extract obstacles *once* from a canonical source (typically the raw car
    graph — the most signal-complete) and reuse for every network type's
    consolidation. This matters because OSMnx's per-network-type filters
    drop ways that signals sit on (e.g. trunk roads excluded from walk
    graphs), losing those signal nodes entirely from the walk graph's node
    set; passing the union of locations via `obstacle_locations=` /
    `roundabout_locations=` to `consolidate_intersections` reattaches them
    to whichever consolidated node is nearest in each network.

    Args:
        graph: raw OSMnx graph (any network_type).
        obstacle_node_tags: see `consolidate_intersections`.
        detect_roundabouts: if True, also extract midpoints of edges with
            `junction=roundabout`.

    Returns:
        `(obstacle_xy_per_type, roundabout_xy_list)`.
    """
    if obstacle_node_tags is None:
        obstacle_node_tags = {
            "traffic_signal": ("highway", "traffic_signals"),
            "stop": ("highway", "stop"),
            "yield": ("highway", "give_way"),
        }
    obstacle_xy: dict[str, list[tuple[float, float]]] = {name: [] for name in obstacle_node_tags}
    for _, ndata in graph.nodes(data=True):
        for obstacle_name, (key, value) in obstacle_node_tags.items():
            tag_value = ndata.get(key)
            if tag_value == value or (isinstance(tag_value, list) and value in tag_value):
                obstacle_xy[obstacle_name].append((ndata["x"], ndata["y"]))
    roundabout_xy: list[tuple[float, float]] = []
    if detect_roundabouts:
        for u, v, _, edata in graph.edges(keys=True, data=True):
            j = edata.get("junction")
            if j == "roundabout" or (isinstance(j, list) and "roundabout" in j):
                u_attr, v_attr = graph.nodes[u], graph.nodes[v]
                roundabout_xy.append(
                    ((u_attr["x"] + v_attr["x"]) / 2, (u_attr["y"] + v_attr["y"]) / 2)
                )
    return obstacle_xy, roundabout_xy


def consolidate_intersections(
    graph: nx.MultiDiGraph,
    tolerance: float,
    *,
    obstacle_buffer: float = 30.0,
    obstacle_node_tags: dict[str, tuple[str, str]] | None = None,
    obstacle_locations: dict[str, list[tuple[float, float]]] | None = None,
    detect_roundabouts: bool = True,
    roundabout_locations: list[tuple[float, float]] | None = None,
    node_attr_aggs: dict | None = None,
    edge_attr_aggs: dict | None = None,
    drop_edge_attrs: list[str] | None = None,
):
    """OSMnx intersection consolidation + obstacle-aware re-flagging.

    Wraps `osmnx.consolidate_intersections(rebuild_graph=True)` with the
    post-processing OSMnx alone misses: traffic-signal / stop / give-way
    nodes typically sit a few metres off the geometric intersection
    centre, so OSMnx's `tolerance`-based merge can throw those nodes away
    rather than carrying the `highway=traffic_signals` tag onto the
    surviving consolidated node. The result is a consolidated graph in
    which most intersections are not flagged as signalised even when
    they actually are — a distortion for any edge-weight model that
    penalises signals.

    This wrapper captures obstacle locations from the *original* graph
    before consolidation, then spatially re-attaches them to the nearest
    surviving consolidated node within `obstacle_buffer` metres. The
    same trick handles roundabouts, whose `junction=roundabout` tag
    lives on edges (not nodes) in OSM and is otherwise lost when the
    roundabout collapses to a single consolidated node.

    The returned graph has the per-node attributes set by
    `flag_node_intersections` (`is_degree_3`, `is_degree_4`,
    `max_highway_rank`, `min_highway_rank`), plus one `is_<name>` per
    requested obstacle type, plus `is_roundabout` if
    `detect_roundabouts=True`. Edge `highway` lists from the consolidation
    are collapsed to the highest-rank single string via
    `collapse_highway_lists_by_rank`. Each edge also gets
    `lanes_per_direction` (the OSM `lanes` tag corrected for two-way
    roads — see `lanes_per_direction()`). **Node IDs are new integer IDs**
    (per OSMnx behaviour) — caller must re-snap geo units to the
    consolidated graph.

    **Geometry guarantee**: every consolidated edge carries a `geometry`
    LineString (OSMnx attaches one during the rebuild). This isn't true
    of raw OSMnx graphs — `simplify=True` omits `geometry` from pure
    point-to-point edges (~10 % of edges typically), and downstream code
    that needs per-edge geometry (e.g. dual-graph construction, plotting
    with curvature) on a raw graph has to call
    `osmnx.graph_to_gdfs(..., fill_edge_geometry=True)` and copy
    `geometry` back. Consolidating first sidesteps that step.

    Args:
        graph: an OSMnx MultiDiGraph (projected; `tolerance` is in graph
            CRS units, usually metres). `osmnx` is required (optional
            extra `osm`).
        tolerance: nodes within this distance are merged. Typical urban
            values: 5–15 m; ~25 m for sparser networks.
        obstacle_buffer: max distance to which an obstacle from the
            original graph is re-attached to a consolidated node.
            Should be at least as large as `tolerance`; default 30 m
            comfortably covers signalised intersections.
        obstacle_node_tags: `{flag_name -> (osm_key, osm_value)}` — OSM
            node tags to extract as obstacles. Default:
            `{'traffic_signal': ('highway', 'traffic_signals')}`. Add
            `'stop': ('highway', 'stop')`, `'give_way': ('highway',
            'give_way')`, etc., as needed.
        obstacle_locations: pre-supplied `{flag_name -> [(x, y), ...]}` map.
            When given, the obstacle extraction from `obstacle_node_tags` is
            skipped — useful when obstacles come from a non-OSM source or
            were captured upstream.
        detect_roundabouts: if True (default), edges with
            `junction=roundabout` are detected before consolidation and
            their midpoints get re-attached as `is_roundabout`.
        roundabout_locations: pre-supplied list of roundabout midpoints
            `[(x, y), ...]`. When given, skips the edge-based roundabout
            detection from `detect_roundabouts`.
        node_attr_aggs: passed through to `ox.consolidate_intersections`.
            Any per-node attribute not listed here that varies across the
            nodes being merged will be carried through as a **list** of
            values.
        edge_attr_aggs: passed through to `ox.consolidate_intersections`
            to control how per-edge attributes are aggregated when parallel
            edges between the same `(u, v)` are collapsed.
        drop_edge_attrs: edge attributes to drop after consolidation. Use
            for attributes that osmnx's aggregation leaves in a confusing
            list-of-values form. Defaults to `DEFAULT_DROP_EDGE_ATTRS`.

    Returns:
        Consolidated `nx.MultiDiGraph` with new integer node IDs.
    """
    import osmnx as ox
    from scipy.spatial import KDTree

    # 1. Obstacle + roundabout locations.
    #    Pre-extracted `obstacle_locations` / `roundabout_locations` win —
    #    pass these from a canonical source (typically the raw car graph)
    #    so all network types share the same obstacle awareness
    #    (signals on trunk roads, for example, are dropped from walk
    #    graphs by OSMnx's network_type filter and would otherwise be
    #    absent from walk-graph consolidation entirely). Otherwise we
    #    extract from `graph` itself via `extract_obstacle_locations`.
    if obstacle_locations is None or (detect_roundabouts and roundabout_locations is None):
        auto_obstacle_xy, auto_roundabout_xy = extract_obstacle_locations(
            graph,
            obstacle_node_tags=obstacle_node_tags,
            detect_roundabouts=detect_roundabouts,
        )
        if obstacle_locations is None:
            obstacle_locations = auto_obstacle_xy
        if detect_roundabouts and roundabout_locations is None:
            roundabout_locations = auto_roundabout_xy

    # 2. Consolidate. OSMnx 2.x doesn't expose `edge_attr_aggs`, so the
    #    edge aggregation (numeric-mean for `lanes` / `maxspeed`, etc.)
    #    runs as a post-pass below on list-valued attrs only.
    # `rebuild_graph=True` guarantees the return is a MultiDiGraph (the
    # GeoSeries return is only when `rebuild_graph=False`), but OSMnx's
    # signature is a union — cast for the type checker.
    consolidated = cast(
        nx.MultiDiGraph,
        ox.consolidate_intersections(
            graph,
            tolerance=tolerance,
            rebuild_graph=True,
            reconnect_edges=True,
            node_attr_aggs=node_attr_aggs,
        ),
    )

    # 3. Post-consolidation edge cleanup:
    #    - drop unwanted attrs (saves disk space + avoids round-trip
    #      ambiguity for non-aggregated lists);
    #    - collapse list-valued attrs in `edge_attr_aggs` to single
    #      values;
    #    - recompute `length` from the edge geometry. OSMnx sums
    #      `length` across merged source edges, which inflates it for
    #      parallel-path merges — the merged edge's actual geometry is
    #      shorter than that sum. `geometry.length` gives metres in our
    #      metric-CRS graphs.
    drop_attrs = DEFAULT_DROP_EDGE_ATTRS if drop_edge_attrs is None else drop_edge_attrs
    eff_edge_aggs = DEFAULT_EDGE_ATTR_AGGS if edge_attr_aggs is None else edge_attr_aggs
    for _, _, _, d in consolidated.edges(keys=True, data=True):
        for attr in drop_attrs:
            d.pop(attr, None)
        for attr, aggregator in eff_edge_aggs.items():
            if isinstance(d.get(attr), list):
                d[attr] = aggregator(d[attr])
        geom = d.get("geometry")
        if geom is not None:
            d["length"] = float(geom.length)
        # Derived per-direction lane count — see `lanes_per_direction()`
        # for rationale. Runs after the lanes aggregator so list-valued
        # OSM tags are already collapsed.
        d["lanes_per_direction"] = lanes_per_direction(d)

    # 4. Collapse list-valued highway to a single string, then per-node
    #    intersection + highway-rank flags.
    collapse_highway_lists_by_rank(consolidated)
    flag_node_intersections(consolidated)

    # 5. Spatial re-attachment: nearest consolidated node within
    #    obstacle_buffer gets the obstacle / roundabout flag.
    node_ids = list(consolidated.nodes)
    if not node_ids:
        return consolidated
    node_xy = np.array([(consolidated.nodes[n]["x"], consolidated.nodes[n]["y"]) for n in node_ids])
    tree = KDTree(node_xy)

    def _allocate(locations: list[tuple[float, float]], flag_name: str) -> None:
        for nid in consolidated.nodes():
            consolidated.nodes[nid][f"is_{flag_name}"] = 0
        if not locations:
            return
        dists, idxs = tree.query(np.asarray(locations), distance_upper_bound=obstacle_buffer)
        # query returns idx == len(node_xy) for misses with distance_upper_bound.
        valid = (idxs < len(node_ids)) & np.isfinite(dists)
        for i in np.where(valid)[0]:
            consolidated.nodes[node_ids[int(idxs[i])]][f"is_{flag_name}"] = 1

    for name, locs in obstacle_locations.items():
        _allocate(locs, name)
    if detect_roundabouts and roundabout_locations is not None:
        _allocate(roundabout_locations, "roundabout")

    return consolidated


def flag_node_intersections(graph: nx.Graph) -> None:
    """Mutate `graph` in place to add per-node intersection + highway-rank flags.

    Four float attributes per node:

    - `is_degree_3`      — 1.0 if degree == 3 (T-intersection / fork), else 0.
    - `is_degree_4`      — 1.0 if degree >= 4 (cross / multi-way), else 0.
    - `max_highway_rank` — max `HIGHWAY_RANKS` value over edges incident to
      this node (`-1.0` for unknown / not-a-real-road, e.g. footways).
    - `min_highway_rank` — same with min.

    `is_degree_3` and `is_degree_4` are **mutually exclusive** — a degree-4
    node carries only the latter. (Degree 1 / 2 nodes — leaves and mid-edge
    nodes — get neither.) This is a deliberate change from a previous
    `has_intersection` (≥3) / `has_intersection_4` (≥4) cumulative encoding:
    edge-weight models that penalise 4-way more than 3-way should set the
    two coefficients independently rather than additively.

    Per-node obstacle flags (`is_traffic_signal`, `is_stop`, etc.) live in
    `consolidate_intersections`, which captures them from the original
    graph's OSM tags and re-attaches them spatially after consolidation.
    """
    is_directed = graph.is_directed()
    is_multi = graph.is_multigraph()

    # Per-node max / min highway rank from incident edges.
    node_max = {n: float("-inf") for n in graph.nodes}
    node_min = {n: float("inf") for n in graph.nodes}
    if is_multi:
        for u, v, _, d in graph.edges(keys=True, data=True):
            rank = _highway_rank(d.get("highway"))
            for endpoint in (u, v):
                if rank > node_max[endpoint]:
                    node_max[endpoint] = rank
                if rank < node_min[endpoint]:
                    node_min[endpoint] = rank
    else:
        for u, v, d in graph.edges(data=True):
            rank = _highway_rank(d.get("highway"))
            for endpoint in (u, v):
                if rank > node_max[endpoint]:
                    node_max[endpoint] = rank
                if rank < node_min[endpoint]:
                    node_min[endpoint] = rank

    for nid in graph.nodes():
        if is_directed:
            neighbours = set(graph.predecessors(nid)) | set(graph.successors(nid))
        else:
            neighbours = set(graph.neighbors(nid))
        degree = len(neighbours)
        graph.nodes[nid]["is_degree_3"] = int(degree == 3)
        graph.nodes[nid]["is_degree_4"] = int(degree >= 4)
        mx = node_max[nid]
        mn = node_min[nid]
        graph.nodes[nid]["max_highway_rank"] = int(mx) if mx != float("-inf") else -1
        graph.nodes[nid]["min_highway_rank"] = int(mn) if mn != float("inf") else -1


def snap_to_network_nodes(
    points: gpd.GeoDataFrame,
    graph: nx.Graph,
    *,
    max_distance: float | None = None,
    eligible_node_ids: set | list | pd.Index | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Snap each row in `points` to its nearest node in `graph`.

    For each point, finds the closest network node by Euclidean distance and
    returns both the node ID and the distance. The point CRS and the graph
    coordinates must already agree — this function does no reprojection.

    Network nodes must carry `x` and `y` attributes (aperta convention).
    Typical sources of such graphs are OSMnx (`ox.project_graph(...)` produces
    nodes with `x` / `y` in the target CRS) or aperta's own
    `network_processing` builders.

    Args:
        points: GeoDataFrame of points to snap. Output is indexed by
            `points.index`.
        graph: NetworkX (or compatible) graph with `x` / `y` node attributes.
        max_distance: optional cap. Points farther than this from every node
            return `NaN` for both ID and distance. `None` means no cap.
        eligible_node_ids: optional restriction — only nodes in this set are
            considered as snap targets. Use with `aggregate_edges_to_nodes`
            to filter out structurally undesirable snap targets (e.g.,
            motorway nodes, dead-end nodes, pedestrian-only paths for car
            analyses). `None` (default) considers all graph nodes.

    Returns:
        Tuple `(node_ids, distances)`:
            - `node_ids`: `pd.Series` of nearest-node IDs, indexed by `points.index`.
            - `distances`: `pd.Series` of distances (in CRS units), indexed by
              `points.index`.
    """
    from aperta import geo_mapping  # local import to avoid module-load cycle

    if eligible_node_ids is None:
        node_ids = list(graph.nodes)
    else:
        eligible_set = set(eligible_node_ids)
        node_ids = [n for n in graph.nodes if n in eligible_set]
        if not node_ids:
            raise ValueError(
                "`eligible_node_ids` filter excluded every node in the graph. "
                "Cannot snap to an empty set of targets."
            )
    node_x = [graph.nodes[n]["x"] for n in node_ids]
    node_y = [graph.nodes[n]["y"] for n in node_ids]
    nodes_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(node_x, node_y),
        index=pd.Index(node_ids, name="node_id"),
        crs=points.crs,
    )
    return geo_mapping.map_points_to_points(points, nodes_gdf, max_distance=max_distance)


def aggregate_edges_to_nodes(
    graph: nx.Graph,
    edge_attribute: str | Callable,
    *,
    aggregator: str | Callable = "max",
) -> pd.Series:
    """For each node in `graph`, aggregate `edge_attribute` across its connected edges.

    The inverse of `add_node_features_to_edges` (which propagates per-node
    features onto edges). Common use: classify each node by the highest-class
    road that touches it (`aggregator='max'`) — useful for filtering snap
    targets in `snap_to_network_nodes` (skip motorway-only nodes, etc.).

    For MultiGraphs / MultiDiGraphs, parallel edges between the same `(u, v)`
    each contribute their own value — for `'max'` this is harmless, for
    `'mean'` it slightly weights duplicated edges. For OSMnx graphs (where
    parallel edges typically carry identical attributes), this is fine.

    Args:
        graph: NetworkX graph.
        edge_attribute: name of an edge attribute (`str`) or a callable
            `(u, v, data) -> value`. Edges where the attribute is missing
            and the string form is used contribute `NaN`.
        aggregator: `'max'` (default), `'min'`, `'mean'`, `'sum'`, or a
            callable that takes a 1-D numpy array of per-edge values and
            returns a scalar. NaN handling is left to the aggregator
            (`'max'`/`'min'`/`'mean'`/`'sum'` use the nan-safe numpy
            variants and silently skip NaN edge values).

    Returns:
        `pd.Series` indexed by node ID with the per-node aggregated value.
        Isolated nodes (no edges) are absent from the result.
    """
    if isinstance(edge_attribute, str):
        attr_name = edge_attribute

        def _attr(u, v, data):
            return data.get(attr_name, np.nan)
    elif callable(edge_attribute):
        _attr = edge_attribute  # signature (u, v, data) -> value
    else:
        raise ValueError(
            f"`edge_attribute` must be a string or callable, got {type(edge_attribute).__name__}."
        )

    _agg: Callable
    if aggregator == "max":
        _agg = np.nanmax
    elif aggregator == "min":
        _agg = np.nanmin
    elif aggregator == "mean":
        _agg = np.nanmean
    elif aggregator == "sum":
        _agg = np.nansum
    elif callable(aggregator):
        _agg = aggregator
    else:
        raise ValueError(
            f"Unknown aggregator {aggregator!r}; expected "
            f"'max', 'min', 'mean', 'sum', or a callable."
        )

    per_node: defaultdict = defaultdict(list)
    is_multi = isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))
    if is_multi:
        for u, v, _k, data in graph.edges(keys=True, data=True):
            val = float(_attr(u, v, data))
            per_node[u].append(val)
            per_node[v].append(val)
    else:
        for u, v, data in graph.edges(data=True):
            val = float(_attr(u, v, data))
            per_node[u].append(val)
            per_node[v].append(val)

    # Aggregate with nan-safe semantics; suppress the "all-NaN slice"
    # warning since we return NaN in that case (and the user can filter).
    with np.errstate(all="ignore"):
        out = {}
        for n, vals in per_node.items():
            arr = np.asarray(vals, dtype=float)
            finite = (
                arr[np.isfinite(arr)]
                if _agg in (np.nanmax, np.nanmin, np.nanmean, np.nansum)
                else arr
            )
            if _agg in (np.nanmax, np.nanmin, np.nanmean) and finite.size == 0:
                out[n] = np.nan
            else:
                out[n] = float(_agg(arr))
    return pd.Series(out, name="node_value")


def assign_to_eligible_centroid(
    polygons: gpd.GeoDataFrame,
    graph: nx.Graph,
    eligible_node_ids: set | list | pd.Index,
    *,
    centroid_method: Literal["median", "mean"] = "median",
    fallback_to_geometric_centroid: bool = True,
    max_distance: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """For each polygon in `polygons`, assign it to a network node via a
    transport-weighted centroid built from the *eligible* network nodes
    inside the polygon.

    Designed for snapping zones (especially uniformly-tiled units
    like H3 hexes) whose geometric centroid often lands on an arbitrary
    minor node — a service road, a dead-end, or worse. Using the
    median / mean coordinates of the eligible nodes within the polygon
    produces a "gravitational centre" of the polygon's transportation
    grid; snapping to the nearest eligible node from that point reliably
    lands on a more representative node.

    Workflow per polygon:
        1. Find eligible nodes whose location falls within the polygon.
        2. Compute their median (or mean) (x, y) — the transport centroid.
        3. Snap that centroid to the nearest eligible node anywhere.

    Polygons with no eligible node inside fall back to their geometric
    centroid (snapped to the nearest eligible node anywhere) if
    `fallback_to_geometric_centroid=True`. Otherwise they get NaN.

    Args:
        polygons: GeoDataFrame of polygons to snap. Output is indexed by
            `polygons.index`.
        graph: NetworkX graph with `x` / `y` node attributes.
        eligible_node_ids: set of nodes that are valid snap targets.
            Typically built from `aggregate_edges_to_nodes` + a tier filter
            (e.g., `nodes where tier in {residential, tertiary, secondary}`).
        centroid_method: `'median'` (default) or `'mean'`. Median is more
            robust against outlier nodes (e.g., a single highway-on-ramp
            node included by accident).
        fallback_to_geometric_centroid: when True, polygons with no eligible
            node inside use their geometric centroid (then snapped to the
            nearest eligible node anywhere — could be outside the polygon).
            When False, such polygons get NaN ID + NaN distance.
        max_distance: optional cap on the final snap distance (CRS units).

    Returns:
        Tuple `(node_ids, distances)` indexed by `polygons.index`.
    """
    eligible_set = set(eligible_node_ids)
    if not eligible_set:
        raise ValueError("`eligible_node_ids` must be non-empty.")

    # Build a points GeoDataFrame of all eligible nodes (for sjoin + later snap).
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

    # Build a points GeoDataFrame of transport centroids (in polygon order).
    ordered_ids = [p for p in polygons.index if p in transport_xy]
    centroids_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [transport_xy[p][0] for p in ordered_ids],
            [transport_xy[p][1] for p in ordered_ids],
        ),
        index=pd.Index(ordered_ids, name=polygons.index.name),
        crs=polygons.crs,
    )

    # Snap each transport centroid to the nearest eligible node.
    snapped_ids, snapped_dists = snap_to_network_nodes(
        centroids_gdf,
        graph,
        max_distance=max_distance,
        eligible_node_ids=eligible_set,
    )

    # Reindex to the full polygons.index (NaN for any that fell through).
    return (snapped_ids.reindex(polygons.index), snapped_dists.reindex(polygons.index))
