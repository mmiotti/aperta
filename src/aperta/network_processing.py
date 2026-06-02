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
- **Mode-aware graph preparation**: `prepare_network` resolves per-mode
  defaults (directedness, network type, cost-excluded tags), applies
  `to_undirected()` where appropriate, and precomputes the largest
  connected (or strongly-connected) component as a snap-eligible node set.
  This avoids the trapped-node problem where consolidation isolates short
  one-way segments from which routing cannot escape.
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from aperta.errors import DataError

# OSM highway-type ranking used by `collapse_osm_highway_lists_by_rank` and
# `flag_node_osm_classification` (for max/min per-node highway rank). Higher
# value = more major road. Anything not listed (or `None`) is treated
# as rank -1 ("not a real motor-vehicle road").
OSM_HIGHWAY_RANKS: dict[str, int] = {
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


def _osm_highway_rank(value) -> int:
    """Rank lookup tolerant of strings, lists (OSMnx-merged), and None."""
    if value is None:
        return -1
    if isinstance(value, list):
        return max((OSM_HIGHWAY_RANKS.get(v, -1) for v in value), default=-1)
    return OSM_HIGHWAY_RANKS.get(value, -1)


def collapse_osm_highway_lists_by_rank(graph: nx.Graph) -> None:
    """Mutate `graph` in place: collapse list-valued edge `highway` to a single
    string (the highest-rank value via `OSM_HIGHWAY_RANKS`).

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
        ranks = [OSM_HIGHWAY_RANKS.get(v_, -1) for v_ in hw]
        d["highway"] = hw[ranks.index(max(ranks))]


def set_nx_edge_attributes_filled(
    graph: nx.MultiGraph, attr: dict | pd.Series, attr_name: str, fill_value=0, strict: bool = False
):
    """Set per-edge attribute `attr_name` on `graph`, filling missing edges with `fill_value`.

    `nx.set_edge_attributes` silently leaves edges absent from the input mapping
    without the attribute, which is a footgun for downstream code that expects
    the attribute to be present on every edge. This wrapper writes `fill_value`
    instead.

    Args:
        graph: a MultiGraph (uses `(u, v, k)` edge keys).
        attr: edge → value mapping, keyed by `(u, v, k)` tuples.
        attr_name: edge attribute name to write.
        fill_value: value to assign to edges missing from `attr`. Default 0.
        strict: if True, raise `DataError` when `attr` is missing any of the
            graph's edges. Default False (silently fill).

    Returns:
        `graph`, mutated in place.
    """
    if strict:
        _idx = pd.Series(index=list(graph.edges(keys=True)))
        n = len(_idx.index.difference(pd.Series(attr).index))
        if n > 0:
            raise DataError("Incomplete data: {n:,} edges are missing in `attr'.")
    _data = {k: attr.get(k, fill_value) for k in graph.edges(keys=True)}
    nx.set_edge_attributes(graph, _data, attr_name)
    return graph


def get_nested_edge_betweenness(
    graph: nx.Graph,
    nested_node_sample: dict,
    weight: str | None = None,
    *,
    cutoff: float | None = None,
) -> pd.Series:
    """Edge usage counts from a nested (origin → sampled-destinations) sample.

    For each origin in `nested_node_sample`, runs a single-source Dijkstra
    on `graph` (via `scipy.sparse.csgraph.dijkstra` with `return_predecessors`),
    walks the predecessor chain from each sampled destination back to the
    origin, and adds 1 to every edge on the path. The result is the
    weighted sum over all sampled OD pairs — a "traffic-stress"-style edge
    usage count, not classical Brandes' betweenness.

    Repeated destinations in the per-origin sample naturally count multiple
    times (each occurrence adds 1 to its path's edges), so weight comes
    from the upstream sampling step's destination distribution.

    Args:
        graph: networkx graph (any variant). MultiGraph parallel edges with the
            same `(u, v)` collapse to the min-`weight` edge for routing,
            and the chosen key is the one credited in the output.
        nested_node_sample: `{origin_node -> array_of_dest_nodes}`, typically
            from `traffic_flows.nested_node_sample`. Origins are unique;
            duplicate destinations within an origin's array are fine.
        weight: edge attribute name to use as the per-edge cost (e.g.
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

    if weight is None:
        raise ValueError("`weight` is required: traffic-flow sampling needs a real edge cost.")
    is_multi = graph.is_multigraph()
    csr, nx_to_seq, seq_to_nx, parallel_keys = _graph_to_csr(
        graph, weight, return_parallel_keys=True
    )
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


def aggregate_nodes_to_edges(
    df_nodes: pd.DataFrame,
    cols: list[str],
    node_edge_relations: str | nx.Graph,
    *,
    aggregator: str,
) -> pd.DataFrame:
    """Aggregate node-level features onto the edges they touch (sum or mean).

    Args:
        df_nodes: list of nodes, supplied as a DataFrame.
        cols: list of columns in df_nodes to be mapped to edges.
        node_edge_relations: if str, must list the edges belonging to each node in column
            'node_edge_relations' in df_nodes, separated by a comma (,). Otherwise, supply an
            nx.Graph where the ID of each node corresponds to the index in df_nodes.
        aggregator: how to aggregate values from different nodes onto a single edge.
            One of `'sum'`, `'mean'`, `'median'`.
    """

    collected_edge_information: dict = {}
    df_nodes.apply(
        lambda row: _add_to_edge_info(row, collected_edge_information, cols, node_edge_relations),
        axis=1,
    )
    for k, d in collected_edge_information.items():
        for col, values in d.items():
            if aggregator == "sum":
                collected_edge_information[k][col] = sum(values)
            elif aggregator == "mean":
                collected_edge_information[k][col] = float(np.average(values))
            elif aggregator == "median":
                collected_edge_information[k][col] = float(np.median(values))
            else:
                raise NotImplementedError(f"aggregator `{aggregator}` is not implemented.")
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
    """Per-direction lane count for a directed edge. OSM-specific: reads the
    OSM `lanes` and `oneway` tag conventions.

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
    "n_streets": _int_via_float,
    "is_t_junction": _int_via_float,
    "is_4way": _int_via_float,
    "is_t_junction_major": _int_via_float,
    "is_4way_major": _int_via_float,
    "is_t_junction_anchor": _int_via_float,
    "is_4way_anchor": _int_via_float,
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

# Per-edge attribute dtypes used downstream by aperta but not covered by
# OSMnx's `default_edge_dtypes`. Without these, the attrs round-trip as
# strings and arithmetic breaks downstream.
#
# Sources:
#   - `lanes_per_direction`: written by `consolidate_intersections`.
#   - `density_norm`, `is_t_junction`, `is_4way`, `is_traffic_signal`:
#     endpoint-mean of the per-node values, written by per-project density
#     prep steps (see `examples/extended/prepare/5_density.py`). Values
#     live in {0, 0.5, 1} on edges (mean of {0, 1} node flags), so `float`
#     casts cleanly. Listed here so per-edge-feature aggregation (e.g.
#     `calibration.calibrate_edge_weights`) doesn't choke on string-typed
#     numerics post-reload.
CONSOLIDATED_EDGE_DTYPES: dict[str, Callable] = {
    "lanes_per_direction": float,
    "density_norm": float,
    "is_t_junction": float,
    "is_4way": float,
    "is_traffic_signal": float,
}


def load_consolidated_graphml(
    filepath, *, node_dtypes: dict | None = None, edge_dtypes: dict | None = None, **kwargs
):
    """Load a graphml saved by `consolidate_intersections`, casting our
    custom `is_*` / `*_highway_rank` attrs back to float. OSM-specific —
    expects graphs in the OSMnx node/edge attribute convention; for non-OSM
    pickle/graphml saves, call `nx.read_graphml` directly and apply your own
    project-specific dtype casts.

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
    `flag_node_intersection_topology` (`n_streets`, `is_t_junction`,
    `is_4way`) and `flag_node_osm_classification` (`max_highway_rank`,
    `min_highway_rank`, `is_t_junction_major`, `is_4way_major`,
    `is_t_junction_anchor`, `is_4way_anchor`), plus one `is_<name>` per
    requested obstacle type, plus `is_roundabout` if
    `detect_roundabouts=True`. Edge `highway` lists from the consolidation
    are collapsed to the highest-rank single string via
    `collapse_osm_highway_lists_by_rank`. Each edge also gets
    `lanes_per_direction` (the OSM `lanes` tag corrected for two-way
    roads — see `lanes_per_direction()`). **Node IDs are new integer IDs**
    (per OSMnx behaviour) — caller must re-snap geo units to the
    consolidated graph.

    OSM-specific. The wrapper consumes OSMnx graphs, reads the OSM
    `highway` / `junction` / `traffic_signals` tag conventions, and
    writes `OSM_HIGHWAY_RANKS`-derived per-node attributes. Projects
    working with non-OSM road networks (e.g. LUMOS's simplified 3-tier
    network) should skip consolidation entirely and call
    `flag_node_intersection_topology` directly for the network-agnostic
    `is_t_junction` / `is_4way` flags.

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
    #    intersection topology + OSM highway-rank classification.
    collapse_osm_highway_lists_by_rank(consolidated)
    flag_node_intersection_topology(consolidated)
    flag_node_osm_classification(consolidated)

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


def flag_node_intersection_topology(graph: nx.Graph) -> None:
    """Mutate `graph` in place to add per-node **topology-only** intersection
    flags. Network-agnostic — works on any graph regardless of where it came
    from (OSM, a custom road dataset, a synthetic graph) since it inspects
    only neighbour count, not edge tags.

    Per-node attributes written:

    - `n_streets` — number of distinct neighbour nodes (degree in the
      undirected sense, ignoring edge direction and parallel edges). The
      "physical" intersection size: 1 = dead-end, 2 = passthrough,
      3 = T-junction, 4+ = 4-way intersection or denser.
    - `is_t_junction` — 1 if `n_streets == 3`, else 0.
    - `is_4way` — 1 if `n_streets >= 4`, else 0.

    `is_t_junction` and `is_4way` are **mutually exclusive** — a 4-way
    node carries only `is_4way`. (Degree 1 / 2 nodes — leaves and
    passthroughs — get neither.)

    OSM-tag-based per-node classifications (highway rank, `_major` /
    `_anchor` variants) live in the companion function
    `flag_node_osm_classification`, which must be called AFTER this one
    if you want the rank-conditional variants (since they're conditional
    on `is_t_junction` / `is_4way`). A project working with a non-OSM
    road network (e.g., LUMOS's simplified 3-tier networks) can call
    this function alone and supply its own project-specific classifier
    on top.

    Per-node obstacle flags (`is_traffic_signal`, `is_stop`, etc.) live
    in `consolidate_intersections`, which is also OSM-specific.
    """
    is_directed = graph.is_directed()

    for nid in graph.nodes():
        if is_directed:
            neighbours = set(graph.predecessors(nid)) | set(graph.successors(nid))
        else:
            neighbours = set(graph.neighbors(nid))
        n_streets = len(neighbours)
        graph.nodes[nid]["n_streets"] = n_streets
        graph.nodes[nid]["is_t_junction"] = int(n_streets == 3)
        graph.nodes[nid]["is_4way"] = int(n_streets >= 4)


def flag_node_osm_classification(graph: nx.Graph) -> None:
    """Mutate `graph` in place to add **OSM-tag-based** per-node classification
    attributes derived from the per-edge `highway` tag (OSM convention).

    Reads the per-edge `highway` attribute via `OSM_HIGHWAY_RANKS` and the
    per-node `is_t_junction` / `is_4way` flags. Call
    `flag_node_intersection_topology` first so those flags are present.

    Per-node attributes written:

    - `max_highway_rank` — max `OSM_HIGHWAY_RANKS` value over edges
      incident to this node (`-1` for unknown / not-a-real-road, e.g.
      footways).
    - `min_highway_rank` — same with min.
    - `is_t_junction_major` — `is_t_junction` AND `min_highway_rank >= 3`
      (every incident edge is tertiary or better — a "fully classified"
      T-junction with no minor branches).
    - `is_4way_major` — `is_4way` AND `min_highway_rank >= 3`.
    - `is_t_junction_anchor` — `is_t_junction` AND `max_highway_rank >= 3`
      AND `min_highway_rank <= 5` (at least one tertiary-or-better edge,
      and not exclusively trunk / motorway — a trip-anchor T-junction
      where car trips can naturally begin or end).
    - `is_4way_anchor` — `is_4way` AND the same rank condition.

    The two OSM-derived intersection tiers — `_major`, `_anchor` —
    capture progressively different selection criteria for downstream
    snap targets and edge-weight features:

    - **`_major`**: intersections where every connecting street is at
      least tertiary class. Used when only "real road" junctions matter
      (e.g., generating a coarse zone-snap candidate set).
    - **`_anchor`**: intersections that touch at least one main road
      (tertiary or better) and aren't purely highway interchanges. Used
      as priority snap targets for car routing — trips begin and end at
      anchor nodes.

    **Non-OSM networks**: this function only fires on graphs whose edges
    carry the OSM `highway` attribute (or `OSM_HIGHWAY_RANKS`-compatible
    string values for it). For networks with a different classification
    scheme (e.g., LUMOS's simplified 3-tier network with `highway` /
    `autostrasse` / `main_street` tiers), write a project-specific
    classifier that follows the same per-node-attribute pattern.
    """
    is_multi = graph.is_multigraph()

    # Per-node max / min highway rank from incident edges.
    node_max = {n: float("-inf") for n in graph.nodes}
    node_min = {n: float("inf") for n in graph.nodes}
    if is_multi:
        for u, v, _, d in graph.edges(keys=True, data=True):
            rank = _osm_highway_rank(d.get("highway"))
            for endpoint in (u, v):
                if rank > node_max[endpoint]:
                    node_max[endpoint] = rank
                if rank < node_min[endpoint]:
                    node_min[endpoint] = rank
    else:
        for u, v, d in graph.edges(data=True):
            rank = _osm_highway_rank(d.get("highway"))
            for endpoint in (u, v):
                if rank > node_max[endpoint]:
                    node_max[endpoint] = rank
                if rank < node_min[endpoint]:
                    node_min[endpoint] = rank

    for nid in graph.nodes():
        is_t = bool(graph.nodes[nid].get("is_t_junction", 0))
        is_4 = bool(graph.nodes[nid].get("is_4way", 0))
        mx = node_max[nid]
        mn = node_min[nid]
        max_rank = int(mx) if mx != float("-inf") else -1
        min_rank = int(mn) if mn != float("inf") else -1
        is_major = min_rank >= 3
        is_anchor = (max_rank >= 3) and (min_rank <= 5)

        graph.nodes[nid]["max_highway_rank"] = max_rank
        graph.nodes[nid]["min_highway_rank"] = min_rank
        graph.nodes[nid]["is_t_junction_major"] = int(is_t and is_major)
        graph.nodes[nid]["is_4way_major"] = int(is_4 and is_major)
        graph.nodes[nid]["is_t_junction_anchor"] = int(is_t and is_anchor)
        graph.nodes[nid]["is_4way_anchor"] = int(is_4 and is_anchor)


def _snap_to_subset(
    points: gpd.GeoDataFrame,
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
    points: gpd.GeoDataFrame,
    graph: nx.Graph,
    *,
    max_distance: float | None = None,
    eligible_node_ids: set | list | pd.Index | None = None,
    eligible_node_flag: str | None = None,
    priority_node_ids: set | list | pd.Index | None = None,
    priority_node_flag: str | None = None,
    priority_max_distance: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Snap each row in `points` to its nearest node in `graph`.

    For each point, finds the closest network node by Euclidean distance and
    returns both the node ID and the distance. The point CRS and the graph
    coordinates must already agree — this function does no reprojection.

    Two-pass snapping. When either `priority_node_ids` or
    `priority_node_flag` is given, the function first tries to snap each
    point to its nearest **priority** node within `priority_max_distance`.
    Points that don't find a priority node in range fall through to a
    second pass that snaps to the nearest **eligible** node within
    `max_distance`. Use this to prefer high-quality snap targets (real
    intersections, well-connected nodes) while still guaranteeing every
    point gets snapped to something safe.

    Network nodes must carry `x` and `y` attributes (aperta convention).
    Typical sources of such graphs are OSMnx (`ox.project_graph(...)` produces
    nodes with `x` / `y` in the target CRS) or aperta's own
    `network_processing` builders.

    Args:
        points: GeoDataFrame of points to snap. Output is indexed by
            `points.index`.
        graph: NetworkX (or compatible) graph with `x` / `y` node attributes.
        max_distance: optional cap. Points farther than this from every
            eligible node return `NaN` for both ID and distance. `None`
            means no cap.
        eligible_node_ids: optional restriction — only nodes in this set are
            considered as snap targets. Use with `aggregate_edges_to_nodes`
            to filter out structurally undesirable snap targets (e.g.,
            motorway nodes, dead-end nodes, pedestrian-only paths for car
            analyses). `None` (default) considers all graph nodes.
        eligible_node_flag: optional alternative to `eligible_node_ids` —
            name of a per-node boolean attribute on `graph` that marks
            eligible snap targets (e.g., the `snap_eligible_flag` written
            by `prepare_network`). Useful when the graph has been prepared
            elsewhere (e.g., loaded from `.graphml`) and the eligible-set
            travels with it as a node attribute. Ignored if
            `eligible_node_ids` is also given.
        priority_node_ids: optional **priority** snap targets — high-quality
            nodes preferred over plain eligible nodes when within
            `priority_max_distance`. Use `prepared.snap_priority_nodes`
            from `prepare_network`. When given (or `priority_node_flag` is),
            the function runs in two-pass mode: priority first within
            `priority_max_distance`, eligible fallback within
            `max_distance`. Points beyond the priority radius fall back to
            the eligible second pass.
        priority_node_flag: optional alternative to `priority_node_ids` —
            name of a per-node bool attribute on `graph` (e.g.,
            `prepared.snap_priority_flag`). Ignored if `priority_node_ids`
            is also given.
        priority_max_distance: optional cap on the priority first pass.
            Points farther than this from every priority node fall through
            to the eligible second pass. `None` means no priority-side cap
            (every point gets snapped to a priority node if any exist).

    Returns:
        Tuple `(node_ids, distances)`:
            - `node_ids`: `pd.Series` of nearest-node IDs, indexed by `points.index`.
            - `distances`: `pd.Series` of distances (in CRS units), indexed by
              `points.index`.
    """
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

    # Resolve priority subset (optional). Empty priority set is fine — it
    # just yields all-NaN from the first pass, so every point falls through
    # to the eligible second pass.
    priority_requested = priority_node_ids is not None or priority_node_flag is not None
    if not priority_requested:
        return _snap_to_subset(points, graph, eligible_node_list, max_distance=max_distance)

    if priority_node_ids is None:
        priority_set = {n for n, data in graph.nodes(data=True) if data.get(priority_node_flag)}
    else:
        priority_set = set(priority_node_ids)
    priority_node_list = [n for n in graph.nodes if n in priority_set]

    # First pass: priority within priority_max_distance.
    pri_ids, pri_dists = _snap_to_subset(
        points, graph, priority_node_list, max_distance=priority_max_distance
    )
    unmatched = pri_ids.isna()
    if not unmatched.any():
        return pri_ids, pri_dists

    # Second pass: eligible (with broader max_distance) for the unmatched.
    fallback_points = points.loc[unmatched]
    elig_ids, elig_dists = _snap_to_subset(
        fallback_points, graph, eligible_node_list, max_distance=max_distance
    )
    pri_ids = pri_ids.copy()
    pri_dists = pri_dists.copy()
    pri_ids.loc[unmatched] = elig_ids
    pri_dists.loc[unmatched] = elig_dists
    return pri_ids, pri_dists


def aggregate_edges_to_nodes(
    graph: nx.Graph,
    edge_attribute: str | Callable,
    *,
    aggregator: str | Callable = "max",
) -> pd.Series:
    """For each node in `graph`, aggregate `edge_attribute` across its connected edges.

    The inverse of `aggregate_nodes_to_edges` (which propagates per-node
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


# --- Mode-aware graph preparation -------------------------------------------

BaseMode = Literal["walk", "bike", "car"]
Directedness = Literal["undirected", "directed_scc"]


# Per-mode default configuration for `prepare_network`. The rationale for each
# choice is documented in the design memo `project_aperta_node_snap_trap_fix`:
#   - walk: undirected (walking does not respect one-ways) + `network_type='all'`
#     (avoids stripping pedestrian paths topologically connected through highway
#     nodes — the Cambridge MA pitfall) + motorway/trunk excluded at cost time.
#   - bike: undirected (untagged contraflow is more common than legit-bike-only
#     one-ways; the permissive default is the right error budget) + `'bike'`.
#   - car:  directed with snap-to-largest-SCC (one-ways are semantically real
#     for cars; flipping them produces wrong-direction routes) + `'drive'`.
# OSM-specific. `network_type` values (`'all'`, `'bike'`, `'drive'`) are OSMnx
# network filters, and `cost_excluded_tags` reads the OSM `highway` tag.
# Projects working with non-OSM road networks should override these settings
# explicitly when calling `prepare_network` (or skip it and write the
# `is_snap_eligible_*` / `cost_excluded_*` decorations directly).
MODE_DEFAULTS: dict[BaseMode, dict] = {
    "walk": {
        "network_type": "all",
        "directedness": "undirected",
        "cost_excluded_tags": frozenset({"motorway", "motorway_link", "trunk", "trunk_link"}),
    },
    "bike": {
        "network_type": "bike",
        "directedness": "undirected",
        "cost_excluded_tags": frozenset(),
    },
    "car": {
        "network_type": "drive",
        "directedness": "directed_scc",
        "cost_excluded_tags": frozenset(),
    },
}


# Per-mode default predicates for "priority" snap-target classification.
# A priority node is a high-quality snap target — typically a well-connected
# intersection — used as the first-pass choice in two-pass snapping. Cells
# / zones snap to a priority node within a tight radius if one exists, and
# fall back to the broader eligible set otherwise. The walk / bike predicate
# reads only `is_4way` (topology, from `flag_node_intersection_topology`).
# The car predicate reads `is_*_anchor` attributes, which depend on the OSM
# `highway` tag and are written by `flag_node_osm_classification` — for
# non-OSM road networks a project-specific car predicate is required.
def _priority_walk_bike(node_data: dict) -> bool:
    """4-way intersections only (well-connected, regardless of road class)."""
    return bool(node_data.get("is_4way", 0))


def _priority_car(node_data: dict) -> bool:
    """Anchor T-junctions and 4-way intersections: at least one tertiary+
    road touching, AND not a pure trunk/motorway interchange. OSM-specific:
    reads `is_*_anchor` written by `flag_node_osm_classification`."""
    return bool(node_data.get("is_t_junction_anchor", 0) or node_data.get("is_4way_anchor", 0))


MODE_PRIORITY_DEFAULTS: dict[BaseMode, Callable[[dict], bool]] = {
    "walk": _priority_walk_bike,
    "bike": _priority_walk_bike,
    "car": _priority_car,
}


@dataclass(frozen=True)
class PreparedGraph:
    """Routing-ready graph plus the in-session metadata derived by `prepare_network`.

    `prepare_network` decorates the underlying `graph` in place with two
    boolean attributes — one per node (`snap_eligible_flag`) and one per
    edge (`cost_excluded_flag`) — so the trap-fix information survives
    `.graphml` roundtripping and is available to downstream consumers that
    receive only the graph (e.g., Pandana). The `PreparedGraph` itself is a
    thin session-time wrapper that memoizes the snap-eligible node set for
    direct 1:1 hand-off to `snap_to_network_nodes` and records the
    resolved-config metadata for introspection.

    Attributes:
        graph: The routing graph. `MultiGraph` (undirected) when
            `directedness='undirected'`, `MultiDiGraph` (directed) when
            `directedness='directed_scc'`.
        snap_eligible_nodes: Node IDs from which every node in the set is
            mutually reachable under the chosen directedness — the largest
            connected component for `'undirected'`, the largest strongly
            connected component for `'directed_scc'`. Pass as
            `eligible_node_ids=` to `snap_to_network_nodes` to prevent cells
            from snapping to trapped nodes. Memoized; the same set is also
            recoverable from the per-node `snap_eligible_flag` attribute on
            `graph`.
        snap_eligible_flag: Name of the per-node boolean attribute written
            onto `graph` (default `f"is_snap_eligible_{mode}"`).
        snap_priority_nodes: Subset of `snap_eligible_nodes` classified as
            high-quality snap targets (e.g., real intersections of suitable
            road class). Empty if no priority predicate is in effect. Pass
            as `priority_node_ids=` to `snap_to_network_nodes` for two-pass
            snapping (priority first within a tight radius, eligible
            fallback otherwise).
        snap_priority_flag: Name of the per-node boolean attribute written
            onto `graph` (default `f"is_snap_priority_{mode}"`). False for
            every node when no priority predicate is in effect.
        cost_excluded_flag: Name of the per-edge boolean attribute written
            onto `graph` (default `f"cost_excluded_{mode}"`). Mode-specific
            cost functions consult this flag to assign cost = ∞.
        mode: The user-supplied mode label (free-form string).
        directedness: Resolved directedness setting.
        network_type: Resolved OSMnx network-type string (recorded for
            metadata / warnings; `prepare_network` does not re-fetch).
    """

    graph: nx.Graph
    snap_eligible_nodes: frozenset
    snap_eligible_flag: str
    snap_priority_nodes: frozenset
    snap_priority_flag: str
    cost_excluded_flag: str
    mode: str
    directedness: Directedness
    network_type: str


def prepare_network(
    graph: nx.MultiDiGraph,
    mode: str,
    *,
    base_mode: BaseMode | None = None,
    directedness: Directedness | None = None,
    network_type: str | None = None,
    cost_excluded_tags: Iterable[str] | None = None,
    snap_eligible_flag: str | None = None,
    cost_excluded_flag: str | None = None,
    priority_node_filter: Callable[[dict], bool] | None = None,
    snap_priority_flag: str | None = None,
) -> PreparedGraph:
    """Apply mode-aware graph preparation: directedness + snap-eligibility + cost-exclusion flags.

    Wraps an already-fetched (and typically already-consolidated) graph to
    produce a `PreparedGraph` ready for trap-free routing. The two axes —
    `directedness` and `network_type` — each have per-mode defaults
    (`MODE_DEFAULTS`) that callers can override. The trap-fix outputs
    (`snap_eligible_flag` per node, `cost_excluded_flag` per edge) are
    written onto the underlying graph in place so they survive `.graphml`
    roundtripping and are available to downstream consumers that don't
    know about `PreparedGraph`.

    The `mode` argument is a free-form string label used for default flag
    names and for warning-policy lookup. Common cases (`'walk'`, `'bike'`,
    `'car'`) pick up defaults directly from `MODE_DEFAULTS`. Subtype /
    context labels (e.g., `'car_night'`, `'ebike25'`) work too; pass
    `base_mode` to inherit defaults from a known base, and override
    individual flags as needed. Default flag names embed `mode`, so
    multiple calls on the same graph with different mode labels accumulate
    independent decorations (`is_snap_eligible_car_peak`,
    `is_snap_eligible_car_night`, etc.) instead of overwriting each other.

    Args:
        graph: Routing graph, typically the output of `fetch_network` +
            `consolidate_intersections`. Must be directed
            (`nx.MultiDiGraph`) when `directedness='directed_scc'` is
            chosen.
        mode: Free-form mode label. Used to derive default flag names and,
            if it matches a `BaseMode` (`'walk'`/`'bike'`/`'car'`), to pull
            defaults from `MODE_DEFAULTS`.
        base_mode: Optional `BaseMode` for `MODE_DEFAULTS` lookup when
            `mode` itself is not a `BaseMode` (e.g., `mode='car_night'`,
            `base_mode='car'`). Resolution order: `base_mode` if given,
            else `mode` if in `MODE_DEFAULTS`, else no defaults available
            and all flags must be supplied explicitly.
        directedness: `'undirected'` applies `to_undirected()` and snaps to
            the largest connected component. `'directed_scc'` keeps the
            graph directed and snaps to the largest strongly connected
            component. `None` resolves from defaults.
        network_type: The OSMnx `network_type` used to fetch `graph`. Used
            only for warning-policy decisions and recorded as metadata;
            this function does not re-fetch. `None` resolves from defaults.
        cost_excluded_tags: Highway-tag values to mark for cost = ∞
            downstream (e.g., `{'motorway', 'trunk'}` for walk on
            `network_type='all'`). `None` resolves from defaults; an empty
            iterable is a valid explicit override.
        snap_eligible_flag: Name of the per-node bool attribute to write.
            `None` defaults to `f"is_snap_eligible_{mode}"`.
        cost_excluded_flag: Name of the per-edge bool attribute to write.
            `None` defaults to `f"cost_excluded_{mode}"`.
        priority_node_filter: Optional predicate `(node_data) -> bool` that
            classifies each node as a "priority" snap target (high-quality
            intersection where trips naturally begin / end). Used for the
            first pass of two-pass snapping; nodes that don't satisfy this
            but ARE in `snap_eligible_nodes` remain available as fallback.
            `None` resolves from `MODE_PRIORITY_DEFAULTS` if a base mode is
            available, otherwise leaves the priority set empty. The
            predicate consults node attributes typically written by
            `flag_node_intersection_topology` and (for OSM-derived flags
            like `is_*_anchor`) `flag_node_osm_classification`.
        snap_priority_flag: Name of the per-node bool attribute to write
            for priority nodes. `None` defaults to `f"is_snap_priority_{mode}"`.

    Returns:
        A `PreparedGraph` carrying the (possibly transformed) routing
        graph, the snap-eligible node set (also written per-node onto the
        graph), and the resolved metadata.

    Warns:
        `UserWarning` on combinations known to silently produce wrong
        answers (e.g., `walk + network_type='walk'` strips pedestrian
        paths; `car + directedness='undirected'` routes the wrong way down
        one-way streets). Warnings are keyed by the resolved base mode, so
        they fire for subtypes too (`'car_night'` with `base_mode='car'`
        still warns on `directedness='undirected'`). Deviations from a
        mode default that are not known-problematic do NOT warn.

    Raises:
        ValueError: if `mode` has no defaults available (neither
            `base_mode` nor `mode in MODE_DEFAULTS`) and any of
            `directedness` / `network_type` / `cost_excluded_tags` is
            unspecified; or if `directedness='directed_scc'` is requested
            on an already-undirected graph.
    """
    if base_mode is not None and base_mode not in MODE_DEFAULTS:
        raise ValueError(f"`base_mode` must be one of {list(MODE_DEFAULTS)}; got {base_mode!r}.")

    if base_mode is not None:
        defaults: dict | None = MODE_DEFAULTS[base_mode]
        effective_base: BaseMode | None = base_mode
    elif mode in MODE_DEFAULTS:
        defaults = MODE_DEFAULTS[cast(BaseMode, mode)]
        effective_base = cast(BaseMode, mode)
    else:
        defaults = None
        effective_base = None

    def _resolve(flag_value, default_key: str, flag_name: str):
        if flag_value is not None:
            return flag_value
        if defaults is None:
            raise ValueError(
                f"`{flag_name}` not provided and no defaults are available for "
                f"mode={mode!r}. Pass `base_mode` to inherit from a known base, "
                f"or supply `{flag_name}` explicitly."
            )
        return defaults[default_key]

    directedness_r: Directedness = _resolve(directedness, "directedness", "directedness")
    network_type_r: str = _resolve(network_type, "network_type", "network_type")
    cost_excluded_tags_r: frozenset = frozenset(
        _resolve(cost_excluded_tags, "cost_excluded_tags", "cost_excluded_tags")
    )

    snap_eligible_flag_r: str = (
        snap_eligible_flag if snap_eligible_flag is not None else f"is_snap_eligible_{mode}"
    )
    cost_excluded_flag_r: str = (
        cost_excluded_flag if cost_excluded_flag is not None else f"cost_excluded_{mode}"
    )
    snap_priority_flag_r: str = (
        snap_priority_flag if snap_priority_flag is not None else f"is_snap_priority_{mode}"
    )
    # Resolve priority predicate: explicit arg wins; else mode default; else None.
    priority_filter_r: Callable[[dict], bool] | None
    if priority_node_filter is not None:
        priority_filter_r = priority_node_filter
    elif effective_base is not None:
        priority_filter_r = MODE_PRIORITY_DEFAULTS.get(effective_base)
    else:
        priority_filter_r = None

    if effective_base is not None:
        _check_combination(effective_base, directedness_r, network_type_r, cost_excluded_tags_r)

    if directedness_r == "undirected":
        prepared_graph: nx.Graph = graph.to_undirected()
    else:
        if not graph.is_directed():
            raise ValueError(
                "`directedness='directed_scc'` requires a directed input graph; "
                "got an undirected graph."
            )
        prepared_graph = graph

    # Write per-edge cost_excluded_flag FIRST so the snap-eligible
    # computation below can mask cost-excluded edges out of its topology
    # analysis. Without this ordering, a node connected to the rest of
    # the largest CC only via cost-excluded edges (e.g., a pedestrian
    # cell snapped to a motorway-ramp node) would appear topologically
    # eligible but route to inf at every query — silent zero-accessibility
    # outliers.
    if prepared_graph.is_multigraph():
        for u, v, k, edata in prepared_graph.edges(keys=True, data=True):
            prepared_graph.edges[u, v, k][cost_excluded_flag_r] = _osm_highway_in_excluded(
                edata.get("highway"), cost_excluded_tags_r
            )
    else:
        for u, v, edata in prepared_graph.edges(data=True):
            prepared_graph.edges[u, v][cost_excluded_flag_r] = _osm_highway_in_excluded(
                edata.get("highway"), cost_excluded_tags_r
            )

    snap_nodes = _compute_snap_eligible_nodes(prepared_graph, directedness_r, cost_excluded_flag_r)

    # Compute priority nodes (subset of eligible): nodes that satisfy
    # `priority_filter_r` on their per-node attributes. Empty if no filter.
    if priority_filter_r is None:
        priority_nodes: frozenset = frozenset()
    else:
        priority_nodes = frozenset(
            n for n in snap_nodes if priority_filter_r(prepared_graph.nodes[n])
        )

    # Decorate the graph in place so the trap-fix information rides along
    # with the graph (e.g., through .graphml roundtripping) and downstream
    # consumers can use it without knowing about `PreparedGraph`.
    for n in prepared_graph.nodes:
        prepared_graph.nodes[n][snap_eligible_flag_r] = n in snap_nodes
        prepared_graph.nodes[n][snap_priority_flag_r] = n in priority_nodes

    return PreparedGraph(
        graph=prepared_graph,
        snap_eligible_nodes=snap_nodes,
        snap_eligible_flag=snap_eligible_flag_r,
        snap_priority_nodes=priority_nodes,
        snap_priority_flag=snap_priority_flag_r,
        cost_excluded_flag=cost_excluded_flag_r,
        mode=mode,
        directedness=directedness_r,
        network_type=network_type_r,
    )


def _compute_snap_eligible_nodes(
    graph: nx.Graph,
    directedness: Directedness,
    cost_excluded_flag: str,
) -> frozenset:
    """Largest CC (undirected) or SCC (directed) of the *cost-masked* subgraph.

    The cost-masked subgraph contains only edges where the per-edge attribute
    `cost_excluded_flag` is absent or `False`. Computing eligibility on this
    subgraph (rather than on the raw topology) is what distinguishes
    "topologically reachable" from "reachable at finite routing cost." A node
    connected to the rest of the largest CC only via cost-excluded edges
    (e.g., a pedestrian-network node at a motorway interchange) is in the
    largest topological CC but practically unreachable at routing time — the
    cost-masked subgraph correctly identifies it as ineligible.

    When `cost_excluded_tags` is empty (the default for bike + car), no edges
    carry `cost_excluded_flag=True`, so the cost-masked subgraph equals the
    full graph and the result matches plain topological analysis.
    """
    ok_edges: list
    if graph.is_multigraph():
        ok_edges = [
            (u, v, k)
            for u, v, k, d in graph.edges(keys=True, data=True)
            if not d.get(cost_excluded_flag, False)
        ]
    else:
        ok_edges = [
            (u, v) for u, v, d in graph.edges(data=True) if not d.get(cost_excluded_flag, False)
        ]
    subgraph = graph.edge_subgraph(ok_edges)
    if directedness == "undirected":
        components = nx.connected_components(subgraph)
    else:
        components = nx.strongly_connected_components(subgraph)
    largest: set = max(components, key=len, default=set())
    return frozenset(largest)


def _osm_highway_in_excluded(highway_value, excluded: frozenset) -> bool:
    """True if the edge's `highway` tag is in `excluded`. Tolerates list-valued
    tags (post-`consolidate_intersections` edges can carry a list of highway
    strings) and `None`.
    """
    if highway_value is None:
        return False
    if isinstance(highway_value, list):
        return any(v in excluded for v in highway_value)
    return highway_value in excluded


def _check_combination(
    base_mode: BaseMode,
    directedness: Directedness,
    network_type: str,
    cost_excluded_tags: frozenset,
) -> None:
    """Emit `UserWarning` for combinations known to silently produce wrong
    answers. Keyed by the resolved base mode, so warnings fire for subtypes
    (e.g., `'car_night'` with `base_mode='car'` triggers car-rule warnings).
    Deviations from defaults that are not known-problematic stay silent —
    warnings are reserved for actual risk, not "this is not the recommended
    setting".
    """
    if base_mode == "walk":
        if directedness == "directed_scc":
            warnings.warn(
                "walk + directedness='directed_scc' is unnecessarily restrictive; "
                "walking does not need to respect one-ways. Consider 'undirected'.",
                UserWarning,
                stacklevel=3,
            )
        if network_type == "walk":
            warnings.warn(
                "walk + network_type='walk' may strip pedestrian paths topologically "
                "connected to the rest of the network through highway nodes "
                "(the Cambridge MA pitfall). Consider network_type='all' with "
                "cost_excluded_tags={'motorway', 'motorway_link', 'trunk', "
                "'trunk_link'}.",
                UserWarning,
                stacklevel=3,
            )
        if network_type == "all" and not cost_excluded_tags:
            warnings.warn(
                "walk + network_type='all' with empty cost_excluded_tags: walking "
                "will be routed across motorways. Consider excluding {'motorway', "
                "'motorway_link', 'trunk', 'trunk_link'}.",
                UserWarning,
                stacklevel=3,
            )
    elif base_mode == "car":
        if directedness == "undirected":
            warnings.warn(
                "car + directedness='undirected' treats one-way streets as "
                "bidirectional. Was this intentional? Routes may go the wrong way "
                "down one-way streets.",
                UserWarning,
                stacklevel=3,
            )
        if network_type == "all" and not cost_excluded_tags:
            warnings.warn(
                "car + network_type='all' with empty cost_excluded_tags: car routing "
                "may traverse footways, pedestrian paths, or steps. Consider "
                "excluding non-drivable highway tags.",
                UserWarning,
                stacklevel=3,
            )
