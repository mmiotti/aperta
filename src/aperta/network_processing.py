"""Graph-construction and -annotation helpers for transport networks (OSM-flavoured).

Aperta operates on `networkx.Graph` (and its multi/directed variants) as its
canonical graph type. This module supplies the operations that take a raw
(typically OSMnx) graph and produce a richly-annotated one ready for
downstream snap / route work:

- **Intersection consolidation**: `consolidate_intersections` wraps
  `osmnx.consolidate_intersections` but preserves intersection-attribute
  nodes (traffic signals, stop signs, roundabouts) that the OSMnx default
  drops, which matters for any route-level analysis that counts those
  features (Section §3.3 of the toolkit paper). Calls
  `flag_node_intersection_topology` and `flag_node_osm_classification` as
  post-processing.
- **Highway-class machinery**: `OSM_HIGHWAY_RANKS`,
  `collapse_osm_highway_lists_by_rank`, `lanes_per_direction`, the per-node
  topology / OSM-classification flag functions.
- **Edge / node attribute helpers**: aggregate node attributes onto edges
  (`aggregate_nodes_to_edges`), aggregate edge attributes onto nodes
  (`aggregate_edges_to_nodes`), and write attribute values through to a
  graph in a tolerant way (`set_nx_edge_attributes_filled`).
- **Edge betweenness sampling**: `get_nested_edge_betweenness` runs the
  per-origin Dijkstra + path-walking accumulator used by the traffic-flow
  estimation pipeline in `traffic_flows.py`.
- **Graphml round-trip**: `load_consolidated_graphml` with the dtype maps
  that preserve custom integer flags through `.graphml` serialization.

Sibling modules cover the workflow steps that consume this module's output:

- `aperta.network_snap` — snap-target resolution: `snap_to_network_nodes`,
  `insert_projected_nodes`, `nodes_incident_to_edges`, `transport_centroid`,
  `split_edge_at_point`, `split_two_way_edge_at_point`. Re-exported below
  for back-compat.
- `aperta.routing_prep` — mode-aware preparation: `prepare_network`,
  `compute_snap_eligibility`, `PreparedGraph`, `MODE_DEFAULTS`.
  Re-exported below for back-compat.

New code should import directly from `aperta.network_snap` /
`aperta.routing_prep`; the re-exports here exist so existing callsites
(`from aperta.network_processing import prepare_network` etc.) keep working
without churn.
"""

from collections import defaultdict
from typing import Callable, Literal, cast

import networkx as nx
import numpy as np
import pandas as pd

from aperta.errors import DataError

# Back-compat re-exports — snap mechanics and routing prep moved to dedicated
# modules to keep this file under the size limit. Keep these here so existing
# `from aperta.network_processing import …` callsites and `import
# aperta.network_processing as np` namespace aliases continue to work
# unchanged. New code should import from `aperta.network_snap` and
# `aperta.routing_prep` directly.
from aperta.network_snap import (  # noqa: F401  (re-export)
    SplitEdgeResult,
    SplitTwoWayEdgeResult,
    insert_projected_nodes,
    nodes_incident_to_edges,
    snap_to_network_nodes,
    split_edge_at_point,
    split_two_way_edge_at_point,
    transport_centroid,
)
from aperta.routing_prep import (  # noqa: F401  (re-export)
    MODE_DEFAULTS,
    BaseMode,
    Directedness,
    PreparedGraph,
    compute_snap_eligibility,
    prepare_network,
)

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
_DEFAULT_EDGE_ATTR_AGGS = {
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
_DEFAULT_DROP_EDGE_ATTRS = ["name"]


def _int_via_float(value) -> int:
    """`int(float(v))` — tolerates both `'0'`/`'1'` and `'0.0'`/`'1.0'` strings.

    Plain `int()` raises on float-formatted strings (`int('0.0')` → ValueError).
    Used as the cast for graphml-loaded `is_*` flags so older saves (where
    these were written as floats) and newer saves (ints) both round-trip
    cleanly.
    """
    return int(float(value))


def _int_via_bool_or_float(value) -> int:
    """Tolerant graphml-cast for attrs that may have been written as Python
    `bool` (round-trips as `'True'` / `'False'` strings) or as `int` (`'0'`,
    `'1'`, `'0.0'`, `'1.0'`).

    Used by the prefix-scan dtype map for the per-mode flags
    (`is_snap_eligible_<mode>`, `cost_excluded_<mode>`) — `prepare_network`
    writes them as Python `bool` so they stringify as `'True'`/`'False'`,
    while `insert_projected_nodes` writes `is_virtual=1` (int) which
    stringifies as `'1'`. Both must come back as `0` / `1`.
    """
    if isinstance(value, (bool, int)):
        return int(value)
    if value in ("True", "true"):
        return 1
    if value in ("False", "false"):
        return 0
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
_CONSOLIDATED_NODE_DTYPES: dict[str, Callable] = {
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
_CONSOLIDATED_EDGE_DTYPES: dict[str, Callable] = {
    "lanes_per_direction": float,
    "density_norm": float,
    "is_t_junction": float,
    "is_4way": float,
    "is_traffic_signal": float,
}


# Prefix-scan attribute prefixes for dynamic per-mode bool flags. Graphml
# stringifies bool/int values; without a dtype cast on load, `'False'`
# arrives as the literal string and `bool('False') == True` silently
# inverts downstream tests. `prepare_network` / `compute_snap_eligibility`
# embed the mode in the flag name (`is_snap_eligible_walk`,
# `cost_excluded_car_night`, etc.), so we can't hardcode the keys — instead
# we scan the graphml file's `<key>` elements at load time and add an
# `_int_via_float` cast for every attribute whose name starts with one of
# these prefixes (or matches a fixed name like `is_virtual`).
_PREFIX_SCAN_NODE = ("is_snap_eligible_", "is_virtual")
_PREFIX_SCAN_EDGE = ("cost_excluded_",)


def _scan_graphml_keys(
    filepath, prefixes: tuple[str, ...], scope: Literal["node", "edge"]
) -> list[str]:
    """Return the names of `<key for=scope>` graphml attributes that start
    with any of `prefixes` (or equal one — handles the no-suffix `is_virtual`
    case naturally via the prefix-match condition).

    Cheap: parses the `<key>` schema header only, stops before iterating
    nodes / edges (which is where the cost lives on large graphs).
    """
    import xml.etree.ElementTree as ET

    ns = "{http://graphml.graphdrawing.org/xmlns}"
    out: list[str] = []
    for _event, elem in ET.iterparse(filepath, events=("end",)):
        if elem.tag == f"{ns}key" and elem.get("for") == scope:
            name = elem.get("attr.name", "")
            if any(name.startswith(p) or name == p for p in prefixes):
                out.append(name)
        elif elem.tag == f"{ns}graph":
            break
        elem.clear()
    return out


def load_consolidated_graphml(
    filepath, *, node_dtypes: dict | None = None, edge_dtypes: dict | None = None, **kwargs
):
    """Load a graphml saved by `consolidate_intersections`, casting aperta's
    custom `is_*` / `*_highway_rank` / `is_snap_eligible_<mode>` /
    `cost_excluded_<mode>` / `is_virtual` attrs back to their proper types.
    OSM-specific — expects graphs in the OSMnx node/edge attribute
    convention; for non-OSM pickle/graphml saves, call `nx.read_graphml`
    directly and apply your own project-specific dtype casts.

    Thin wrapper around `osmnx.load_graphml`. Three sources of casts get
    merged in (in order; later ones win):

    1. `_CONSOLIDATED_NODE_DTYPES` / `_CONSOLIDATED_EDGE_DTYPES` — fixed
       casts for attrs `consolidate_intersections` writes (e.g.
       `n_streets`, `is_t_junction`, `lanes_per_direction`).
    2. **Prefix-scan** of the graphml `<key>` schema for the dynamic
       per-mode flags written by `prepare_network` /
       `compute_snap_eligibility` / `insert_projected_nodes`:
       `is_snap_eligible_<mode>` and `is_virtual` (per-node);
       `cost_excluded_<mode>` (per-edge). Handles any `mode` label,
       including subtypes like `car_night`, without hardcoding.
    3. Caller-supplied `node_dtypes` / `edge_dtypes` overrides.

    Without the prefix-scan, the per-mode bool flags round-trip as the
    literal strings `'True'` / `'False'`, and `bool('False') == True` would
    silently invert the `if not d.get(cost_excluded_flag, False)` test in
    `_compute_snap_eligible_nodes`.

    Args:
        filepath: path to a `.graphml` produced by `consolidate_intersections`
            or `prepare_network` / `compute_snap_eligibility`.
        node_dtypes: optional override merged on top (caller's values win).
        edge_dtypes: optional override merged on top (caller's values win).
        **kwargs: forwarded to `osmnx.load_graphml`.

    Returns:
        `nx.MultiDiGraph` for graphs saved from a directed source, or
        `nx.MultiGraph` for undirected ones. (OSMnx's docstring claims
        `MultiDiGraph` unconditionally; that's wrong — `nx.read_graphml`
        honors the file's `<graph edgedefault>` attribute.)
    """
    import osmnx as ox

    auto_node = {
        name: _int_via_bool_or_float
        for name in _scan_graphml_keys(filepath, _PREFIX_SCAN_NODE, "node")
    }
    auto_edge = {
        name: _int_via_bool_or_float
        for name in _scan_graphml_keys(filepath, _PREFIX_SCAN_EDGE, "edge")
    }
    merged_node = {**_CONSOLIDATED_NODE_DTYPES, **auto_node, **(node_dtypes or {})}
    merged_edge = {**_CONSOLIDATED_EDGE_DTYPES, **auto_edge, **(edge_dtypes or {})}
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
            list-of-values form. Defaults to `_DEFAULT_DROP_EDGE_ATTRS`.

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
    drop_attrs = _DEFAULT_DROP_EDGE_ATTRS if drop_edge_attrs is None else drop_edge_attrs
    eff_edge_aggs = _DEFAULT_EDGE_ATTR_AGGS if edge_attr_aggs is None else edge_attr_aggs
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
