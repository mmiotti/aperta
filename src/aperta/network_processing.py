"""Graph-construction and -annotation helpers for transport networks.

Aperta operates on `networkx.Graph` (and its multi/directed variants) as its
canonical graph type. This module is **OSM-agnostic** — it consumes whatever
edge/node attributes the caller provides and never reads OSM-specific tag
content. (OSM-aware helpers — tag classification by `highway` rank, OSMnx-
wrapped consolidation, OSM POI fetch — live in `aperta_atlas.osm` /
`aperta_atlas.osm_helpers`. See the 2026-06-07 CHANGELOG entry.)

What's here:

- **Intersection topology flags**: `flag_node_intersection_topology` —
  network-agnostic per-node `n_streets` / `is_t_junction` / `is_4way` from
  graph degree.
- **Spatial primitives**: `snap_features_to_nodes` — KDTree nearest-within-
  radius assignment of point locations to graph nodes, mutating a per-node
  boolean flag. Used for any "label nearby nodes" workflow (POI snap, AOI
  membership, sensor attribution), not OSM-specific.
- **Edge / node attribute helpers**: aggregate node attributes onto edges
  (`aggregate_nodes_to_edges`), aggregate edge attributes onto nodes
  (`aggregate_edges_to_nodes`), and write attribute values through to a
  graph in a tolerant way (`set_nx_edge_attributes_filled`).
- **Edge betweenness sampling**: `get_nested_edge_betweenness` runs the
  per-origin Dijkstra + path-walking accumulator used by the traffic-flow
  estimation pipeline in `traffic_flows.py`.

Sibling modules cover the workflow steps that consume this module's output:

- `aperta.network_snap` — snap-target resolution: `snap_to_network_nodes`,
  `insert_projected_nodes`, and companions.
- `aperta.routing_prep` — mode-aware preparation: `prepare_network`,
  `compute_snap_eligibility`, `PreparedGraph`, `MODE_DEFAULTS`.
"""

import logging
from collections import defaultdict
from typing import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from aperta.errors import DataError


def parse_edge_id(s: str) -> tuple:
    """Parse a `'u:v[:k]'` edge-id string into a `(u, v[, key])` tuple.

    The string form is aperta's canonical on-disk edge encoding (produced
    when serialising graph edges to CSV / GeoPackage). Numeric-looking
    parts are coerced back to int; non-numeric parts pass through as
    strings.
    """
    parts = s.split(":")
    coerced: list = []
    for p in parts:
        try:
            coerced.append(int(p))
        except ValueError:
            coerced.append(p)
    return tuple(coerced)


def _verify_graph_match(graph_keys, prop_keys, what: str) -> None:
    """Verify a property table's index matches the graph's nodes or edges
    (set equality). Strict by default — raises `DataError` listing a sample
    of the asymmetric IDs on either side. Used by `attach_*_properties`."""
    g = set(graph_keys)
    p = set(prop_keys)
    if g == p:
        logging.info(f"Verified {what} index match ({len(g)} {what}).")
        return
    msg = []
    only_g = g - p
    only_p = p - g
    if only_g:
        msg.append(f"{len(only_g)} {what} only in graph (e.g. {list(only_g)[:3]})")
    if only_p:
        msg.append(f"{len(only_p)} {what} only in properties (e.g. {list(only_p)[:3]})")
    raise DataError(f"Graph {what} don't match property table: {'; '.join(msg)}")


def attach_node_properties(graph: nx.Graph, df: pd.DataFrame) -> None:
    """Attach a node-properties DataFrame to `graph` in place.

    `df.index` must be the graph's node IDs (strict set equality). Layer
    property tables from arbitrary sources onto a network — typically used
    after loading a graph from disk to attach companion CSVs.
    """
    df = df.drop(columns=[df.index.name], errors="ignore")
    _verify_graph_match(graph.nodes, df.index, "nodes")
    nx.set_node_attributes(graph, df.to_dict(orient="index"))


def attach_edge_properties(graph: nx.Graph, df: pd.DataFrame) -> None:
    """Attach an edge-properties DataFrame to `graph` in place.

    `df.index` is either:
      - **tuple-keyed** `(u, v)` (non-multigraph) or `(u, v, key)` (multigraph)
        — for callers building the frame programmatically; or
      - **string-keyed** in aperta's canonical `'u:v[:key]'` form — for the
        common case of loading edge properties from a CSV produced via
        edge-id serialisation. Auto-parsed via `parse_edge_id`.

    For non-multigraphs the parsed `key` element is dropped before
    matching. Keys must match the graph's edges with strict set equality.
    """
    df = df.drop(columns=[df.index.name], errors="ignore")
    is_multi = graph.is_multigraph()
    keys = list(df.index)
    if keys and isinstance(keys[0], str):
        keys = [parse_edge_id(s) for s in keys]
    if not is_multi:
        keys = [k[:2] for k in keys]
    graph_edge_keys = list(graph.edges(keys=True)) if is_multi else list(graph.edges)
    _verify_graph_match(graph_edge_keys, keys, "edges")
    edge_attrs = {pk: row.to_dict() for pk, (_, row) in zip(keys, df.iterrows())}
    nx.set_edge_attributes(graph, edge_attrs)


def verify_odm_against_network(
    odm: dict,
    network_nodes: pd.DataFrame | gpd.GeoDataFrame,
    check_destinations: bool = False,
) -> None:
    """Verify ODM origins (and optionally destinations) appear in the network's nodes.

    Use this after loading an ODM (e.g. `Context.get_tiered_odm`, on each
    populated tier dict) to catch ODM / network drift — e.g. an ODM that was
    built against an older snapshot of the network than what the caller is now
    routing on. Pure data check; does no I/O.

    `check_destinations=True` validates the values too — only meaningful for
    `'idx'`-style ODMs whose values are destination node IDs. For value-style
    ODMs (floats, type codes), leave it False.
    """
    valid = set(network_nodes.index)
    bad_origins = [o for o in odm if o not in valid]
    if bad_origins:
        raise DataError(
            f"{len(bad_origins)} ODM origin(s) are not in the network's nodes "
            f"(e.g. {bad_origins[:3]}).",
        )
    if check_destinations:
        bad_dests: set = set()
        for dest_list in odm.values():
            for d in dest_list:
                if d not in valid:
                    bad_dests.add(d)
                    if len(bad_dests) >= 10:
                        break
            if len(bad_dests) >= 10:
                break
        if bad_dests:
            raise DataError(
                f"{'10+' if len(bad_dests) >= 10 else len(bad_dests)} destination ID(s) "
                f"in ODM values are not in the network's nodes "
                f"(e.g. {sorted(bad_dests)[:3]}).",
            )


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
    weight: str,
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
    """Fan a node's feature values out onto each edge it touches.

    For directed graphs (e.g. osmnx `MultiDiGraph`), `graph.edges(node)`
    returns only OUTGOING edges. Without explicitly fetching `in_edges`
    too, each edge would only ever receive a contribution from its source
    node — its destination node's row would never reach it. Mean
    aggregation would then collapse to "value at source", missing the
    half-allocation behavior callers expect.
    """
    if isinstance(node_edge_relations, str):
        edge_ids = node_row[node_edge_relations].split(",")
    elif isinstance(node_edge_relations, nx.Graph):
        if node_edge_relations.is_directed():
            edge_ids = list(node_edge_relations.out_edges(node_row.name, keys=True)) + list(
                node_edge_relations.in_edges(node_row.name, keys=True)
            )
        else:
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
    aggregator: str | Callable,
) -> pd.DataFrame:
    """Aggregate node-level features onto the edges they touch.

    Args:
        df_nodes: list of nodes, supplied as a DataFrame.
        cols: list of columns in df_nodes to be mapped to edges.
        node_edge_relations: if str, must list the edges belonging to each node in column
            'node_edge_relations' in df_nodes, separated by a comma (,). Otherwise, supply an
            nx.Graph where the ID of each node corresponds to the index in df_nodes.
        aggregator: how to aggregate values from different nodes onto a single edge.
            One of `'max'`, `'min'`, `'mean'`, `'sum'`, `'median'`, or a callable that
            takes a 1-D numpy array of per-node values and returns a scalar. String
            aggregators use the nan-safe numpy variants (silently skip NaN values).
    """
    _agg: Callable
    if aggregator == "max":
        _agg = np.nanmax
    elif aggregator == "min":
        _agg = np.nanmin
    elif aggregator == "mean":
        _agg = np.nanmean
    elif aggregator == "sum":
        _agg = np.nansum
    elif aggregator == "median":
        _agg = np.nanmedian
    elif callable(aggregator):
        _agg = aggregator
    else:
        raise ValueError(
            f"Unknown aggregator {aggregator!r}; expected "
            f"'max', 'min', 'mean', 'sum', 'median', or a callable."
        )

    collected_edge_information: dict = {}
    df_nodes.apply(
        lambda row: _add_to_edge_info(row, collected_edge_information, cols, node_edge_relations),
        axis=1,
    )
    for k, d in collected_edge_information.items():
        for col, values in d.items():
            collected_edge_information[k][col] = float(_agg(np.asarray(values, dtype=float)))
    return pd.DataFrame.from_dict(collected_edge_information, orient="index")


def snap_features_to_nodes(
    graph: nx.MultiDiGraph,
    locations: list[tuple[float, float]],
    *,
    flag_name: str,
    max_distance: float,
) -> None:
    """Snap a list of point locations to nearest-within-radius graph
    nodes, writing `is_<flag_name>=1` on matched nodes and
    `is_<flag_name>=0` on all others. Mutates `graph` in place.

    Use case: re-attach OSM-tagged obstacle features (traffic signals,
    stops, give-ways, roundabout midpoints) to a consolidated graph
    after their original OSM nodes were merged away by
    `osmnx.consolidate_intersections`. The same primitive works for
    any "label nearby nodes with a flag" workflow — POI snap, AOI
    membership, sensor-station attribution, etc.

    Uses a `scipy.spatial.KDTree` query with `distance_upper_bound` for
    a strict radius cap; locations whose nearest node is farther than
    `max_distance` are silently dropped (no flag set). For an
    OSMnx-style MultiDiGraph the node coordinates come from each
    node's `x` / `y` attributes; the graph must be in a metric CRS so
    `max_distance` is meaningful in meters.

    Args:
        graph: MultiDiGraph with node `x` and `y` attributes in a
            metric CRS.
        locations: list of `(x, y)` tuples in the same CRS as the
            graph's node coords.
        flag_name: bare attribute name; the boolean is written as
            `f"is_{flag_name}"` (so `flag_name='traffic_signal'`
            produces `is_traffic_signal`).
        max_distance: maximum snap distance, in the graph CRS's units.
            Locations farther than this are dropped.
    """
    node_ids = list(graph.nodes)
    if not node_ids:
        return
    attr_key = f"is_{flag_name}"
    # Zero out the flag for all nodes (idempotent re-runs, well-defined
    # default for nodes we don't snap to).
    for nid in node_ids:
        graph.nodes[nid][attr_key] = 0
    if not locations:
        return
    node_xy = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in node_ids])
    tree = KDTree(node_xy)
    dists, idxs = tree.query(
        np.asarray(locations),
        distance_upper_bound=max_distance,
    )
    # KDTree returns idx == len(node_xy) for misses under
    # distance_upper_bound (and dist == inf).
    valid = (idxs < len(node_ids)) & np.isfinite(dists)
    for i in np.where(valid)[0]:
        graph.nodes[node_ids[int(idxs[i])]][attr_key] = 1


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
    `aperta_atlas.osm.flag_node_osm_classification`, which must be called
    AFTER this one if you want the rank-conditional variants (since
    they're conditional on `is_t_junction` / `is_4way`). A project
    working with a non-OSM road network (e.g., LUMOS's simplified
    3-tier networks) can call this function alone and supply its own
    project-specific classifier on top.

    Per-node obstacle flags (`is_traffic_signal`, `is_stop`, etc.) are
    written by `aperta_atlas.osm.consolidate_intersections`'s obstacle
    re-attachment pass — also OSM-specific and in aperta-lab.
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
        aggregator: `'max'` (default), `'min'`, `'mean'`, `'sum'`, `'median'`,
            or a callable that takes a 1-D numpy array of per-edge values and
            returns a scalar. String aggregators use the nan-safe numpy
            variants (silently skip NaN edge values).

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
    elif aggregator == "median":
        _agg = np.nanmedian
    elif callable(aggregator):
        _agg = aggregator
    else:
        raise ValueError(
            f"Unknown aggregator {aggregator!r}; expected "
            f"'max', 'min', 'mean', 'sum', 'median', or a callable."
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


def smooth_node_attribute(
    graph: nx.Graph,
    node_attr: str,
    *,
    length_scale: float,
    length_attr: str = "length",
    out_attr: str | None = None,
    n_iterations: int = 1,
) -> None:
    """Topology-weighted Gaussian smoothing of a per-node attribute.

    For each node `v`, the smoothed value is a weighted mean of `v`
    itself (distance 0 → weight 1) plus its one-hop graph neighbours,
    each weighted by a Gaussian kernel of the connecting edge length::

        smoothed[v] = ( val[v] + Σ_u w(v, u) · val[u] )  /  ( 1 + Σ_u w(v, u) )
        w(v, u)     = exp(-(edge_length(v, u) / length_scale)² / 2)

    `length_scale` is the Gaussian's 1-σ: neighbours at this edge-length
    distance get weight ≈ 0.61; at `2·length_scale`, ≈ 0.14; at
    `3·length_scale`, ≈ 0.01. Bigger `length_scale` → more smoothing.

    Topology-based — neighbours are graph-incident, not Euclidean-near.
    Avoids the Euclidean-ambiguity failure modes of disk-mean smoothing
    (bridges / parallel roads at different levels / switchbacks all
    stay correctly separated because they're not connected by an edge).
    For directed graphs, uses `nx.all_neighbors` so both successors and
    predecessors contribute. For MultiGraphs, the shortest parallel
    edge sets the distance.

    `n_iterations > 1` re-applies the one-hop kernel; weights compound,
    so a 2-iteration pass implicitly reaches 2-hop neighbours via the
    convolution of two Gaussians — a cheap way to widen the effective
    kernel without running BFS or assembling a graph Laplacian.

    NaN / missing-value semantics:
        * A NaN at `v` itself passes through unchanged.
        * NaN values at neighbours are skipped (don't enter the sum).
        * Nodes without the input attribute are left untouched.

    Args:
        graph: graph with `node_attr` set on a subset of nodes and
            `length_attr` set on every edge.
        node_attr: per-node attribute to smooth.
        length_scale: Gaussian 1-σ in `length_attr` units (typically
            metres for road networks). Pick to match the noise length
            scale you want to suppress.
        length_attr: edge attribute holding edge length. Default
            `'length'`.
        out_attr: where to write the smoothed values. If None,
            overwrites `node_attr`.
        n_iterations: number of one-hop passes. Default 1.

    Mutates `graph` in place via `nx.set_node_attributes`.
    """
    import math

    out_attr = out_attr or node_attr
    is_multi = isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))
    values = dict(nx.get_node_attributes(graph, node_attr))
    inv_2sigma_sq = 1.0 / (2.0 * length_scale * length_scale)

    for _ in range(n_iterations):
        new_values: dict = {}
        for v, val_v in values.items():
            try:
                self_val = float(val_v)
            except (TypeError, ValueError):
                new_values[v] = val_v
                continue
            if not np.isfinite(self_val):
                new_values[v] = val_v
                continue
            # Centre node: distance 0 → Gaussian weight exp(0) = 1.
            weighted_sum = self_val
            weight_sum = 1.0
            for u in nx.all_neighbors(graph, v):
                val_u = values.get(u)
                if val_u is None:
                    continue
                try:
                    nbr_val = float(val_u)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(nbr_val):
                    continue
                ed = graph.get_edge_data(v, u) or graph.get_edge_data(u, v)
                if not ed:
                    continue
                if is_multi:
                    edge_len = min(d.get(length_attr, float("inf")) for d in ed.values())
                else:
                    edge_len = ed.get(length_attr, float("inf"))
                if not np.isfinite(edge_len) or edge_len <= 0:
                    continue
                w = math.exp(-edge_len * edge_len * inv_2sigma_sq)
                weighted_sum += w * nbr_val
                weight_sum += w
            new_values[v] = weighted_sum / weight_sum
        values = new_values

    nx.set_node_attributes(graph, values, out_attr)
