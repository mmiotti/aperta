"""
Routing primitives backed by `scipy.sparse.csgraph.dijkstra`. Input graphs
are `networkx.Graph` (or its multi/directed variants); aperta converts to a
scipy CSR matrix internally per call. No backend choice — scipy throughout.

The library exposes two concerns separately:

1. **Edge weighting** — `apply_edge_weights` runs a user-supplied callable on each
   edge of a graph and writes the result to a named edge attribute. Mode-specific
   behavior (car vs. bike vs. walking, peak vs. off-peak, density-adjusted speeds,
   intersection penalties, etc.) lives in these callables, not in this module.
   `combine_edge_weights` sums multiple per-edge components into a single routing
   weight (e.g. edge travel time + intersection penalty -> total cost).

2. **Routing primitives** — covering the common query shapes:

   - `shortest_distances_from`: single source → all reachable nodes (with optional
     cutoff). Used for accessibility / isochrone calculations.
   - `shortest_distances_pairwise`: full distance matrix between two node lists.
   - `shortest_path_metrics_one_to_one`: paired (origin, destination) routing
     that also aggregates edge features along each path. Used for travel-time
     model calibration and similar trip-by-trip work.
   - `tiered_path_costs` / `tiered_path_aggregate`: bulk many-to-many routing
     across the three-tier OD structure.

Why scipy-only: aperta's routing workflow is one-shot Dijkstra-from-many-origins
on a *live* graph whose edge weights are routinely mutated (calibration loop,
scenario comparison, time-of-day variants). Contraction hierarchies (Pandana,
OSRM) preprocess once and reuse, which fights this workflow. The historical
networkx / igraph backends were dropped in favour of scipy CSR + scipy
`csgraph.dijkstra`, which is competitive in raw speed, zero-preprocessing,
and has a single code path.

A future `RoutingProfile` class will bundle the duration callable + parameters +
graph into one object; for now, callers compose the pieces themselves.
"""

import logging
from collections.abc import Sequence
from typing import Callable, NamedTuple

import networkx as nx
import numpy as np
import pandas as pd

from aperta.errors import DataError
from aperta.od_pairs import TieredODNodePairs, TieredODPairs

# ---------------------------------------------------------------------------
# Edge weighting
# ---------------------------------------------------------------------------


def apply_edge_weights(g: nx.Graph, weight_fn: Callable, weight_name: str, **fn_kwargs) -> None:
    """Apply `weight_fn` to each edge of `g` (mutates `g` in place).

    `weight_fn` receives the edge data dict plus any extra `fn_kwargs`. The
    dict supports `row['key']` access just like a pandas Series, so callables
    written against a GeoDataFrame edge-row pattern work without modification.
    """
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        for _u, _v, _k, data in g.edges(keys=True, data=True):
            data[weight_name] = weight_fn(data, **fn_kwargs)
    else:
        for _u, _v, data in g.edges(data=True):
            data[weight_name] = weight_fn(data, **fn_kwargs)


def combine_edge_weights(g: nx.Graph, source_names: list[str], target_name: str) -> None:
    """Sum multiple per-edge attributes into one combined weight (in place).

    Use case (lumos pattern): travel time and intersection penalty are computed
    separately, stored as `duration_edge_t3` and `duration_node_t3`, then summed
    into the routing weight `duration_t3`.
    """
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        for _u, _v, _k, data in g.edges(keys=True, data=True):
            data[target_name] = sum(float(data[name]) for name in source_names)
    else:
        for _u, _v, data in g.edges(data=True):
            data[target_name] = sum(float(data[name]) for name in source_names)


# ---------------------------------------------------------------------------
# scipy backend (cutoff-aware) — opt-in via `cutoff=` on tiered routing.
#
# When the caller passes `cutoff=T` to `tiered_path_costs` /
# `tiered_path_aggregate`, the per-origin Dijkstra is run via
# scipy.sparse.csgraph.dijkstra with `limit=T`. This truncates the
# Dijkstra at network distance T (in weight units), which can be
# dramatically faster than igraph's no-cutoff exploration when T is
# small relative to graph diameter (verified empirically: ~35× speed-up
# for the Bern + 25 km walk case at T = 2 km, 144k-node graph).
#
# igraph's Graph.distances() doesn't support cutoff; scipy does. See
# memory `aperta-routing-cutoff-design-and-constraints` for the design
# discussion + benchmark.
# ---------------------------------------------------------------------------


def _graph_to_csr(graph: nx.Graph, weight: str, return_parallel_keys: bool = False):
    """Build a scipy CSR matrix from `graph` using `weight` as edge cost.

    For MultiGraph / MultiDiGraph parallels, keeps the minimum-weight edge
    per (u, v) — matches igraph's `distances()` choice.

    Undirected graphs (`nx.Graph` / `nx.MultiGraph`) are emitted as symmetric
    CSR so callers can use scipy's default `directed=True` dijkstra. For
    parallel-edge multigraphs the per-direction min is taken independently,
    so an undirected MultiGraph with one parallel weighted 3 in (u, v) and
    another weighted 5 still gets `csr[u, v] = csr[v, u] = 3` (the user's
    intent for "undirected" is symmetric).

    Args:
        graph: networkx graph (any variant).
        weight: edge attribute name used as the per-edge routing cost.
        return_parallel_keys: when `True`, also return a `parallel_keys`
            dict mapping `(u_seq, v_seq) -> nx_edge_key` for MultiGraph
            inputs (the key of the parallel edge whose weight was kept).
            For non-Multi graphs the dict is empty. Use this when the
            caller needs to attribute path edges back to specific
            MultiGraph edge IDs (e.g. edge betweenness on a MultiDiGraph
            where output keys are `(u, v, k)` triples).

    Returns:
        `(csr, nx_to_seq, seq_to_nx)` by default, or
        `(csr, nx_to_seq, seq_to_nx, parallel_keys)` when
        `return_parallel_keys=True`.
    """
    import scipy.sparse

    node_ids = list(graph.nodes())
    nx_to_seq = {n: i for i, n in enumerate(node_ids)}
    seq_to_nx = np.array(node_ids, dtype=object)
    is_multi = isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))
    is_directed = graph.is_directed()
    track_keys = return_parallel_keys and is_multi

    min_weight: dict[tuple[int, int], float] = {}
    parallel_keys: dict[tuple[int, int], object] = {} if track_keys else {}

    def _update(key_uv: tuple[int, int], w: float, k=None) -> None:
        if key_uv not in min_weight or min_weight[key_uv] > w:
            min_weight[key_uv] = w
            if track_keys:
                parallel_keys[key_uv] = k

    if is_multi:
        for u, v, k, data in graph.edges(keys=True, data=True):
            w = float(data[weight])
            ui, vi = nx_to_seq[u], nx_to_seq[v]
            _update((ui, vi), w, k)
            if not is_directed:
                _update((vi, ui), w, k)
    else:
        for u, v, data in graph.edges(data=True):
            w = float(data[weight])
            ui, vi = nx_to_seq[u], nx_to_seq[v]
            _update((ui, vi), w)
            if not is_directed:
                _update((vi, ui), w)
    n = len(node_ids)
    if min_weight:
        rows = np.fromiter(
            (u for u, _v in min_weight.keys()), dtype=np.int64, count=len(min_weight)
        )
        cols = np.fromiter(
            (v for _u, v in min_weight.keys()), dtype=np.int64, count=len(min_weight)
        )
        data = np.fromiter(min_weight.values(), dtype=float, count=len(min_weight))
    else:
        rows = cols = np.empty(0, dtype=np.int64)
        data = np.empty(0, dtype=float)
    csr = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    if return_parallel_keys:
        return csr, nx_to_seq, seq_to_nx, parallel_keys
    return csr, nx_to_seq, seq_to_nx


def _walk_predecessors_to_path(
    predecessors_row: np.ndarray, origin_seq: int, target_seq: int, seq_to_nx: np.ndarray
) -> list:
    """Reconstruct path (as list of nx node IDs) from scipy's predecessor row.

    Returns `[]` if `target_seq` is unreachable (predecessor chain hits -9999
    before reaching origin). Returns `[origin_id]` if target == origin.
    """
    if target_seq == origin_seq:
        return [seq_to_nx[origin_seq]]
    path_seq = [target_seq]
    while path_seq[-1] != origin_seq:
        p = predecessors_row[path_seq[-1]]
        if p < 0:
            return []  # unreachable
        path_seq.append(int(p))
    path_seq.reverse()
    return [seq_to_nx[s] for s in path_seq]


# ---------------------------------------------------------------------------
# Routing primitives — all backed by `scipy.sparse.csgraph.dijkstra`. The
# input is always an `nx.Graph` (or a multi/directed variant); aperta
# converts to a scipy CSR matrix internally via `_graph_to_csr`.
# ---------------------------------------------------------------------------


def shortest_distances_from(
    graph: nx.Graph,
    origin,
    weight: str,
    cutoff: float | None = None,
) -> dict:
    """Single-source shortest distances from `origin` to all reachable nodes.

    Returns a dict mapping node -> total weight. With `cutoff`, only nodes
    within that weight threshold are returned. Unreachable nodes are omitted.
    """
    import scipy.sparse.csgraph as csg

    csr, nx_to_seq, seq_to_nx = _graph_to_csr(graph, weight)
    limit = cutoff if cutoff is not None else np.inf
    dist_row = csg.dijkstra(
        csr, indices=[nx_to_seq[origin]], limit=limit, return_predecessors=False
    )[0]
    return {seq_to_nx[i]: float(d) for i, d in enumerate(dist_row) if np.isfinite(d)}


def shortest_distances_pairwise(
    graph: nx.Graph,
    origins: list,
    destinations: list,
    weight: str,
    cutoff: float | None = None,
) -> np.ndarray:
    """Distance matrix for `origins` x `destinations`.

    Returns ndarray of shape (len(origins), len(destinations)). Unreachable
    destinations are `np.inf`.
    """
    import scipy.sparse.csgraph as csg

    csr, nx_to_seq, _ = _graph_to_csr(graph, weight)
    limit = cutoff if cutoff is not None else np.inf
    origin_seqs = [nx_to_seq[o] for o in origins]
    dest_seqs = np.fromiter(
        (nx_to_seq[d] for d in destinations), dtype=np.int64, count=len(destinations)
    )
    dist = csg.dijkstra(csr, indices=origin_seqs, limit=limit, return_predecessors=False)
    return dist[:, dest_seqs]


def shortest_path_metrics_one_to_one(
    graph: nx.Graph,
    trip_ids: list | pd.Series | np.ndarray,
    origins: list | pd.Series | np.ndarray,
    destinations: list | pd.Series | np.ndarray,
    weight: str,
    length_attr: str = "length",
    edge_features: dict[str, str] | None = None,
    *,
    cutoff: float | None = None,
) -> pd.DataFrame:
    """Paired (origin, destination) shortest-path routing with edge-feature aggregation.

    `edge_features` maps an edge attribute name to an aggregation:
      - 'sum'             : element-wise sum along the path (e.g. count of intersections)
      - 'length_weighted' : average weighted by edge length (e.g. average gradient)

    Returns a DataFrame indexed by trip_id with columns:
      - `distance` (sum of `length_attr` along the path)
      - `cost`     (sum of `weight`     along the path)
      - one column per requested edge feature

    Trips with no path are omitted from the output (so output length <= input length).

    `cutoff` (optional): per-origin Dijkstra `limit=` in `weight` units. Set
    this to a value comfortably above the longest expected trip cost — trips
    that would route beyond it are silently dropped (treated as unreachable),
    same as actually-unreachable trips. Big speed-up on large graphs when
    trip costs are small relative to graph diameter (e.g. urban trips on a
    country-scale road graph). Default `None` = no cutoff.

    Raises DataError if 'distance' or 'cost' appear in `edge_features` (would
    collide with the built-in path-total columns).
    """
    if not (len(trip_ids) == len(origins) == len(destinations)):
        raise DataError("trip_ids, origins, and destinations must have equal lengths.")
    edge_features = edge_features or {}
    reserved = {"distance", "cost"} & set(edge_features)
    if reserved:
        raise DataError(f"edge_features may not include reserved column names: {sorted(reserved)}")

    import scipy.sparse.csgraph as csg

    csr, nx_to_seq, seq_to_nx = _graph_to_csr(graph, weight)
    is_multi = isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))
    limit = cutoff if cutoff is not None else np.inf

    # Group trips by origin so we only call dijkstra once per unique origin
    # — far cheaper than per-trip when many trips share an origin (typical
    # of calibration data: per-person trip diaries).
    by_origin: dict = {}
    for trip_id, o, d in zip(trip_ids, origins, destinations):
        by_origin.setdefault(o, []).append((trip_id, d))

    rows = {}
    for origin, trips in by_origin.items():
        origin_seq = nx_to_seq[origin]
        dist, pred = csg.dijkstra(csr, indices=[origin_seq], limit=limit, return_predecessors=True)
        for trip_id, dest in trips:
            target_seq = nx_to_seq[dest]
            if not np.isfinite(dist[0, target_seq]):
                continue
            npath = _walk_predecessors_to_path(pred[0], origin_seq, target_seq, seq_to_nx)
            if not npath:
                continue
            edge_data = [
                _pick_min_weight_edge(graph, u, v, weight, is_multi)
                for u, v in zip(npath[:-1], npath[1:])
            ]
            lengths = np.array([ed.get(length_attr, 0.0) for ed in edge_data])
            costs = np.array([ed[weight] for ed in edge_data])
            row = {"distance": float(lengths.sum()), "cost": float(costs.sum())}
            for feature, agg in edge_features.items():
                values = np.array([ed.get(feature, 0.0) for ed in edge_data])
                row[feature] = _aggregate(values, lengths, agg, feature)
            rows[trip_id] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _pick_min_weight_edge(graph: nx.Graph, u, v, weight: str, is_multi: bool) -> dict:
    """For a (multi)graph, return the edge data dict for the cheapest parallel edge."""
    data = graph.get_edge_data(u, v)
    if data is None:
        raise DataError(f"No edge between {u} and {v}.")
    if is_multi:
        return min(data.values(), key=lambda d: d[weight])
    return data


def _aggregate(values: np.ndarray, weights: np.ndarray, agg: str, feature: str) -> float:
    """Aggregate per-edge `values` along one realised path.

    Supported aggregations: `'sum'` (plain sum) and `'length_weighted'`
    (weighted average using `weights`, typically per-edge length). Raises
    `DataError` for unknown `agg`, naming `feature` so the error points at
    the offending route-feature specification.
    """
    if agg == "sum":
        return float(values.sum())
    if agg == "length_weighted":
        if weights.sum() == 0:
            return float("nan")
        return float(np.average(values, weights=weights))
    raise DataError(f"Unknown aggregation `{agg}` for feature `{feature}`.")


# ---------------------------------------------------------------------------
# Tiered OD routing
# ---------------------------------------------------------------------------


def tiered_path_costs(
    pairs: TieredODPairs,
    graph: nx.Graph,
    weight: str,
    *,
    mask: TieredODPairs | None = None,
    cutoff: float | None = None,
    dtype: np.dtype | type = np.float32,
) -> TieredODPairs:
    """Shortest-path cost (sum of edge `weight` along the path) for every OD pair
    in `pairs`, across all tiers.

    Single-process. For the experimental multi-process variant see
    `tiered_path_costs_mp`. This is the hot path for almost every aperta
    application — the closure-based inner loop is on purpose, not a refactor
    candidate (a module-level worker pattern adds per-origin dict lookups
    that measurably slow down single-process routing).

    Every tier is routed across the same `graph`. All node IDs referenced
    anywhere in `pairs` — cell nodes (cells_to_cells keys + values), zone
    nodes (cells_to_zones values, zones_to_zones keys + values) — must
    therefore be present in `graph`.

    Args:
        pairs: TieredODPairs of destination IDs (typically from `od_pairs.get_pairs`).
        graph: networkx routable graph containing every node referenced in
            `pairs`. Converted internally to a scipy CSR matrix.
        weight: edge attribute name used as the per-edge routing cost (e.g.
            `'duration_naive'`, `'duration_traffic_iterative'`).
        mask: optional boolean `TieredODPairs` (build via `od_pairs.make_mask`).
            Destinations where the mask is `False` are skipped and stored as
            `np.inf` in the output (same convention as unreachable). Output
            arrays keep the same length as the input pairs (position-wise
            alignment is preserved); use the mask itself to distinguish
            "masked-out" from "unreachable" if you care. Missing origins or
            missing tiers in the mask are treated as "no filter".
        cutoff: optional network-distance cutoff in weight units (e.g. seconds
            for time-weighted edges, metres for length-weighted). Passed through
            to `scipy.sparse.csgraph.dijkstra` as `limit=cutoff`, truncating
            each per-origin Dijkstra at the cutoff. Big speed-up when the
            cutoff is small relative to graph diameter (e.g. walk accessibility
            on a country-scale graph). Destinations beyond cutoff are stored
            as `np.inf` — same convention as unreachable, so downstream metrics
            (`count_in_bins` etc.) handle them naturally. Default `None` = no
            cutoff (`limit=np.inf`).
        dtype: dtype of returned cost arrays (default `np.float32` — halves
            memory + on-disk size vs `float64`, with seconds-resolution
            precision more than sufficient for travel costs). Pass
            `np.float64` if downstream arithmetic needs the extra range
            (e.g. logsum with very small scale parameter).

    Returns:
        `TieredODPairs` of cost arrays paired position-wise with `pairs`. Each
        unreachable, masked-out, or beyond-cutoff destination is stored as
        `np.inf`.
    """
    import scipy.sparse.csgraph as csg

    csr, nx_to_seq, _seq_to_nx = _graph_to_csr(graph, weight)
    zero_edge = csr.nnz == 0
    limit = cutoff if cutoff is not None else np.inf

    def _route_subset(orig, sub_dests):
        origin_seq = nx_to_seq[orig]
        dist_row = csg.dijkstra(csr, indices=[origin_seq], limit=limit, return_predecessors=False)[
            0
        ]
        seq_dests = np.fromiter(
            (nx_to_seq[d] for d in sub_dests), dtype=np.int64, count=len(sub_dests)
        )
        return dist_row[seq_dests]

    def _per_origin(orig, dests, dest_mask):
        n = len(dests)
        if n == 0:
            return np.empty(0, dtype=dtype)
        if zero_edge:
            return np.array([0.0 if d == orig else np.inf for d in dests], dtype=dtype)
        if dest_mask is None:
            return _route_subset(orig, dests).astype(dtype, copy=False)
        true_idx = np.where(dest_mask)[0]
        out = np.full(n, np.inf, dtype=dtype)
        if len(true_idx) > 0:
            out[true_idx] = _route_subset(orig, dests[true_idx])
        return out

    logging.info(
        f"tiered_path_costs: routing single-process "
        f"(scipy, cutoff={'none' if cutoff is None else cutoff})..."
    )

    def _process(tier_name: str, tier: dict | None, mask_tier: dict | None) -> dict | None:
        if tier is None:
            return None
        n = len(tier)
        # Per-tier counter and progress step: long-distance tiers (zone-to-zone)
        # typically take much longer per origin than cell-tier ones, so
        # tracking a single global counter would compress the early-tier
        # progress into one big jump and stretch the later tiers' updates.
        log_every = max(1, n // 10)
        out: dict = {}
        for i, (orig, dests) in enumerate(tier.items(), start=1):
            dest_mask = mask_tier.get(orig) if mask_tier is not None else None
            out[orig] = _per_origin(orig, dests, dest_mask)
            if i % log_every == 0 or i == n:
                logging.info(f"  {tier_name}: {i:,} of {n:,} origins routed")
        return out

    cells_mask = mask.cells_to_cells if mask is not None else None
    c2z_mask = mask.cells_to_zones if mask is not None else None
    zones_mask = mask.zones_to_zones if mask is not None else None
    return TieredODNodePairs(
        cells_to_cells=_process("cells_to_cells", pairs.cells_to_cells, cells_mask),
        cells_to_zones=_process("cells_to_zones", pairs.cells_to_zones, c2z_mask),
        zones_to_zones=_process("zones_to_zones", pairs.zones_to_zones, zones_mask),
    )


class PathAggregation(NamedTuple):
    """Named per-edge feature aggregation along realised shortest paths.

    `name` labels the corresponding output column in `tiered_path_aggregate`'s
    return dict. `attribute` extracts a per-edge value; `aggregator` combines
    those values into one scalar per OD pair.

    `attribute`:
        - `str`: name of an edge attribute on the graph; the per-edge value is
          `edge_data[attribute]`.
        - `Callable[(u, v, data) -> float]`: arbitrary per-edge function.

    `aggregator`:
        - `'sum'`: sum across path edges (returns 0 for an empty path).
        - `'mean'`: arithmetic mean (returns NaN for an empty path).
        - `'min'`, `'max'`: respective extremes (NaN for an empty path).
        - `Callable[(np.ndarray) -> float]`: arbitrary callable on the
          per-edge value array.
    """

    name: str
    attribute: str | Callable
    aggregator: str | Callable = "sum"


class NodeAggregation(NamedTuple):
    """Named per-node feature aggregation along realised shortest paths.

    Parallel to `PathAggregation` but for node attributes (e.g. counting
    traffic signals encountered, or finding the highest-elevation node
    along a route). The node sequence of a path is `[u₀, u₁, ..., uₙ]`;
    `include_endpoints` controls whether the route's origin (u₀) and
    destination (uₙ) nodes contribute.

    `name`, `aggregator`: as in `PathAggregation`.

    `attribute`:
        - `str`: name of a node attribute on the graph; the per-node value
          is `node_data[attribute]`.
        - `Callable[(node, data) -> float]`: arbitrary per-node function.

    `include_endpoints`:
        - `True` (default): all `n+1` nodes contribute, including origin
          and destination. Risk: endpoints shared across many routes get
          amplified weight in cross-route counts.
        - `False`: interior nodes only (`u₁ .. uₙ₋₁`). Self-pair `[u]` and
          single-edge path `[u, v]` both yield an empty array → aggregator
          empty-path semantics apply (`'sum'` → 0; `'mean'/'min'/'max'`
          → NaN).
    """

    name: str
    attribute: str | Callable
    aggregator: str | Callable = "sum"
    include_endpoints: bool = True


def _resolve_attribute(attr: str | Callable) -> Callable:
    """Normalise an edge `attribute` spec into `(u, v, data) -> value`."""
    if isinstance(attr, str):
        return lambda u, v, data: data[attr]
    if callable(attr):
        return attr
    raise ValueError(f"`attribute` must be a string or callable, got {type(attr).__name__}.")


def _resolve_node_attribute(attr: str | Callable) -> Callable:
    """Normalise a node `attribute` spec into `(node, data) -> value`."""
    if isinstance(attr, str):
        return lambda node, data: data[attr]
    if callable(attr):
        return attr
    raise ValueError(f"`attribute` must be a string or callable, got {type(attr).__name__}.")


def _resolve_aggregator(agg: str | Callable) -> Callable:
    """Normalise an `aggregator` spec into `(np.ndarray) -> float`.

    Empty-path semantics: `'sum'` returns 0.0 (the additive identity);
    `'mean'` / `'min'` / `'max'` return NaN.
    """
    if agg == "sum":
        return lambda arr: float(arr.sum()) if arr.size else 0.0
    if agg == "mean":
        return lambda arr: float(arr.mean()) if arr.size else np.nan
    if agg == "min":
        return lambda arr: float(arr.min()) if arr.size else np.nan
    if agg == "max":
        return lambda arr: float(arr.max()) if arr.size else np.nan
    if callable(agg):
        return agg
    raise ValueError(
        f"Unknown aggregator {agg!r}; expected 'sum', 'mean', 'min', 'max', or a callable."
    )


def aggregate_along_paths(
    paths: list[list],
    graph: nx.Graph,
    weight: str,
    *,
    edge_aggregations: Sequence[PathAggregation] = (),
    node_aggregations: Sequence[NodeAggregation] = (),
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Walk realised paths and aggregate per-edge / per-node features along each.

    Pure path walker — no routing involved. Use this directly
    when you already have a list of paths (Strava traces, prebuilt routes,
    calibration targets, etc.). `tiered_path_aggregate` is the wrapper that
    routes shortest paths on a `TieredODPairs` and scatters results back
    into per-tier `TieredODPairs` outputs.

    For each path:
      - `cost`     = sum of `weight` along the path's edges
      - each `PathAggregation` reduces per-edge attribute values
      - each `NodeAggregation` reduces per-node attribute values

    `paths` semantics:
      - `[]`           → unreachable: cost=`inf`, all aggs=`NaN`
      - `[u]`          → self-pair: cost=0, edge aggs follow empty-array
                         semantics, node aggs follow each spec's
                         `include_endpoints` setting
      - `[u, v, ...]`  → multi-node path; cost + aggs walked normally

    Args:
        paths: list of node-id sequences (lists). Node IDs must match
            `graph` keys.
        graph: networkx graph used for edge / node attribute lookup. For
            MultiGraph / MultiDiGraph the min-`weight` parallel edge is
            used (matches the router's choice).
        weight: edge attribute name used as the per-edge cost.
        edge_aggregations: list of `PathAggregation` specs (per-edge).
        node_aggregations: list of `NodeAggregation` specs (per-node).
            At least one of `edge_aggregations` / `node_aggregations` must
            be non-empty. Names must be unique across both lists.
        dtype: dtype of returned arrays (default `np.float32`).

    Returns:
        `(costs, aggregations_by_name)`:
            - `costs`: ndarray of shape `(len(paths),)`. `inf` for
              unreachable; `0.0` for self-pairs.
            - `aggregations_by_name`: dict `{name -> ndarray}` with one
              entry per spec across both lists. Unreachable destinations
              are `NaN`.
    """
    if not edge_aggregations and not node_aggregations:
        raise ValueError(
            "At least one of `edge_aggregations` / `node_aggregations` must be "
            "non-empty. For cost-only routing, use `tiered_path_costs` instead."
        )
    edge_aggregations = list(edge_aggregations)
    node_aggregations = list(node_aggregations)
    names = [a.name for a in edge_aggregations] + [a.name for a in node_aggregations]
    if len(set(names)) != len(names):
        raise ValueError(f"Aggregation names must be unique across edge + node specs; got {names}.")

    edge_attr_fns = [_resolve_attribute(a.attribute) for a in edge_aggregations]
    edge_agg_fns = [_resolve_aggregator(a.aggregator) for a in edge_aggregations]
    node_attr_fns = [_resolve_node_attribute(a.attribute) for a in node_aggregations]
    node_agg_fns = [_resolve_aggregator(a.aggregator) for a in node_aggregations]
    node_include_endpoints = [a.include_endpoints for a in node_aggregations]
    is_multi = isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))

    def _get_edge(u, v):
        """Min-`weight` edge between `u` and `v` (collapses MultiGraph parallels)."""
        if is_multi:
            return min(graph[u][v].values(), key=lambda d: d.get(weight, np.inf))
        return graph[u][v]

    n = len(paths)
    n_edge = len(edge_aggregations)
    n_node = len(node_aggregations)
    costs = np.full(n, np.inf, dtype=dtype)
    edge_out = [np.full(n, np.nan, dtype=dtype) for _ in range(n_edge)]
    node_out = [np.full(n, np.nan, dtype=dtype) for _ in range(n_node)]

    for i, path in enumerate(paths):
        if not path:
            continue  # unreachable: cost=inf, aggs=NaN (both preallocated)

        n_edges = len(path) - 1
        edge_vals = np.empty((n_edge, n_edges), dtype=dtype)
        cost_sum = 0.0
        valid = True
        for k in range(n_edges):
            u, v = path[k], path[k + 1]
            try:
                edge = _get_edge(u, v)
            except KeyError:
                valid = False
                break
            cost_sum += float(edge.get(weight, np.inf))
            for j, attr_fn in enumerate(edge_attr_fns):
                edge_vals[j, k] = float(attr_fn(u, v, edge))
        if not valid:
            continue

        costs[i] = cost_sum
        for j, agg_fn in enumerate(edge_agg_fns):
            edge_out[j][i] = float(agg_fn(edge_vals[j]))
        for j, attr_fn in enumerate(node_attr_fns):
            nodes = path if node_include_endpoints[j] else path[1:-1]
            if nodes:
                node_vals = np.fromiter(
                    (float(attr_fn(node, graph.nodes[node])) for node in nodes),
                    dtype=dtype,
                    count=len(nodes),
                )
            else:
                node_vals = np.empty(0, dtype=dtype)
            node_out[j][i] = float(node_agg_fns[j](node_vals))

    aggs: dict[str, np.ndarray] = {}
    for edge_spec, arr in zip(edge_aggregations, edge_out):
        aggs[edge_spec.name] = arr
    for node_spec, arr in zip(node_aggregations, node_out):
        aggs[node_spec.name] = arr
    return costs, aggs


def tiered_path_aggregate(
    pairs: TieredODPairs,
    graph: nx.Graph,
    weight: str,
    *,
    edge_aggregations: Sequence[PathAggregation] = (),
    node_aggregations: Sequence[NodeAggregation] = (),
    mask: TieredODPairs | None = None,
    cutoff: float | None = None,
    dtype: np.dtype | type = np.float32,
) -> tuple[TieredODNodePairs, dict[str, TieredODNodePairs]]:
    """Route shortest paths and aggregate per-edge / per-node features along each.

    Wraps `aggregate_along_paths` with routing on every tier of `pairs`.
    Memory cost matches `tiered_path_costs` for the cost component —
    paths are processed per-origin and discarded.

    For the cost-only case (no aggregations needed), use `tiered_path_costs`
    directly: it can skip path retrieval (more expensive than distance
    retrieval) and is faster.

    Args:
        pairs, graph, weight, mask, cutoff, dtype: as in `tiered_path_costs`.
            Paths are retrieved via `scipy.sparse.csgraph.dijkstra(
            return_predecessors=True)` and reconstructed by walking the
            predecessor chain back from each target to the origin.
        edge_aggregations: list of `PathAggregation` specs (per-edge).
        node_aggregations: list of `NodeAggregation` specs (per-node).
            At least one of the two must be non-empty. Names must be unique
            across both lists.

    Returns:
        `(costs, aggregations_by_name)`:
            - `costs`: `TieredODPairs` of routing costs (sum of `weight`
              along the realised path). Same shape and conventions as
              `tiered_path_costs`. Unreachable / masked-out / beyond-cutoff
              destinations are `np.inf`.
            - `aggregations_by_name`: `dict[name -> TieredODPairs]`. One
              entry per spec (edge + node), keyed by spec name. Unreachable
              / masked-out / beyond-cutoff destinations are `np.nan` (not
              `inf`, since aggregations may be signed or already use `inf`
              semantics).

    For OSMnx-style MultiDiGraphs with multiple parallel edges between the
    same `(u, v)` pair, the edge with the lowest `weight` is used for both
    cost computation and attribute extraction (matching the router's choice).

    For self-pairs (origin == destination, path length 0): cost is 0.0,
    edge aggregations follow each aggregator's empty-array semantics
    (`'sum'` → 0.0; `'mean'`/`'min'`/`'max'` → NaN), node aggregations
    depend on each spec's `include_endpoints` setting.
    """
    if not edge_aggregations and not node_aggregations:
        raise ValueError(
            "At least one of `edge_aggregations` / `node_aggregations` must be "
            "non-empty. For cost-only routing, use `tiered_path_costs` instead."
        )
    edge_aggregations = list(edge_aggregations)
    node_aggregations = list(node_aggregations)
    names = [a.name for a in edge_aggregations] + [a.name for a in node_aggregations]
    if len(set(names)) != len(names):
        raise ValueError(f"Aggregation names must be unique across edge + node specs; got {names}.")

    # Per-origin path retrieval via scipy dijkstra. Path reconstruction
    # walks the predecessor chain from each target back to the origin.
    import scipy.sparse.csgraph as csg

    csr, nx_to_seq, seq_to_nx = _graph_to_csr(graph, weight)
    zero_edge = csr.nnz == 0
    limit = cutoff if cutoff is not None else np.inf

    def _paths(orig, sub_dests):
        if zero_edge:
            return [[orig] if d == orig else [] for d in sub_dests]
        origin_seq = nx_to_seq[orig]
        dist, pred = csg.dijkstra(csr, indices=[origin_seq], limit=limit, return_predecessors=True)
        paths = []
        for d in sub_dests:
            target_seq = nx_to_seq[d]
            if not np.isfinite(dist[0, target_seq]):
                paths.append([])  # unreachable or beyond cutoff
            else:
                paths.append(_walk_predecessors_to_path(pred[0], origin_seq, target_seq, seq_to_nx))
        return paths

    def _per_origin(orig, dests, dest_mask):
        n = len(dests)
        cost_arr = np.full(n, np.inf, dtype=dtype)
        agg_arrs = {name: np.full(n, np.nan, dtype=dtype) for name in names}
        if n == 0:
            return cost_arr, agg_arrs
        if dest_mask is None:
            active_idx = np.arange(n)
            active_dests = dests
        else:
            active_idx = np.where(dest_mask)[0]
            if len(active_idx) == 0:
                return cost_arr, agg_arrs
            active_dests = dests[active_idx]

        paths = _paths(orig, active_dests)
        sub_costs, sub_aggs = aggregate_along_paths(
            paths,
            graph,
            weight,
            edge_aggregations=edge_aggregations,
            node_aggregations=node_aggregations,
            dtype=dtype,
        )
        cost_arr[active_idx] = sub_costs
        for name in names:
            agg_arrs[name][active_idx] = sub_aggs[name]
        return cost_arr, agg_arrs

    logging.info(
        f"tiered_path_aggregate: routing (scipy, cutoff={'none' if cutoff is None else cutoff})..."
    )

    def _process(
        tier_name: str, tier: dict | None, mask_tier: dict | None
    ) -> tuple[dict, dict[str, dict]] | None:
        if tier is None:
            return None
        n = len(tier)
        log_every = max(1, n // 10)
        cost_out: dict = {}
        agg_outs: dict[str, dict] = {name: {} for name in names}
        for i, (orig, dests) in enumerate(tier.items(), start=1):
            dest_mask = mask_tier.get(orig) if mask_tier is not None else None
            cost_arr, agg_arrs = _per_origin(orig, dests, dest_mask)
            cost_out[orig] = cost_arr
            for name in names:
                agg_outs[name][orig] = agg_arrs[name]
            if i % log_every == 0 or i == n:
                logging.info(f"  {tier_name}: {i:,} of {n:,} origins routed")
        return cost_out, agg_outs

    cells_mask = mask.cells_to_cells if mask is not None else None
    c2z_mask = mask.cells_to_zones if mask is not None else None
    zones_mask = mask.zones_to_zones if mask is not None else None

    cells_res = _process("cells_to_cells", pairs.cells_to_cells, cells_mask)
    c2z_res = _process("cells_to_zones", pairs.cells_to_zones, c2z_mask)
    zones_res = _process("zones_to_zones", pairs.zones_to_zones, zones_mask)

    costs = TieredODNodePairs(
        cells_to_cells=cells_res[0] if cells_res is not None else {},
        cells_to_zones=c2z_res[0] if c2z_res is not None else None,
        zones_to_zones=zones_res[0] if zones_res is not None else None,
    )
    aggregations_by_name = {
        name: TieredODNodePairs(
            cells_to_cells=cells_res[1][name] if cells_res is not None else {},
            cells_to_zones=c2z_res[1][name] if c2z_res is not None else None,
            zones_to_zones=zones_res[1][name] if zones_res is not None else None,
        )
        for name in names
    }
    return costs, aggregations_by_name


def add_trip_overhead(
    pairs: TieredODPairs,
    costs: TieredODPairs,
    cell_info: pd.DataFrame,
    *,
    zone_info: pd.DataFrame | None = None,
    origin_overhead: Callable | None = None,
    dest_overhead: Callable | None = None,
    verify_finite: bool = True,
) -> TieredODPairs:
    """Add per-trip origin and/or destination overhead to each OD pair's cost.

    For each (origin_node, dest_node) pair in `costs` (paired position-wise with
    `pairs`), the returned cost is::

        new_cost = old_cost + origin_overhead(info_o.loc[orig])
                            + dest_overhead (info_d.loc[dest])

    where the info dataframe depends on the tier and the endpoint side::

        cells_to_cells:    info_o = cell_info,   info_d = cell_info
        cells_to_zones:    info_o = cell_info,   info_d = zone_info
        zones_to_zones:    info_o = zone_info,   info_d = zone_info

    Each info DataFrame is one row per network node at that tier, indexed by
    node ID. It can mix native node-level attributes (e.g. local density,
    intersection count) with aggregated unit-level attributes (e.g. distance
    from cell centroid to nearest network node) — the function doesn't care
    which is which, just looks up by node ID and hands the row(s) to the
    callback.

    When multiple units share a node (typical for cells), aggregate upstream::

        cell_info = (cells.groupby('node_id_nw').agg({
                        'dist_to_node': 'mean',
                        'population': 'sum',
                    }).join(nodes))   # node-level attrs joined in

    Each callback receives a single `info` argument:
      - For the **origin** side: a 1-D `pd.Series` (the row for that single
        origin). The callback returns a scalar.
      - For the **destination** side: a `pd.DataFrame` (one row per dest,
        ordered the same as the dest array). The callback returns a 1-D
        Series / ndarray of the same length. Pandas column access
        (`info['col']`) yields a scalar in the Series case and a Series in the
        DataFrame case, so the same callable typically works for both modes
        without branching.

    `origin_overhead` and `dest_overhead` are independently optional — pass
    `None` to skip that side.

    Args:
        pairs: TieredODPairs of destination IDs (typically from `od_pairs.get_pairs`).
        costs: TieredODPairs of cost arrays to augment; same shape as `pairs`.
        cell_info: per-cell-node info DataFrame, indexed by the cell-tier node ID.
        zone_info: per-zone-node info DataFrame, indexed by the zone-tier node ID.
            Required iff `cells_to_zones` or `zones_to_zones` is present in `costs`.
        origin_overhead: callable, see above. None to skip the origin contribution.
        dest_overhead: callable, see above. None to skip the dest contribution.
        verify_finite: if True, a ValueError is raised when output is not finite (NaN or Inf).

    Returns:
        New `TieredODPairs` of cost arrays (same shape as `costs`) with the
        overhead added. `costs` is not mutated. Unreachable / masked-out entries
        (np.inf in the input) stay infinite (inf + anything = inf).

    Raises:
        ValueError if overhead result is not finite and verify_finite is True.
    """
    if origin_overhead is None and dest_overhead is None:
        return costs  # nothing to do

    # (tier_attr) -> (origin_info_df, dest_info_df) lookup.
    tier_infos: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]] = {
        "cells_to_cells": (cell_info, cell_info),
        "cells_to_zones": (cell_info, zone_info),
        "zones_to_zones": (zone_info, zone_info),
    }

    def _process(tier_attr: str) -> dict | None:
        cost_tier = getattr(costs, tier_attr)
        if cost_tier is None:
            return None
        pair_tier = getattr(pairs, tier_attr)
        if pair_tier is None:
            raise DataError(
                f"`pairs.{tier_attr}` is None but `costs.{tier_attr}` is set — "
                f"can't look up destination IDs to apply overhead."
            )
        info_o, info_d = tier_infos[tier_attr]
        if origin_overhead is not None and info_o is None:
            raise ValueError(
                f"tier `{tier_attr}` has origin overhead requested but the "
                f"matching info DataFrame is None."
            )
        if dest_overhead is not None and info_d is None:
            raise ValueError(
                f"tier `{tier_attr}` has dest overhead requested but the "
                f"matching info DataFrame is None."
            )

        out: dict = {}
        for orig, cost_arr in cost_tier.items():
            new_c = np.asarray(cost_arr).copy()
            if origin_overhead is not None:
                # Validated non-None above when origin_overhead is set.
                assert info_o is not None
                # Single origin -> Series. Callback returns scalar.
                oh_o = float(origin_overhead(info_o.loc[orig]))
                if verify_finite and not np.isfinite(oh_o):
                    raise ValueError(f"Origin overhead {orig} is not finite.")
                new_c = new_c + oh_o
            if dest_overhead is not None:
                assert info_d is not None
                dests = pair_tier[orig]
                if len(dests) > 0:
                    # Many dests -> DataFrame. Callback returns Series/array.
                    oh_d = np.asarray(dest_overhead(info_d.loc[dests]), dtype=np.float64)
                    if verify_finite:
                        not_finite = (~np.isfinite(oh_d)).sum()
                        if not_finite > 0:
                            raise ValueError(
                                f"Destination overhead for origin {orig} contains "
                                f"{not_finite:,} non-finite numbers."
                            )
                    new_c = new_c + oh_d
            out[orig] = new_c
        return out

    return type(costs)(
        cells_to_cells=_process("cells_to_cells"),
        cells_to_zones=_process("cells_to_zones"),
        zones_to_zones=_process("zones_to_zones"),
    )


def set_min_intrazonal_cost(
    costs: TieredODPairs,
    min_cost: float | dict | pd.Series,
) -> TieredODPairs:
    """Floor cell-tier costs at `min_cost` — applied uniformly to every entry.

    Routing on a graph returns 0 for the trivial origin-to-origin path. That's
    fine for cumulative-opportunity output (cost 0 falls in the smallest bin),
    but degenerate for decay-based metrics like gravity: `exp(-β·0) = 1` puts
    the maximum possible decay weight on the cell itself, and `c^(-β)` at c = 0
    diverges outright.

    The floor is applied uniformly to every cell-tier entry, not just self-
    pairs. Setting only the self-pair to a non-zero floor would create an
    inconsistency: a cell would route to itself at, say, 120 s while a
    different (very close) cell could route at 60 s, implying you can travel
    further faster than you can travel zero distance. The min-cost
    interpretation is the physical floor on per-trip cost — no trip can take
    less than `min_cost`, regardless of distance — and it handles the
    intrazonal-cost-0 case as a side effect.

    Non-finite entries (`np.inf` for unreachable destinations, `np.nan` for
    missing observations) are passed through unchanged — the floor is applied
    only to finite costs. Flooring `inf` would erase reachability information
    (an unreachable destination would become reachable in `min_cost` seconds),
    and flooring `nan` would silently invent data; both behaviours would be
    incorrect.

    Only `cells_to_cells` is modified — `cells_to_zones` and `zones_to_zones`
    are routed between distinct cell-zone / zone-zone pairs and don't have the
    same zero-self-cost degeneracy. Tiers that are `None` pass through.

    Args:
        costs: TieredODPairs of cost arrays.
        min_cost: floor value. Either a scalar `float` (same floor for every
            origin), a `dict[origin_node -> float]` (per-origin floor;
            origins absent from the dict get no floor and their costs pass
            through unchanged), or a `pd.Series` indexed by origin_node
            (same semantics as the dict form).

    Returns:
        New `TieredODPairs` with `cell_tier_cost = max(cell_tier_cost,
        min_cost)` applied per origin to finite entries; non-finite entries
        (`inf`, `nan`) pass through unchanged.
    """
    if isinstance(min_cost, pd.Series):
        cost_lookup: dict = min_cost.to_dict()
        scalar_floor: float | None = None
    elif isinstance(min_cost, dict):
        cost_lookup = dict(min_cost)
        scalar_floor = None
    else:
        cost_lookup = {}
        scalar_floor = float(min_cost)

    new_cells_to_cells: dict = {}
    if costs.cells_to_cells is None:
        raise ValueError(
            "`costs.cells_to_cells` is None; cell-tier is required for this transform."
        )
    for origin, cost_arr in costs.cells_to_cells.items():
        # Preserve input dtype (typically FP32).
        new_arr = np.asarray(cost_arr).copy()
        floor = scalar_floor if scalar_floor is not None else cost_lookup.get(origin)
        if floor is None:
            new_cells_to_cells[origin] = new_arr
            continue
        # Apply max() only to finite entries; inf/nan are left as-is.
        finite_mask = np.isfinite(new_arr)
        new_arr[finite_mask] = np.maximum(new_arr[finite_mask], float(floor))
        new_cells_to_cells[origin] = new_arr

    return type(costs)(
        cells_to_cells=new_cells_to_cells,
        cells_to_zones=costs.cells_to_zones,
        zones_to_zones=costs.zones_to_zones,
    )
