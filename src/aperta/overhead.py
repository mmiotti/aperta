"""
Trip overheads — the first-mile and last-mile costs that aren't on the routed
path.

Aperta routes between network nodes, but real trips start and end at units
(cells, buildings, etc.) that are typically NOT at network nodes, and may
carry additional fixed costs at the origin or destination side (parking-find
time, station-access time, etc.). The "overhead" is the extra cost between
the actual unit and its assigned network node — at the origin (first mile)
and at the destination (last mile).

**Two supported sides:** origin (first-mile, added to every OD leaving that
unit) and destination (last-mile, added to every OD arriving at that unit).
Applied to a **geo-keyed** cost `TieredODGeoPairs` via `add_geo_overheads`,
which auto-derives the zone-tier overhead from the cell-tier one when only
the cell side is passed (see the function's docstring for the footgun this
protects against).

**Destination overhead for the zone tier** is a modelling choice. Three
canonical aggregators are provided:

- `aggregate_dest_overhead_per_node`   — mean per-cell overhead per snap
  node (cell tier).
- `aggregate_dest_overhead_per_group_euclidean` — for road-network
  analyses, where "cost to a typical place in the zone" is best modeled as
  Euclidean centroid-distance ÷ speed.
- `aggregate_dest_overhead_per_group_routed`    — for transit-style
  analyses, where users have to reach a specific stop node so the last-mile
  is a routed Dijkstra distance rather than Euclidean.

**Workflow (canonical geo-keyed pattern)**::

    # 1. Per-cell first-mile (origin side) — typically done in data prep
    cells['walk_overhead_s'] = dist_to_node / WALK_SPEED_MS

    # 2. Compute aggregated destination overheads
    node_overhead = overhead.aggregate_dest_overhead_per_node(
        cells, 'walk_overhead_s')
    zones['walk_dest_overhead_s'] = overhead.aggregate_dest_overhead_per_group_euclidean(
        cells, zones, speed=WALK_SPEED_MS,
        group_id_column='zone_id', cell_overhead_column='walk_overhead_s')

    # 3. Bake overheads into the geo-keyed cost ODM
    times_geo = overhead.add_geo_overheads(
        times_geo, pairs_geo,
        origin_cell=cells['walk_overhead_s'],
        dest_cell=cells['walk_overhead_s'],
        dest_zone=zones['walk_dest_overhead_s'],
        cell_to_zone=cells['zone_id'],
    )

    # 4. Accessibility reads the pre-baked ODM directly.
    accessibility.gravity(times_geo, weights, cell_to_zone, decays)
"""

from typing import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODGeoPairs


def aggregate_dest_overhead_per_node(
    cells: pd.DataFrame,
    cell_overhead_column: str,
    *,
    node_column: str = "node_id",
    weight_column: str | None = None,
) -> pd.Series:
    """Per-network-node destination overhead — (weighted) mean of per-cell
    overheads across cells sharing each node.

    For cell-tier destinations: a destination network node typically represents
    one or more cells (any cell whose `node_column` value is that node). The
    "destination overhead" — the cost of getting from the node back to a
    representative cell — is approximated as the mean of those cells' own
    first-mile overheads.

    Args:
        cells: per-cell DataFrame.
        cell_overhead_column: column on `cells` with the per-cell overhead
            value (typically the first-mile cost — cell centroid → assigned
            network node — divided by speed if `cells` carries distance).
        node_column: column on `cells` mapping each cell to its network node.
        weight_column: optional column to weight the mean (e.g. `'population'`
            or another size-of-cell column). `None` (default) = uniform.

    Returns:
        `pd.Series` indexed by network node ID, with one mean overhead per
        node. Nodes with no associated cells are absent from the result.
    """
    df = cells.dropna(subset=[node_column])
    if weight_column is None:
        return (
            df.groupby(node_column)[cell_overhead_column]
            .mean()
            .rename(f"dest_overhead_per_node({cell_overhead_column})")
        )

    # Weighted mean: Σ(v * w) / Σ(w) per group; skip NaN/zero-weight rows.
    # Explicit per-group loop to avoid the type-stub friction of
    # `groupby().apply()` with `include_groups`.
    out: dict = {}
    for node, idx in df.groupby(node_column).groups.items():
        sub = df.loc[idx]
        v = sub[cell_overhead_column].to_numpy(dtype=float)
        w = sub[weight_column].to_numpy(dtype=float)
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        out[node] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")
    return pd.Series(out).rename(f"dest_overhead_per_node({cell_overhead_column})")


def aggregate_dest_overhead_per_group(
    cells: pd.DataFrame | gpd.GeoDataFrame,
    target_groups: pd.DataFrame | gpd.GeoDataFrame,
    *,
    distance: str,
    group_id_column: str,
    cell_overhead_column: str | None = None,
    weight_column: str | None = None,
    # Euclidean-mode kwargs:
    speed: float | None = None,
    # Routed-mode kwargs:
    graph: nx.Graph | None = None,
    weight: str | None = None,
    node_column: str = "node_id",
    cutoff: float | None = None,
) -> pd.Series:
    """Per-group destination overhead — for zone-tier destinations.

    Pair with `add_geo_overheads(dest_zone=...)` (overhead #4 at zone tier).
    The same function shape handles any grouping via the `group_id_column`
    kwarg.

    For each target group `g` (with representative point / network node):

        overhead(g) = (weighted) mean over cells c in g of:
            (cells[c, cell_overhead_column] if cell_overhead_column else 0)
            + last_mile_distance(c, g)

    The `last_mile_distance` term depends on the `distance` mode:

    - `'euclidean'`: `euclidean(c_centroid, g_centroid) / speed`. Appropriate
      for road-network analyses where users don't actually have to pass
      through the group's representative node — the "geometric distance to
      a typical place in the group" is the more honest approximation.
      Requires `speed` (CRS-units per time-unit) and CRSes on `cells` +
      `target_groups` that agree and are metric.
    - `'routed'`: `route(g_node → c_node, weight)`. Appropriate for
      transit-style analyses where users have to access a specific stop
      node. Direction is `g_node → c_node` (single-source Dijkstra from
      `g_node`) — by symmetry on undirected graphs this equals
      `c_node → g_node`; for directed graphs (one-way streets etc.),
      the `g_node → c_node` direction is the "egress at destination"
      semantic. Requires `graph`, `weight`, and `node_column` on both
      `cells` and `target_groups`.

    The `cell_overhead_column` typically encodes the mode-specific constant
    plus any feature-based overhead (e.g. β · population_density) the user
    has precomputed per cell. The last-mile distance is added on top.

    Args:
        cells: per-cell DataFrame (routed mode) or GeoDataFrame with polygon
            / point geometry (Euclidean mode). Must have `group_id_column`
            linking to `target_groups.index`. In routed mode must also have
            `node_column` (network node ID).
        target_groups: per-group DataFrame / GeoDataFrame, indexed by group
            ID. In Euclidean mode the polygon centroid is the "representative
            point"; in routed mode `node_column` gives the group's
            representative network node.
        distance: `'euclidean'` or `'routed'`. Picks which last-mile distance
            model applies and which mode-specific kwargs are consulted.
        group_id_column: column on `cells` linking to `target_groups.index`
            (typically `'zone_id'` or `'region_id'`).
        cell_overhead_column: optional column on `cells` with per-cell base
            overhead (constant + feature-based), added on top of the
            last-mile distance.
        weight_column: optional column on `cells` to weight the mean (e.g.
            `'population'`). `None` = uniform.
        speed: `'euclidean'` mode only — speed in CRS-units per time-unit,
            used to convert distance to time. Must be > 0.
        graph: `'routed'` mode only — routable networkx (or osmnx) graph.
        weight: `'routed'` mode only — edge attribute name used for routing
            (e.g. `'walk_time_s'`).
        node_column: `'routed'` mode only — column name carrying the network
            node ID, on both `cells` and `target_groups`. Default `'node_id'`.
        cutoff: `'routed'` mode only — optional `csg.dijkstra(limit=cutoff)`
            in `weight` units. Cells beyond it from `g_node` are treated as
            unreachable (contribute NaN, filtered from the mean). Set this
            comfortably above the longest expected last-mile in `weight`
            units (typical zone diameter ÷ slowest mode speed) to speed up
            routing on large graphs.

    Returns:
        `pd.Series` indexed by `target_groups.index`, with one mean overhead
        per group. Groups with no constituent cells (or, in routed mode,
        with all cells unreachable from `g_node`) get `NaN`.
    """
    if distance == "euclidean":
        if speed is None:
            raise ValueError("`speed` is required for `distance='euclidean'`.")
        distances_per_cell, cell_group_ids, cells_valid = _euclidean_last_mile(
            cells,
            target_groups,
            speed=speed,
            group_id_column=group_id_column,
        )
    elif distance == "routed":
        if graph is None or weight is None:
            raise ValueError("`graph` and `weight` are required for `distance='routed'`.")
        distances_per_cell, cell_group_ids, cells_valid = _routed_last_mile(
            cells,
            target_groups,
            graph=graph,
            weight=weight,
            group_id_column=group_id_column,
            node_column=node_column,
            cutoff=cutoff,
        )
    else:
        raise ValueError(f"`distance` must be `'euclidean'` or `'routed'`; got {distance!r}.")

    if cell_overhead_column is not None:
        distances_per_cell = distances_per_cell + (
            cells_valid[cell_overhead_column].to_numpy(dtype=float)
        )
    if weight_column is not None:
        weights = cells_valid[weight_column].to_numpy(dtype=float)
    else:
        weights = np.ones(len(cells_valid))

    out: dict = {}
    for group, idx in pd.Series(cell_group_ids).groupby(cell_group_ids).groups.items():
        i = np.asarray(idx)
        v = distances_per_cell[i]
        w = weights[i]
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        out[group] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")

    return (
        pd.Series(out)
        .reindex(target_groups.index)
        .rename(f"dest_overhead_per_group({group_id_column})")
    )


def _euclidean_last_mile(
    cells: gpd.GeoDataFrame,
    target_groups: gpd.GeoDataFrame,
    *,
    speed: float,
    group_id_column: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-cell Euclidean centroid-to-group-centroid distance, divided by
    `speed`. Returns `(distances, group_ids, cells_valid)` — `cells_valid`
    is the filtered per-cell frame (rows with a resolvable group_id).
    """
    if speed <= 0:
        raise ValueError(f"`speed` must be > 0; got {speed!r}.")

    cells_valid = cells.dropna(subset=[group_id_column])
    if not len(cells_valid):
        return np.array([]), np.array([]), cells_valid

    cells_centroids = cells_valid.geometry.centroid
    cell_x = cells_centroids.x.to_numpy(dtype=float)
    cell_y = cells_centroids.y.to_numpy(dtype=float)

    group_centroids = target_groups.geometry.centroid
    group_x_lookup = group_centroids.x.to_dict()
    group_y_lookup = group_centroids.y.to_dict()
    cell_groups = cells_valid[group_id_column].to_numpy()
    group_x_per_cell = np.array([group_x_lookup.get(g, np.nan) for g in cell_groups])
    group_y_per_cell = np.array([group_y_lookup.get(g, np.nan) for g in cell_groups])

    distances = np.hypot(cell_x - group_x_per_cell, cell_y - group_y_per_cell) / speed
    return distances, cell_groups, cells_valid


def _routed_last_mile(
    cells: pd.DataFrame,
    target_groups: pd.DataFrame,
    *,
    graph: nx.Graph,
    weight: str,
    group_id_column: str,
    node_column: str,
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-cell routed distance from each cell's group representative node
    back to the cell's snap node, via one Dijkstra per group. Returns
    `(distances, group_ids, cells_valid)`.
    """
    import scipy.sparse.csgraph as csg

    from aperta.routing import _graph_to_csr

    csr, nx_to_seq, _ = _graph_to_csr(graph, weight)
    limit = cutoff if cutoff is not None else np.inf

    cells_valid = cells.dropna(subset=[node_column, group_id_column])
    if not len(cells_valid):
        return np.array([]), np.array([]), cells_valid

    group_node_lookup = target_groups[node_column].to_dict()
    distances = np.full(len(cells_valid), np.nan, dtype=float)
    cell_nodes_arr = cells_valid[node_column].to_numpy()
    cell_groups_arr = cells_valid[group_id_column].to_numpy()

    for group_id, idx in pd.Series(cell_groups_arr).groupby(cell_groups_arr).groups.items():
        g_node = group_node_lookup.get(group_id)
        if g_node is None or pd.isna(g_node):
            continue
        i = np.asarray(idx)
        g_seq = nx_to_seq[g_node]
        dist_row = csg.dijkstra(csr, indices=[g_seq], limit=limit, return_predecessors=False)[0]
        cell_seqs = np.fromiter(
            (nx_to_seq[n] for n in cell_nodes_arr[i]), dtype=np.int64, count=len(i)
        )
        distances[i] = dist_row[cell_seqs]

    return distances, cell_groups_arr, cells_valid


def _zones_referenced_as_origins(costs: TieredODGeoPairs) -> set:
    """Set of zone IDs that appear as origins in any zone-keyed origin tier."""
    if costs.zones_to_zones is None:
        return set()
    return set(costs.zones_to_zones.keys())


def _zones_referenced_as_dests(pairs: TieredODGeoPairs) -> set:
    """Set of zone IDs that appear as destinations in any zone-keyed dest tier."""
    zones: set = set()
    for tier in (pairs.cells_to_zones, pairs.zones_to_zones):
        if tier is None:
            continue
        for dest_ids in tier.values():
            zones.update(dest_ids)
    return zones


def _aggregate_cell_to_zone(
    cell_overhead: pd.Series | dict,
    cell_to_zone: pd.Series | dict | None,
    aggregator: str | Callable,
    *,
    referenced_zones: set,
    side: str,
) -> pd.Series:
    """Aggregate per-cell overhead into per-zone overhead.

    Raises if `cell_to_zone` is missing, if any cell in `cell_overhead` is
    missing from `cell_to_zone`, or if any zone in `referenced_zones` is
    missing from the aggregation result (no constituent cells).
    """
    if cell_to_zone is None:
        raise ValueError(
            f"`{side}_cell` overhead provided and zone-tier overhead is needed "
            f"(the costs include zones_to_zones or cells_to_zones), but neither "
            f"`{side}_zone` nor `cell_to_zone` was given. Pass `cell_to_zone` "
            f"(e.g., `cells['zone_id']`) to auto-derive the zone-tier overhead, "
            f"or supply `{side}_zone` explicitly."
        )
    cell_overhead_s: pd.Series = (
        pd.Series(cell_overhead) if isinstance(cell_overhead, dict) else cell_overhead
    )
    cell_to_zone_s: pd.Series = (
        pd.Series(cell_to_zone) if isinstance(cell_to_zone, dict) else cell_to_zone
    )

    # Align cell_to_zone to the cells in cell_overhead.
    aligned = cell_to_zone_s.reindex(cell_overhead_s.index)
    missing_cells = aligned.isna()
    if missing_cells.any():
        n = int(missing_cells.sum())
        sample = list(cell_overhead_s.index[missing_cells][:5])
        raise ValueError(
            f"{n} cells in `{side}_cell` overhead have no zone assignment in "
            f"`cell_to_zone` (first {len(sample)}: {sample})."
        )

    grouped = cell_overhead_s.groupby(aligned).agg(aggregator)

    missing_zones = referenced_zones - set(grouped.index)
    if missing_zones:
        sample = list(missing_zones)[:5]
        raise ValueError(
            f"{len(missing_zones)} zone(s) used in the costs have no "
            f"constituent cells in `cell_to_zone` (first {len(sample)}: "
            f"{sample}). Cannot derive per-zone {side} overhead. Pass "
            f"`{side}_zone` explicitly for these zones."
        )
    return grouped


def _as_lookup(x: pd.Series | dict | None) -> dict | None:
    """Normalise a Series-or-dict-or-None to a dict-or-None."""
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.to_dict()
    return dict(x)


def add_geo_overheads(
    costs: TieredODGeoPairs,
    pairs: TieredODGeoPairs,
    *,
    origin_cell: pd.Series | dict | None = None,
    origin_zone: pd.Series | dict | None = None,
    dest_cell: pd.Series | dict | None = None,
    dest_zone: pd.Series | dict | None = None,
    cell_to_zone: pd.Series | dict | None = None,
    zone_aggregator: str | Callable = "mean",
) -> TieredODGeoPairs:
    """Add per-geo-unit origin and destination overheads to a geo-keyed cost ODM.

    Four independent overhead lookups, one per (side × tier-granularity)
    combination. Each kwarg is a per-unit lookup (`pd.Series` indexed by
    unit ID or `dict[unit_id -> value]`); units absent from a lookup
    contribute 0 overhead.

    Origin (looked up by origin unit ID at each tier):

    - `origin_cell`: per-cell-id overhead, added to every `cells_to_cells`
      AND `cells_to_zones` OD cost (both tiers have cell-id origins). Use for
      per-cell first-mile (e.g. cell-centroid → assigned network node,
      mode-specific). Mode-specific origin overhead baked here propagates
      correctly through `aggregate_across_modes`.
    - `origin_zone`: per-zone-id overhead, added to every `zones_to_zones`
      OD cost. If `origin_cell` is given and `origin_zone` is not, the
      zone-tier version is auto-derived from `origin_cell` + `cell_to_zone`
      using `zone_aggregator` (default `'mean'`).

    Destination (looked up by dest unit ID at each tier):

    - `dest_cell`: per-cell-id overhead, added to every `cells_to_cells`
      destination. Use for per-cell last-mile.
    - `dest_zone`: per-zone-id overhead, added to every `cells_to_zones`
      AND `zones_to_zones` destination (both tiers have zone-id dests).
      If `dest_cell` is given and `dest_zone` is not, the zone-tier version
      is auto-derived from `dest_cell` + `cell_to_zone` using
      `zone_aggregator`.

    Tiers not present in `costs` pass through as `None`. The input is not
    mutated; a new `TieredODGeoPairs` is returned.

    **Why the auto-derivation matters.** Leaving zone-tier overhead absent
    when cell-tier overhead is set produces silently-wrong accessibility:
    `zones_to_zones` OD pairs end up with zero overhead while
    `cells_to_cells` pairs carry the full 2× cell overhead. In nearest-k
    metrics this makes the z2z tier appear artificially cheap, and z2z
    routes (which use origin-zone rep-nodes shared by all cells in a zone)
    produce visible origin-zone outlines in the output. Auto-derivation
    closes this footgun by default.

    Args:
        costs: geo-keyed cost ODM (typically from `reindex_by_geo_unit`).
        pairs: matching geo-keyed pairs (for tier structure + dest lookups).
        origin_cell / origin_zone / dest_cell / dest_zone: see above.
        cell_to_zone: cell-id → zone-id map (`pd.Series`, `dict`, or any
            cell-indexed series with zone values). Required when a cell-tier
            overhead is given but the corresponding zone-tier overhead is
            absent AND a coarser tier exists in `costs`. Cells absent from
            the map raise `ValueError`.
        zone_aggregator: how to collapse per-cell overheads into per-zone
            scalars during auto-derivation. Any pandas groupby-compatible
            string (`'mean'`, `'median'`, etc.) or callable. Default
            `'mean'` — unweighted mean over cells in each zone.

    Raises:
        ValueError: if a cell-tier overhead is given but the corresponding
            zone-tier is needed for a present tier and `cell_to_zone` is
            missing; or if `cell_to_zone` lacks zones referenced by `costs`;
            or if any cell in a cell-tier overhead is missing from
            `cell_to_zone`.
    """
    needs_origin_zone = costs.zones_to_zones is not None
    needs_dest_zone = costs.cells_to_zones is not None or costs.zones_to_zones is not None

    if origin_zone is None and origin_cell is not None and needs_origin_zone:
        origin_zone = _aggregate_cell_to_zone(
            origin_cell,
            cell_to_zone,
            zone_aggregator,
            referenced_zones=_zones_referenced_as_origins(costs),
            side="origin",
        )
    if dest_zone is None and dest_cell is not None and needs_dest_zone:
        dest_zone = _aggregate_cell_to_zone(
            dest_cell,
            cell_to_zone,
            zone_aggregator,
            referenced_zones=_zones_referenced_as_dests(pairs),
            side="destination",
        )

    o_cell_lu = _as_lookup(origin_cell)
    o_zone_lu = _as_lookup(origin_zone)
    d_cell_lu = _as_lookup(dest_cell)
    d_zone_lu = _as_lookup(dest_zone)

    def _augment(
        cost_tier: dict | None,
        pair_tier: dict | None,
        origin_lookup: dict | None,
        dest_lookup: dict | None,
    ) -> dict | None:
        if cost_tier is None:
            return None
        out: dict = {}
        for orig, cost_arr in cost_tier.items():
            # Preserve input dtype (typically FP32 for cost ODMs) — silent
            # FP64 upcast here would double memory for the whole result.
            new_arr = np.asarray(cost_arr).copy()
            dt = new_arr.dtype
            if origin_lookup is not None:
                new_arr = new_arr + dt.type(origin_lookup.get(orig, 0.0))
            if dest_lookup is not None and pair_tier is not None:
                dest_ids = pair_tier.get(orig)
                if dest_ids is not None:
                    dest_arr = np.fromiter(
                        (dest_lookup.get(d, 0.0) for d in dest_ids), dtype=dt, count=len(dest_ids)
                    )
                    new_arr = new_arr + dest_arr
            out[orig] = new_arr
        return out

    return TieredODGeoPairs(
        cells_to_cells=_augment(costs.cells_to_cells, pairs.cells_to_cells, o_cell_lu, d_cell_lu),
        cells_to_zones=_augment(costs.cells_to_zones, pairs.cells_to_zones, o_cell_lu, d_zone_lu),
        zones_to_zones=_augment(costs.zones_to_zones, pairs.zones_to_zones, o_zone_lu, d_zone_lu),
    )
