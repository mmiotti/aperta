"""
Trip overheads — the first-mile and last-mile costs that aren't on the routed
path.

Aperta routes between network nodes, but real trips start and end at units
(cells, buildings, etc.) that are typically NOT at network nodes, and may
carry additional fixed costs at the origin or destination side (parking-find
time, station-access time, etc.). The "overhead" is the extra cost between
the actual unit and its assigned network node — at the origin (first mile)
and at the destination (last mile).

## Four kinds of overhead

Aperta supports four overhead categories, organised by **which side** and
**what granularity**:

|                                          | Origin                                | Destination                              |
|------------------------------------------|---------------------------------------|------------------------------------------|
| **Cell-specific (within one node)**      | (1) per-cell first-mile               | (4) per-cell-tier-node aggregate         |
| **Node-level (all cells at one node)**   | (2) origin overhead at node           | (3) destination overhead at node         |

- **(1) Origin overhead, cell → node** — per-cell first-mile (cell centroid →
  assigned network node). Different cells at the same node have *different*
  values. Cannot be added to a `TieredODPairs` because TieredODPairs is
  keyed by node, not by cell. Applied at accessibility-computation time via
  the `cell_overhead_column` kwarg on `gravity`, `count_in_bins`, etc.

- **(2) Origin overhead at node** — per-node overhead independent of which
  cell at that node is the actual origin (e.g., parking-find time on
  departure). Added to a `TieredODPairs` of costs upfront via
  `add_node_overheads(origin=...)`.

- **(3) Destination overhead at node** — per-destination-node overhead,
  independent of geo unit (e.g., parking-find time on arrival). Added via
  `add_node_overheads(dest_cell=..., dest_zone=...)` per
  tier.

- **(4) Destination overhead, node → cell (aggregated)** — for cell-tier
  destinations, the mean per-cell overhead across cells sharing the node;
  for zone-tier / region-tier destinations, the (weighted) average across
  cells in the group of the "intra-group" access cost. Computed via
  `aggregate_dest_overhead_per_node` (cell tier),
  `aggregate_dest_overhead_per_group_euclidean` (zone / region tier — for
  road networks, where users don't actually pass through a specific node),
  or `aggregate_dest_overhead_per_group_routed` (zone / region tier — for
  transit-style analyses where users do have to access a specific stop).
  Apply via `add_node_overheads`.

## Recommended pattern: pick ONE side

Mixing the two granularities on the same side (e.g., per-cell first-mile +
per-node origin overhead) is technically supported but makes the analysis
structure harder to reason about. We recommend picking ONE granularity per
side:

- **Cell-granularity workflow:** (1) + (4). Per-cell first-mile via
  `cell_overhead_column`; aggregated destination overhead per tier via
  this module. Natural for analyses where intra-node heterogeneity matters
  (e.g., walking accessibility with hectare cells).

- **Node-granularity workflow:** (2) + (3). Per-node origin and destination
  overheads via `add_node_overheads`. Natural for analyses where the unit-
  of-interest IS the network node (e.g., transit-stop-to-stop accessibility).

The two patterns can be mixed when there's a clear reason — but document
the reason in your project.

## Workflow

For the cell-granularity case:

```python
# 1. Per-cell first-mile (origin side, #1) — typically done in data prep
cells['walk_overhead_s'] = dist_to_node / WALK_SPEED_MS

# 2. Compute aggregated destination overheads (#4)
node_overhead = overhead.aggregate_dest_overhead_per_node(
    cells, 'walk_overhead_s')
zones['walk_dest_overhead_s'] = overhead.aggregate_dest_overhead_per_group_euclidean(
    cells, zones, speed=WALK_SPEED_MS,
    group_id_column='zone_id', cell_overhead_column='walk_overhead_s')

# 3. Apply destination overheads to costs
times_aug = overhead.add_node_overheads(
    times, pairs,
    dest_cell=node_overhead,
    dest_zone=zones.set_index('node_id')['walk_dest_overhead_s'],
)

# 4. Accessibility — origin first-mile still applied here via cell_overhead_column
accessibility.gravity(
    times_aug, weights, c2z, decays,
    cells=cells, node_column='node_id', cell_overhead_column='walk_overhead_s',
)
```
"""

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd

from aperta.od_pairs import TieredODGeoPairs, TieredODPairs
from aperta.utils import timeit


def aggregate_dest_overhead_per_node(
    cells: pd.DataFrame,
    cell_overhead_column: str,
    *,
    node_column: str = 'node_id',
    weight_column: str | None = None,
) -> pd.Series:
    """Per-network-node destination overhead — (weighted) mean of per-cell
    overheads across cells sharing each node.

    Use as `dest_cell=...` in `add_node_overheads` (overhead #4 at cell tier).

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
        node. Nodes with no associated cells are absent from the result —
        callers using the result via `add_node_overheads` will receive `0`
        overhead for those nodes (the `dict.get(..., 0)` fallback).
    """
    df = cells.dropna(subset=[node_column])
    if weight_column is None:
        return (df.groupby(node_column)[cell_overhead_column]
                  .mean()
                  .rename(f'dest_overhead_per_node({cell_overhead_column})'))

    # Weighted mean: Σ(v * w) / Σ(w) per group; skip NaN/zero-weight rows.
    # Explicit per-group loop to avoid the type-stub friction of
    # `groupby().apply()` with `include_groups`.
    out: dict = {}
    for node, idx in df.groupby(node_column).groups.items():
        sub = df.loc[idx]
        v = sub[cell_overhead_column].to_numpy(dtype=float)
        w = sub[weight_column].to_numpy(dtype=float)
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        out[node] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float('nan')
    return (pd.Series(out)
              .rename(f'dest_overhead_per_node({cell_overhead_column})'))


def aggregate_dest_overhead_per_group_euclidean(
    cells: gpd.GeoDataFrame,
    target_groups: gpd.GeoDataFrame,
    speed: float,
    *,
    group_id_column: str,
    cell_overhead_column: str | None = None,
    weight_column: str | None = None,
) -> pd.Series:
    """Per-group destination overhead — Euclidean-distance-based, for zone-
    or region-tier destinations.

    Use as `dest_zone=...` in `add_node_overheads`
    (overhead #4 at zone tier). The same function shape handles any
    via the `group_id_column` kwarg.

    For each target group `g` (with polygon centroid `g_centroid`):

        overhead(g) = (weighted) mean over cells c in g of:
            (cells[c, cell_overhead_column] if cell_overhead_column else 0)
            + euclidean(c_centroid, g_centroid) / speed

    The Euclidean variant is appropriate for road-network analyses where
    users don't actually have to pass through the group's representative
    node — the "geometric distance to a typical place in the group" is the
    more honest approximation. For transit-style analyses where users do
    have to access a specific stop, use
    `aggregate_dest_overhead_per_group_routed` instead.

    The CRS of `cells` and `target_groups` must agree and be metric
    (Euclidean distance computations require it). `speed` is in CRS-units
    per time-unit (e.g. for walking with cells in metres: `speed=1.4`).

    The `cell_overhead_column` typically encodes the mode-specific constant
    plus any feature-based overhead (e.g. β · population_density) the user
    has precomputed per cell. The Euclidean penalty is added on top.

    Args:
        cells: per-cell GeoDataFrame with polygon (or point) geometry.
            Must have `group_id_column` linking to `target_groups.index`.
        target_groups: per-group GeoDataFrame with polygon (or point)
            geometry, indexed by group ID. Polygon centroid is used as the
            "representative point".
        speed: speed in CRS-units per time-unit. Used to convert distance
            to time. Must be > 0.
        group_id_column: column on `cells` linking to `target_groups.index`
            (typically `'zone_id'` or `'region_id'`).
        cell_overhead_column: optional column on `cells` with per-cell base
            overhead (constant + feature-based), added on top of the
            Euclidean penalty.
        weight_column: optional column on `cells` to weight the mean (e.g.
            `'population'`). `None` = uniform.

    Returns:
        `pd.Series` indexed by `target_groups.index`, with one mean overhead
        per group. Groups with no constituent cells get `NaN`.
    """
    if speed <= 0:
        raise ValueError(f"`speed` must be > 0; got {speed!r}.")

    cells_valid = cells.dropna(subset=[group_id_column])
    if not len(cells_valid):
        return pd.Series(
            dtype=float, index=target_groups.index,
            name=f'dest_overhead_per_group_euclidean({group_id_column})')

    # Per-cell centroid coords (works for both polygons and points).
    cells_centroids = cells_valid.geometry.centroid
    cell_x = cells_centroids.x.to_numpy(dtype=float)
    cell_y = cells_centroids.y.to_numpy(dtype=float)

    # Per-group centroid coords, looked up by cell's group_id.
    group_centroids = target_groups.geometry.centroid
    group_x_lookup = group_centroids.x.to_dict()
    group_y_lookup = group_centroids.y.to_dict()
    cell_groups = cells_valid[group_id_column].to_numpy()
    group_x_per_cell = np.array([group_x_lookup.get(g, np.nan) for g in cell_groups])
    group_y_per_cell = np.array([group_y_lookup.get(g, np.nan) for g in cell_groups])

    # Euclidean distance per cell, divided by speed.
    distances = np.hypot(cell_x - group_x_per_cell, cell_y - group_y_per_cell)
    times = distances / speed

    if cell_overhead_column is not None:
        times = times + cells_valid[cell_overhead_column].to_numpy(dtype=float)

    # (Weighted) mean per group — explicit per-group loop to avoid the
    # type-stub friction of `groupby().apply()` with `include_groups`.
    weights = (cells_valid[weight_column].to_numpy(dtype=float)
               if weight_column is not None else np.ones(len(cells_valid)))
    out: dict = {}
    df_groups = pd.Series(cell_groups).groupby(cell_groups).groups
    for group, idx in df_groups.items():
        i = np.asarray(idx)
        v = times[i]
        w = weights[i]
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        out[group] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float('nan')

    return (pd.Series(out)
              .reindex(target_groups.index)
              .rename(f'dest_overhead_per_group_euclidean({group_id_column})'))


@timeit
def aggregate_dest_overhead_per_group_routed(
    cells: pd.DataFrame,
    target_groups: pd.DataFrame,
    graph: nx.Graph,
    weight: str,
    *,
    group_id_column: str,
    node_column: str = 'node_id',
    cell_overhead_column: str | None = None,
    weight_column: str | None = None,
    cutoff: float | None = None,
) -> pd.Series:
    """Per-group destination overhead via routing — for zone- or region-tier
    destinations.

    Use as `dest_zone=...` in `add_node_overheads`
    (overhead #4 at zone tier). The same function shape handles any
    via the `group_id_column` kwarg.

    For each target group `g` (with representative network node `g_node`):

        overhead(g) = (weighted) mean over cells c in g of:
            (cells[c, cell_overhead_column] if cell_overhead_column else 0)
            + route(g_node → c_node, weight)

    Routing direction is `g_node → c_node` (single-source Dijkstra from
    `g_node`) — by symmetry on undirected graphs this equals
    `c_node → g_node`. The "egress at destination" semantic is the
    `g_node → c_node` direction; for directed graphs (one-way streets etc.),
    that's the right direction.

    Args:
        cells: per-cell DataFrame. Must have `node_column` (network node ID)
            and `group_id_column` (target-group ID linking to
            `target_groups.index`).
        target_groups: per-group DataFrame (e.g. `zones` or `regions`),
            indexed by group ID, with `node_column` giving the group's
            representative network node.
        graph: routable networkx (or osmnx) graph.
        weight: edge attribute name used for routing (e.g. `'walk_time_s'`).
        group_id_column: column on `cells` linking to `target_groups.index`
            (typically `'zone_id'` or `'region_id'`).
        node_column: column name carrying the network node ID, on both
            `cells` and `target_groups`. Default `'node_id'`.
        cell_overhead_column: optional column on `cells` with per-cell
            first-mile overhead to add to the routed distance before
            averaging. `None` = routed cost only (zone-internal first-mile
            ignored).
        weight_column: optional column on `cells` to weight the mean (e.g.
            `'population'`). `None` = uniform.
        cutoff: optional `csg.dijkstra(limit=cutoff)` in `weight` units.
            Cells beyond it from `g_node` are treated as unreachable
            (contribute NaN, filtered from the mean). Set this comfortably
            above the longest expected last-mile in `weight` units (typical
            zone diameter ÷ slowest mode speed) to speed up routing on
            large graphs.

    Returns:
        `pd.Series` indexed by `target_groups.index`, with one mean overhead
        per group. Groups with no constituent cells (or with all cells
        unreachable from `g_node`) get `NaN`.
    """
    # Local import to keep scipy.sparse out of the module load path.
    import scipy.sparse.csgraph as csg
    from aperta.routing import _graph_to_csr
    csr, nx_to_seq, _ = _graph_to_csr(graph, weight)
    limit = cutoff if cutoff is not None else np.inf

    def _distances_from(g_node, cell_nodes):
        g_seq = nx_to_seq[g_node]
        dist_row = csg.dijkstra(csr, indices=[g_seq], limit=limit,
                                return_predecessors=False)[0]
        cell_seqs = np.fromiter((nx_to_seq[n] for n in cell_nodes),
                                dtype=np.int64, count=len(cell_nodes))
        return dist_row[cell_seqs]

    cells_valid = cells.dropna(subset=[node_column, group_id_column])
    cells_by_group = cells_valid.groupby(group_id_column)

    out: dict = {}
    for group_id, g_node in target_groups[node_column].items():
        if pd.isna(g_node):
            continue
        if group_id not in cells_by_group.groups:
            continue
        group_cells = cells_by_group.get_group(group_id)
        cell_nodes = group_cells[node_column].to_numpy()
        distances = _distances_from(g_node, cell_nodes)
        if cell_overhead_column is not None:
            first_mile = group_cells[cell_overhead_column].to_numpy(dtype=float)
            distances = distances + first_mile
        if weight_column is not None:
            wgts = group_cells[weight_column].to_numpy(dtype=float)
        else:
            wgts = np.ones_like(distances)
        m = np.isfinite(distances) & np.isfinite(wgts) & (wgts > 0)
        if not m.any():
            out[group_id] = float('nan')
            continue
        out[group_id] = float((distances[m] * wgts[m]).sum() / wgts[m].sum())

    return pd.Series(out, name=f'dest_overhead_per_group({group_id_column})')


def _as_lookup(x: pd.Series | dict | None) -> dict | None:
    """Normalise a Series-or-dict-or-None to a dict-or-None."""
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.to_dict()
    return dict(x)


@timeit
def add_node_overheads(
    costs: TieredODPairs,
    pairs: TieredODPairs,
    *,
    origin: pd.Series | dict | None = None,
    dest_cell: pd.Series | dict | None = None,
    dest_zone: pd.Series | dict | None = None,
) -> TieredODPairs:
    """Add per-node origin and destination overheads to a cost `TieredODPairs`.

    Each kwarg is a per-node lookup (`pd.Series` indexed by node ID, or a
    `dict[node_id -> overhead]`). Nodes absent from a lookup contribute `0`
    overhead.

    - `origin`: added to every OD cost whose origin matches a key. Looked up
      by the origin node of each TieredODPairs entry. Applies to all tiers
      (cells_to_cells and cells_to_zones use cell-tier origin nodes;
      zones_to_zones uses zone-tier origin nodes).
    - `dest_cell`: added to cells_to_cells OD costs, looked up by destination
      cell-tier node. Use for overhead #3 (cell-tier dest, at-node) and / or
      #4 (cell-tier dest, aggregated — from
      `aggregate_dest_overhead_per_node`).
    - `dest_zone`: added to BOTH `cells_to_zones` and `zones_to_zones` OD
      costs, looked up by destination zone-tier node. Use for overhead #3 / #4
      at zone tier (both middle and far tier have zone destinations).

    Any kwarg can be `None` (no overhead applied at that side / tier). The
    returned `TieredODPairs` is a new object — the input is not mutated.

    Note on origin overhead: the same Series is looked up at all tiers, but
    the *origin nodes themselves differ by tier* (cell-tier and middle-tier
    origins are cell-nodes; far-tier origins are zone-nodes). To apply a
    single per-cell-node origin overhead to all tiers, you would need to
    combine it with `cell_overhead_column` at accessibility time instead —
    see this module's docstring on the per-cell-vs-per-node granularity choice.

    Args:
        costs: `TieredODPairs` of routed costs.
        pairs: `TieredODPairs` of destination IDs (typically from
            `od_pairs.get_pairs`), position-aligned with `costs`.
        origin, dest_cell, dest_zone: per-node overhead lookups.

    Returns:
        New `TieredODPairs` of cost arrays with the requested overheads added.
        Tiers that are `None` in `costs` pass through as `None`.
    """
    origin_lu = _as_lookup(origin)
    dest_cell_lu = _as_lookup(dest_cell)
    dest_zone_lu = _as_lookup(dest_zone)

    def _augment(cost_tier: dict | None,
                 pair_tier: dict | None,
                 dest_lookup: dict | None) -> dict | None:
        if cost_tier is None:
            return None
        out: dict = {}
        for orig, cost_arr in cost_tier.items():
            # Preserve input dtype (typically FP32 for cost ODMs) — silent
            # FP64 upcast here would double memory for the whole result.
            new_arr = np.asarray(cost_arr).copy()
            dt = new_arr.dtype
            if origin_lu is not None:
                new_arr = new_arr + dt.type(origin_lu.get(orig, 0.0))
            if dest_lookup is not None and pair_tier is not None:
                dest_ids = pair_tier.get(orig)
                if dest_ids is not None:
                    dest_arr = np.fromiter(
                        (dest_lookup.get(d, 0.0) for d in dest_ids),
                        dtype=dt, count=len(dest_ids))
                    new_arr = new_arr + dest_arr
            out[orig] = new_arr
        return out

    return type(costs)(
        cells_to_cells=_augment(
            costs.cells_to_cells, pairs.cells_to_cells, dest_cell_lu),
        cells_to_zones=_augment(
            costs.cells_to_zones, pairs.cells_to_zones, dest_zone_lu),
        zones_to_zones=_augment(
            costs.zones_to_zones, pairs.zones_to_zones, dest_zone_lu),
    )


# ---------------------------------------------------------------------------
# Geo-keyed overhead application
# ---------------------------------------------------------------------------

@timeit
def add_geo_overheads(
    costs: TieredODGeoPairs,
    pairs: TieredODGeoPairs,
    *,
    origin_cell: pd.Series | dict | None = None,
    origin_zone: pd.Series | dict | None = None,
    dest_cell: pd.Series | dict | None = None,
    dest_zone: pd.Series | dict | None = None,
) -> TieredODGeoPairs:
    """Add per-geo-unit origin and destination overheads to a geo-keyed cost ODM.

    Geo-keyed twin of `add_node_overheads`. Four independent overhead lookups,
    one per (side × tier-granularity) combination. Each kwarg is a per-unit
    lookup (`pd.Series` indexed by unit ID or `dict[unit_id -> value]`); units
    absent from a lookup contribute 0 overhead.

    Origin (looked up by origin unit ID at each tier):

    - `origin_cell`: per-cell-id overhead, added to every `cells_to_cells`
      AND `cells_to_zones` OD cost (both tiers have cell-id origins). Use for
      per-cell first-mile (e.g. cell-centroid → assigned network node,
      mode-specific). Mode-specific origin overhead baked here propagates
      correctly through `aggregate_across_modes`.
    - `origin_zone`: per-zone-id overhead, added to every `zones_to_zones`
      OD cost. Use the per-zone average of per-cell first-mile overheads
      (cells in the same zone share a far-tier OD pair, so we collapse to
      a single zone-level scalar — see `add_origin_cell_overhead` for the
      canonical convenience wrapper).

    Destination (looked up by dest unit ID at each tier):

    - `dest_cell`: per-cell-id overhead, added to every `cells_to_cells`
      destination. Use for per-cell last-mile.
    - `dest_zone`: per-zone-id overhead, added to every `cells_to_zones`
      AND `zones_to_zones` destination (both tiers have zone-id dests).
      Plug in the output of `aggregate_dest_overhead_per_group_euclidean`
      (or `_routed`) directly — no zone-id → zone-node-id detour needed.

    Tiers not present in `costs` pass through as `None`. The input is not
    mutated; a new `TieredODGeoPairs` is returned.
    """
    o_cell_lu = _as_lookup(origin_cell)
    o_zone_lu = _as_lookup(origin_zone)
    d_cell_lu = _as_lookup(dest_cell)
    d_zone_lu = _as_lookup(dest_zone)

    def _augment(cost_tier: dict | None,
                 pair_tier: dict | None,
                 origin_lookup: dict | None,
                 dest_lookup: dict | None) -> dict | None:
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
                        (dest_lookup.get(d, 0.0) for d in dest_ids),
                        dtype=dt, count=len(dest_ids))
                    new_arr = new_arr + dest_arr
            out[orig] = new_arr
        return out

    return TieredODGeoPairs(
        cells_to_cells=_augment(
            costs.cells_to_cells, pairs.cells_to_cells, o_cell_lu, d_cell_lu),
        cells_to_zones=_augment(
            costs.cells_to_zones, pairs.cells_to_zones, o_cell_lu, d_zone_lu),
        zones_to_zones=_augment(
            costs.zones_to_zones, pairs.zones_to_zones, o_zone_lu, d_zone_lu),
    )


def add_origin_cell_overhead(
    costs: TieredODGeoPairs,
    pairs: TieredODGeoPairs,
    cells: pd.DataFrame,
    overhead_column: str,
    *,
    zone_id_column: str = 'zone_id',
    zone_aggregator: str = 'mean',
) -> TieredODGeoPairs:
    """Bake per-cell origin overhead into a geo-keyed cost ODM at all tiers.

    Convenience wrapper around `add_geo_overheads`. The per-cell first-mile
    overhead is added directly at the cell tier; at the zone and region tiers
    (where origins are zones, not cells), the per-zone aggregate of per-cell
    overheads is added — cells in the same zone share their zone-tier OD pair,
    so collapsing to a per-zone scalar is the natural granularity.

    Why this matters for cross-modal accessibility: the per-cell first-mile is
    mode-specific (`dist_to_node / WALK_SPEED_MS` vs
    `dist_to_node / CAR_SPEED_MS`). Baking it into the per-mode cost ODM
    before `aggregate_across_modes` lets the cross-modal logsum see the right
    per-mode disutility, instead of conflating origin time across modes.

    Args:
        costs: geo-keyed cost ODM (typically from `reindex_by_geo_unit`).
        pairs: matching geo-keyed pairs (for tier structure; not used for
            dest lookups here since this function only touches origins).
        cells: cell-level DataFrame indexed by `cell_id`. Must have
            `overhead_column` and (when zone-tier is populated) `zone_id_column`.
        overhead_column: per-cell overhead column on `cells`.
        zone_id_column: column on `cells` mapping each cell to its zone.
            Required iff `costs.zones_to_zones` is populated (only the
            far tier uses zone-level origin overhead — the cells_to_cells
            and cells_to_zones tiers both have cell-id origins and consume
            `origin_cell` directly).
        zone_aggregator: pandas-compatible string aggregator (default `'mean'`),
            applied to per-cell overhead values within each zone.

    Returns:
        New `TieredODGeoPairs` with the overhead added. `costs` is not mutated.
    """
    if overhead_column not in cells.columns:
        raise ValueError(f"`cells` is missing column {overhead_column!r}.")
    origin_cell = cells[overhead_column]
    origin_zone: pd.Series | None = None
    needs_zone = costs.zones_to_zones is not None
    if needs_zone:
        if zone_id_column not in cells.columns:
            raise ValueError(
                f"`cells` is missing zone-link column {zone_id_column!r} "
                f"(required because zones_to_zones costs are populated).")
        origin_zone = cells.groupby(zone_id_column)[overhead_column].agg(zone_aggregator)
    return add_geo_overheads(
        costs, pairs,
        origin_cell=origin_cell,
        origin_zone=origin_zone,
    )


def linear_per_cell_overhead(
    cells: pd.DataFrame,
    constant: float,
    feature_coefficients: dict[str, float],
) -> pd.Series:
    """Canonical aperta linear per-cell trip overhead:

        overhead(cell) = constant + Σ coef_i × cells[col_i]

    One side (origin or destination) of the per-cell trip overhead. The
    classic decomposition is a constant ("door-to-curb" time), a snap-
    distance term (`coef_i = seconds-per-metre`, `col_i = 'snap_dist'`),
    and a density term (`coef_i = seconds-per-density-unit`,
    `col_i = 'density_norm'`), but any per-cell numeric column works —
    the formula is mode- and feature-agnostic.

    NaN handling: missing values in a feature column are treated as 0
    (the assumption is that "data not available" doesn't add overhead).
    If a different convention is needed, pre-process `cells` before
    calling.

    Returns a per-cell `pd.Series` indexed like `cells`, ready to pass
    as `origin_cell=` or `dest_cell=` to `overhead.add_geo_overheads`.

    Args:
        cells: per-cell DataFrame; must contain every column named in
            `feature_coefficients`.
        constant: side constant (e.g. seconds of "door-to-curb" time).
        feature_coefficients: `{column_name -> coefficient}`. Empty dict
            is allowed (returns a constant Series).

    Returns:
        `pd.Series` of floats indexed by `cells.index`.

    See also: [[add_geo_overheads]] for applying the result to a cost
    ODM, and [[aggregate_dest_overhead_per_group_euclidean]] for
    aggregating destination-side overheads to zones / regions.
    """
    result = pd.Series(constant, index=cells.index, dtype=float)
    for col, coef in feature_coefficients.items():
        result = result + coef * cells[col].fillna(0.0)
    return result
