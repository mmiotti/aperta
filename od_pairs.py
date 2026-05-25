"""
Build tiered origin-destination tables for a routable network.

The names *cells*, *zones*, *regions* describe **tier roles**, not specific
spatial units. A cell is the finest analysis unit (typically hectare-scale or
smaller — H3 res-9/10 hexes, 100 m square grids, individual buildings); a
zone is the mid-scale aggregation (typically neighbourhood-scale — a traffic-
analysis zone, a census tract, an H3 res-7/8 hex); a region is the coarsest
aggregation (typically political — a municipality, canton, NUTS-2 unit — or
an H3 res-5/6 hex). The library imposes no constraint on what each tier
represents in the real world; what matters is that `cells ⊂ zones ⊂ regions`
in a many-to-one sense. Pick whatever instantiations match the analytical
question; the tier names just label the roles.

`get_pairs` returns a `TieredODPairs` with up to three OD dicts at three spatial
resolutions:

    cells_to_cells:    cell_node -> [cell_nodes]      # close pairs
    zones_to_zones:    zone_node -> [zone_nodes]      # medium-distance pairs
    zones_to_regions:  zone_node -> [region_nodes]    # far-distance pairs

The third tier is keyed by *zone* (not region) — so cells in different zones within
the same origin region can have different region-tier dest sets. This is what
makes the conservation invariant exact: per-zone storage gives each origin zone
its own carve-out for the rare cross-region cell-tier crossings.

Tier rule (per ordered zone pair Z, Z' with regions R, R'):

    Z == Z'  OR  d(Z, Z') < r_cells            → cell tier (per zone-pair)
    R == R'                                    → zone tier (always, w/ cell-tier carve-out)
    d(R, R') < r_zones                         → zone tier (all zone pairs in R x R')
    ∃ cell-tier (Z⊂R, Z'⊂R')                  → zone tier (promotion; same emission)
    d(R, R') < r_regions  (R ≠ R')             → region tier (per zone Z⊂R)
    else                                       → drop

Promotion: whenever any zone pair (Z⊂R, Z'⊂R') is at cell tier, the whole region
pair (R, R') is upgraded to zone tier. This guarantees conservation: the cells in
R' that aren't directly at cell-tier from a given origin Z get picked up by the
zone-tier emissions of the other zones in R x R' — no under-count, no double-
count via the (now absent) region-tier emission.

Symmetry: `d(Z, Z')` and `d(R, R')` are symmetric, so the tier assignment is too.
The output is therefore inherently symmetric — no `ensure_symmetrical` flag.

`r_zones` is consulted in two places:
    - With regions: as the region-pair-distance threshold separating zone tier
      from region tier.
    - Without regions: as the zone-pair-distance threshold for the zone tier (the
      only way to bound it without a region hierarchy to make zone tier unconditional).

Storage caveat: when promotion fires, all `|Z⊂R| × |Z'⊂R'|` zone pairs are emitted.
Usually small (promotion only fires when a cell-tier crossing exists, which is
rare across regions), but worth knowing for r_cells / r_zones sizing decisions.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from numba import njit
from shapely.geometry import Point

from aperta.utils import timeit


@dataclass(frozen=True)
class TieredODPairs:
    """Abstract base for tiered origin-destination pair tables.

    Holds three per-tier dict-of-arrays. The two concrete subclasses differ in
    what the dict KEYS represent — see [[TieredODNodePairs]] and
    [[TieredODGeoPairs]] below. Functions that don't care about key space (e.g.
    `make_mask`, `__repr__`, `describe`) accept this base type.

    `zones_to_zones` and `zones_to_regions` are `None` when the corresponding
    tier wasn't requested.
    """
    cells_to_cells: dict
    zones_to_zones: dict | None = None
    zones_to_regions: dict | None = None

    def __repr__(self) -> str:
        # Concise summary instead of the dataclass default (which would dump every
        # destination dict in full). Counts origins (dict keys) and total dests
        # (sum of array lengths) per tier. `type(self).__name__` so subclasses
        # report their own name.
        def _summary(d: dict | None) -> str:
            if d is None:
                return 'None'
            n_orig = len(d)
            n_dest = sum(len(v) for v in d.values())
            return f'{n_orig:,} orig → {n_dest:,} dest'
        return (
            f'{type(self).__name__}('
            f'cells_to_cells: {_summary(self.cells_to_cells)}; '
            f'zones_to_zones: {_summary(self.zones_to_zones)}; '
            f'zones_to_regions: {_summary(self.zones_to_regions)})'
        )

    def describe(self) -> str:
        """Print and return a richer per-tier summary than `repr()`.

        Always shows origin and destination counts. For tiers whose values are
        numeric (typical for cost / distance / weight TieredODPairs), also shows
        mean, median, 5-95th percentile, and min-max. Non-finite entries
        (`np.inf` for unreachable / masked-out, `np.nan`) are excluded from the
        stats and reported separately. For bool-typed tiers (masks), reports the
        True count and rate instead.

        Goes via `print` (not `logging`) so it is always visible regardless of
        logging config, and returns the string so the caller can route it
        elsewhere (e.g. into a log file).
        """
        lines = [f"=== {type(self).__name__} ==="]
        for tier_name in ('cells_to_cells', 'zones_to_zones', 'zones_to_regions'):
            d: dict | None = getattr(self, tier_name)
            if d is None:
                lines.append(f"  {tier_name}: None")
                continue
            if not d:
                lines.append(f"  {tier_name}: empty")
                continue
            n_orig = len(d)
            n_dest = sum(len(v) for v in d.values())
            lines.append(f"  {tier_name}: {n_orig:,} origins, {n_dest:,} dests")
            try:
                all_values = np.concatenate([np.asarray(v) for v in d.values()])
            except (ValueError, TypeError):
                continue  # mixed shapes / non-array values; skip stats
            if all_values.size == 0:
                continue
            kind = all_values.dtype.kind
            if kind == 'b':
                n_true = int(all_values.sum())
                lines.append(
                    f"    True: {n_true:,} / {all_values.size:,} "
                    f"({n_true / all_values.size * 100:.1f}%)")
                continue
            if kind not in ('f', 'i', 'u'):
                continue  # not numeric (e.g. string IDs) — counts only
            finite = all_values[np.isfinite(all_values)]
            n_non_finite = int(all_values.size - finite.size)
            if finite.size == 0:
                lines.append(f"    (no finite values; {n_non_finite:,} non-finite)")
                continue
            p5, p95 = np.percentile(finite, [5, 95])
            line = (
                f"    mean {float(np.mean(finite)):.1f} / "
                f"median {float(np.median(finite)):.1f} / "
                f"5-95p [{float(p5):.1f}, {float(p95):.1f}] / "
                f"min-max [{float(finite.min()):.1f}, {float(finite.max()):.1f}]"
            )
            if n_non_finite:
                line += f" / non-finite {n_non_finite:,}"
            lines.append(line)
        out = '\n'.join(lines)
        print(out)
        return out


class TieredODNodePairs(TieredODPairs):
    """Tiered OD pairs keyed by network node IDs of one mode's graph.

    Dict keys at every tier are network node IDs; per-origin arrays carry
    destination node IDs (for `get_pairs`) or per-OD-pair values (costs, weights,
    masks, distances) aligned to those dest IDs.

    Produced by `get_pairs`, `routing.tiered_path_costs`,
    `routing.tiered_path_aggregate`, `dest_values`, `get_euclidian_dists`,
    `make_mask`, `overhead.add_node_overheads`, `utility.route_utility`,
    `utility.add_endpoint_utility`. The default working representation for
    single-mode pipelines — lightweight, no fan-out.

    Accessibility metrics consuming a `TieredODNodePairs` return *node-indexed*
    DataFrames. For per-cell accessibility output, per-cell origin overhead,
    or cross-modal aggregation, lift to `TieredODGeoPairs` via
    `od_pairs.reindex_by_geo_unit`.
    """


class TieredODGeoPairs(TieredODPairs):
    """Tiered OD pairs keyed by geo-unit IDs (cell_id / zone_id / region_id).

    Dict keys are mode-agnostic geo-unit IDs:
        cells_to_cells:    cell_id  ->  array of dest cell_ids
        zones_to_zones:    zone_id  ->  array of dest zone_ids
        zones_to_regions:  zone_id  ->  array of dest region_ids

    Created via `od_pairs.reindex_by_geo_unit` from a `TieredODNodePairs` +
    cells (+ optional zones, regions). Required input to:
      - `od_pairs.aggregate_across_modes` for cross-modal accessibility,
      - accessibility metrics that should return cell/zone-indexed output,
      - `add_geo_overheads` / `add_origin_cell_overhead` for geo-unit-keyed
        overhead baking.

    Heavier than `TieredODNodePairs` (multiple cells sharing a node fan out into
    per-cell entries), but mode-agnostic by construction: cell IDs are the same
    across modes, so per-mode geo-keyed ODMs align directly.
    """


@njit
def _points_within_buffer(
    x_origin: float,
    y_origin: float,
    xy_destinations: np.ndarray,
    buffer: float,
) -> np.ndarray:
    """Sequential positions of destinations within `buffer` (euclidean) of `(x_origin, y_origin)`.

    Numba-JIT-compiled, ~10-15× faster than the equivalent
    `centroids.within(Point(x, y).buffer(r))` path. Strict-less-than matches
    shapely's `.within()` boundary semantics: a point at exactly `buffer` distance
    is excluded.
    """
    matches = np.nonzero(np.sqrt(np.power(xy_destinations[:, 0] - x_origin, 2) +
                                 np.power(xy_destinations[:, 1] - y_origin, 2)) < buffer)[0]
    return matches


def _build_node_xy_map(nodes: pd.DataFrame | gpd.GeoDataFrame) -> dict:
    """Build a `{node_id -> (x, y)}` dict from a DataFrame or GeoDataFrame.

    Accepted input shapes:
      - GeoDataFrame with Point geometries (preferred — `geom.x` / `geom.y` are used).
      - DataFrame (or GeoDataFrame without usable geometry) with `'x'` and `'y'` columns.

    Index of `nodes` is used as the node ID. Rows with null geometry / missing
    coords are silently skipped.
    """
    out: dict = {}
    if isinstance(nodes, gpd.GeoDataFrame) and 'geometry' in nodes.columns:
        for nid, geom in nodes.geometry.items():
            if geom is None or geom.is_empty:
                continue
            if not isinstance(geom, Point):
                raise ValueError(
                    f"Node {nid!r}: geometry is {type(geom).__name__}, expected Point. "
                    f"For non-Point geometries, precompute centroids "
                    f"(e.g. nodes.assign(geometry=nodes.geometry.centroid)) or pass a "
                    f"plain DataFrame with 'x' and 'y' columns.")
            out[nid] = (float(geom.x), float(geom.y))
        return out
    if 'x' not in nodes.columns or 'y' not in nodes.columns:
        raise ValueError(
            "`nodes` must be a GeoDataFrame with Point geometries OR a DataFrame "
            "with 'x' and 'y' columns.")
    for nid, x, y in zip(nodes.index, nodes['x'], nodes['y']):
        out[nid] = (float(x), float(y))
    return out


def _node_to_value_lookup(df: pd.DataFrame, node_column: str, value_column: str) -> dict:
    """Build a `{node_id -> value}` dict, summing values when multiple rows share a node.

    A node can have several units mapped to it (e.g. two cells whose nearest-
    network-node is the same N). For an additive metric like population, the
    node's effective value is the sum across those units.
    """
    rows_with_node = df[df[node_column].notna()]
    if len(rows_with_node) == 0:
        return {}
    return rows_with_node.groupby(node_column)[value_column].sum().to_dict()


# ---------------------------------------------------------------------------
# Tiered-OD lookup builders
# ---------------------------------------------------------------------------

def build_cell_to_zone_node_map(
    cells: pd.DataFrame, zones: pd.DataFrame, node_column: str,
) -> dict:
    """Build the `{cell_node -> zone_node}` lookup that tiered helpers use to find
    each origin cell's parent zone (which keys `zones_to_zones` and `zones_to_regions`).

    Cells without a mapped network node, or whose zone has no mapped network node,
    are omitted (they can't participate in zone- or region-tier sampling).
    """
    zone_to_node = zones[node_column].dropna().to_dict()
    cells_valid = cells[cells[node_column].notna()]
    return {
        cell_node: zone_to_node[zone_id]
        for cell_node, zone_id in zip(cells_valid[node_column], cells_valid['zone_id'])
        if zone_id in zone_to_node
    }


def make_mask(
    values: TieredODPairs,
    rule: Callable[[np.ndarray], np.ndarray],
) -> TieredODPairs:
    """Build a boolean-mask `TieredODPairs` by applying `rule` to every per-origin
    value array.

    `rule` is a vectorized callable: it takes a 1-D numpy array and returns a
    bool array of the same length, e.g. `lambda d: d < 50_000` to keep only
    pairs with distance under 50 km.

    The returned `TieredODPairs` has the same structure as `values` (same
    origins, same per-origin array lengths) but with bool arrays. Pass it as
    `mask=` to `routing.tiered_path_costs`, `traffic_flows.nested_node_sample`,
    and other tiered helpers to ignore `False` entries.

    Tiers that are `None` in `values` stay `None` in the result.
    """
    def _apply(tier: dict | None) -> dict | None:
        if tier is None:
            return None
        return {origin: np.asarray(rule(arr), dtype=bool)
                for origin, arr in tier.items()}
    # Preserve the input subclass — masks make sense for either key space.
    return type(values)(
        cells_to_cells=_apply(values.cells_to_cells),
        zones_to_zones=_apply(values.zones_to_zones),
        zones_to_regions=_apply(values.zones_to_regions),
    )


# ---------------------------------------------------------------------------
# get_pairs
# ---------------------------------------------------------------------------

def _validate_inputs(
    cells: gpd.GeoDataFrame,
    node_column: str,
    zones: gpd.GeoDataFrame | None,
    r_zones: float | None,
    regions: gpd.GeoDataFrame | None,
    r_regions: float | None,
) -> None:
    if (zones is None) != (r_zones is None):
        raise ValueError("`zones` and `r_zones` must both be provided or both omitted.")
    if (regions is None) != (r_regions is None):
        raise ValueError("`regions` and `r_regions` must both be provided or both omitted.")
    if regions is not None and zones is None:
        raise ValueError("`regions` requires `zones` (region tier sits above zone tier).")
    if node_column not in cells.columns:
        raise ValueError(f"`cells` is missing required column {node_column!r}.")
    if zones is not None:
        if node_column not in zones.columns:
            raise ValueError(f"`zones` is missing required column {node_column!r}.")
        if 'zone_id' not in cells.columns:
            raise ValueError("`cells` must have a 'zone_id' column when zones are provided.")
    if regions is not None:
        assert zones is not None
        if node_column not in regions.columns:
            raise ValueError(f"`regions` is missing required column {node_column!r}.")
        if 'region_id' not in zones.columns:
            raise ValueError("`zones` must have a 'region_id' column when regions are provided.")


def _get_pairs_cells_only(
    cells_with_node: gpd.GeoDataFrame,
    r_cells: float,
    node_column: str,
    *,
    orig_node_set: set | None = None,
    dest_node_set: set | None = None,
) -> TieredODPairs:
    """Single-tier fallback when no zones are supplied: per-cell distance filter.

    The output OD matrix is node-keyed, so we dedupe `cells_with_node` to one row
    per unique network node first (many cells can map to the same node, especially
    at hectare resolution). The node's representative coordinate is the mean of
    its contributing cells' centroids — for hectare cells the resulting positional
    error is well below typical `r_cells` values. This reduces both the outer loop
    length and the per-iteration distance array from N_cells to N_unique_nodes;
    for a 500k-cells / 50k-nodes dataset that's a ~100× speedup.

    Optional `orig_node_set` and `dest_node_set` filter origins / destinations
    to a subset of nodes — see `get_pairs` for the user-level semantics.
    """
    centroids = cells_with_node.geometry.centroid
    per_node = pd.DataFrame({
        'node': cells_with_node[node_column].to_numpy(),
        'x': centroids.x.to_numpy(),
        'y': centroids.y.to_numpy(),
    }).groupby('node', sort=False).mean()
    node_ids = per_node.index.to_numpy()
    xy = per_node[['x', 'y']].to_numpy()

    cells_to_cells: defaultdict = defaultdict(set)
    for i in range(len(per_node)):
        if orig_node_set is not None and node_ids[i] not in orig_node_set:
            continue
        positions = _points_within_buffer(xy[i, 0], xy[i, 1], xy, r_cells)
        dests = node_ids[positions]
        if dest_node_set is not None:
            dests = [n for n in dests if n in dest_node_set]
            if not dests:
                continue
        else:
            dests = dests.tolist()
        cells_to_cells[node_ids[i]].update(dests)
    return TieredODNodePairs(
        cells_to_cells={k: np.asarray(list(v)) for k, v in cells_to_cells.items()},
    )


def _mask_to_node_set(
    mask: pd.Series | np.ndarray | None,
    df: pd.DataFrame,
    node_column: str,
    df_name: str,
) -> set | None:
    """Convert a boolean mask aligned with `df` to a set of node IDs.

    Returns `None` when the mask itself is `None` (signalling 'no filter' to
    downstream emission loops).
    """
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.dtype != bool:
        raise ValueError(
            f"`{df_name}` mask must be a boolean array; got dtype {arr.dtype}.")
    if len(arr) != len(df):
        raise ValueError(
            f"`{df_name}` mask length {len(arr)} does not match `{df_name}` "
            f"length {len(df)}.")
    sub = df[arr]
    return set(sub[sub[node_column].notna()][node_column].unique().tolist())


@timeit
def get_pairs(
    cells: gpd.GeoDataFrame,
    r_cells: float,
    node_column: str,
    *,
    zones: gpd.GeoDataFrame | None = None,
    r_zones: float | None = None,
    zones_centroids: gpd.GeoSeries | None = None,
    regions: gpd.GeoDataFrame | None = None,
    r_regions: float | None = None,
    regions_centroids: gpd.GeoSeries | None = None,
    orig_cells: pd.Series | np.ndarray | None = None,
    dest_cells: pd.Series | np.ndarray | None = None,
    dest_zones: pd.Series | np.ndarray | None = None,
    dest_regions: pd.Series | np.ndarray | None = None,
) -> TieredODPairs:
    """Build a tiered OD-pair table with cell, zone, and (zone→region) tiers.

    See the module docstring for the tier-assignment rule and conservation
    guarantees. Each tier in the returned `TieredODPairs` is at its own
    resolution; cells, zones, and regions are looked up by their respective
    `node_column` values.

    Required input contract:
        cells:   GeoDataFrame; `node_column` must give the network node for each
                 cell (NaN cells contribute no destinations). If `zones` is given,
                 also requires `zone_id` column.
        zones:   if provided, must have `node_column`. If `regions` is also given,
                 must have `region_id`.
        regions: if provided, must have `node_column`. Implies `zones`.

    Args:
        cells: cell-level GeoDataFrame.
        r_cells: per-zone-pair distance threshold (CRS units, typically metres) for
            cell tier. Zone pairs closer than this emit per-cell OD pairs.
        node_column: column name on cells/zones/regions holding the network node ID.
        zones: optional zones GeoDataFrame to enable the zone (and zone→region) tier.
        r_zones: per-region-pair distance threshold for zone tier (when regions are
            given) OR per-zone-pair threshold (when regions are not). Required iff
            `zones` is given.
        zones_centroids: optional custom zone centroids (e.g. population-weighted).
            Falls back to `zones.geometry.centroid`.
        regions: optional regions GeoDataFrame to enable the third (zones→regions) tier.
        r_regions: per-region-pair distance threshold for region tier. Required iff
            `regions` is given.
        regions_centroids: optional custom region centroids; falls back to
            `regions.geometry.centroid`.

        orig_cells: optional boolean mask (Series or numpy array) aligned with
            `cells.index`. When provided, only cells where the mask is `True`
            act as origins; cells where `False` contribute no OD pairs FROM
            them. `None` (default) treats every cell as an origin.
        dest_cells: optional boolean mask aligned with `cells.index`. When
            provided, only cells where `True` are emitted as cell-tier
            destinations (other cells can still be routed TO, just not at
            cell-tier resolution). `None` = every cell is a valid cell-tier
            destination.
        dest_zones: optional boolean mask aligned with `zones.index`. When
            provided, only zones where `True` are emitted as zone-tier
            destinations. `None` = every zone is a valid zone-tier dest.
        dest_regions: optional boolean mask aligned with `regions.index`.
            When provided, only regions where `True` are emitted as
            region-tier destinations. `None` = every region is a valid
            region-tier dest.

        The mask filters are critical for **large-area analyses**: when most
        of the area has no opportunity-of-interest (e.g. supermarkets exist in
        ~1% of cells), filtering to relevant destinations drops OD-pair counts
        by 1-2 orders of magnitude and routing time accordingly. For small
        areas like the minimal example, the filters can usually be left as
        `None`.

    Returns:
        `TieredODPairs` with `cells_to_cells` always populated, plus
        `zones_to_zones` and `zones_to_regions` if their tier is enabled.
    """
    _validate_inputs(cells, node_column, zones, r_zones, regions, r_regions)
    cells_with_node = cells[cells[node_column].notna()]

    # Convert masks to node sets (or None = no filter). Length-mismatched or
    # non-boolean masks raise a clear error here, before any heavy work.
    orig_node_set = _mask_to_node_set(orig_cells, cells, node_column, 'orig_cells')
    dest_cell_node_set = _mask_to_node_set(dest_cells, cells, node_column, 'dest_cells')
    dest_zone_node_set = (_mask_to_node_set(dest_zones, zones, node_column, 'dest_zones')
                          if zones is not None else None)
    dest_region_node_set = (_mask_to_node_set(dest_regions, regions, node_column, 'dest_regions')
                            if regions is not None else None)

    if zones is None:
        return _get_pairs_cells_only(
            cells_with_node, r_cells, node_column,
            orig_node_set=orig_node_set,
            dest_node_set=dest_cell_node_set,
        )
    assert r_zones is not None

    # --- Setup ---
    if zones_centroids is None:
        zones_centroids = zones.geometry.centroid
    zone_ids: list = zones.index.tolist()
    zone_xy = np.column_stack([zones_centroids.x.to_numpy(), zones_centroids.y.to_numpy()])
    zone_nodes: list = zones[node_column].tolist()
    cells_in_zone: dict = (
        cells_with_node.groupby('zone_id')[node_column]
        .apply(lambda s: s.unique()).to_dict()
    )
    n_zones = len(zone_ids)

    # --- Pre-compute cell-tier zone pairs (per-zone-pair, d(Z, Z') < r_cells) ---
    # `cell_tier_dests[i]` is the array of zone indices j where (Z_i, Z_j) is cell-tier
    # (includes j == i — same-zone is always cell-tier).
    logging.info(f"get_pairs: tiered pass over {n_zones:,} zones...")
    log_every = max(1, n_zones // 10)
    cell_tier_dests: list[np.ndarray] = []
    for i in range(n_zones):
        d = np.hypot(zone_xy[:, 0] - zone_xy[i, 0], zone_xy[:, 1] - zone_xy[i, 1])
        mask = d < r_cells
        mask[i] = True
        cell_tier_dests.append(np.where(mask)[0])

    # --- Region setup (only when regions provided) ---
    use_regions = regions is not None
    zone_region_idx = np.full(n_zones, -1, dtype=np.int64)
    region_ids: list = []
    region_node_of: dict = {}
    zone_tier_mat: np.ndarray | None = None  # (n_regions, n_regions); cross-region only
    region_tier_mat: np.ndarray | None = None
    if use_regions:
        assert regions is not None and r_regions is not None
        if regions_centroids is None:
            regions_centroids = regions.geometry.centroid
        region_ids = regions.index.tolist()
        region_node_of = regions[node_column].to_dict()
        region_xy = np.column_stack(
            [regions_centroids.x.to_numpy(), regions_centroids.y.to_numpy()]
        )
        region_idx_of = {r: i for i, r in enumerate(region_ids)}
        zone_to_region = zones['region_id'].to_dict()
        zone_region_idx = np.array(
            [region_idx_of.get(zone_to_region.get(z), -1) for z in zone_ids], dtype=np.int64,
        )

        # --- Promotion trigger: cross-region cell-tier (R, R') pairs ---
        n_regions = len(region_ids)
        cross_region_cell_tier = np.zeros((n_regions, n_regions), dtype=bool)
        for i in range(n_zones):
            ri = zone_region_idx[i]
            if ri < 0:
                continue
            for j in cell_tier_dests[i]:
                rj = zone_region_idx[j]
                if rj >= 0 and ri != rj:
                    cross_region_cell_tier[ri, rj] = True

        # --- Region-pair tier matrices ---
        d_rr = np.hypot(
            region_xy[:, 0:1] - region_xy[:, 0:1].T,
            region_xy[:, 1:2] - region_xy[:, 1:2].T,
        )
        zone_tier_mat = (d_rr < r_zones) | cross_region_cell_tier
        np.fill_diagonal(zone_tier_mat, False)  # same-region handled separately
        region_tier_mat = (d_rr < r_regions) & ~zone_tier_mat
        np.fill_diagonal(region_tier_mat, False)

    # --- Identify zones that contain at least one origin cell ---
    # When `orig_cells` filter is active, we skip zones that contribute no
    # origin nodes — they have no cells_to_cells / zones_to_zones / region-tier
    # OD pairs since there's nothing to route FROM. Without the filter, every
    # zone is potentially an origin (current behaviour).
    if orig_node_set is not None:
        zones_with_origin: set = {
            zone_ids[i] for i in range(n_zones)
            if any(n in orig_node_set
                   for n in cells_in_zone.get(zone_ids[i], np.array([])))
        }
    else:
        zones_with_origin = None  # signals "every zone"

    # --- Emit cells_to_cells ---
    cells_to_cells: defaultdict = defaultdict(set)
    for i in range(n_zones):
        if i and i % log_every == 0:
            logging.info(f"  cells_to_cells: {i:,} of {n_zones:,} origin zones")
        if zones_with_origin is not None and zone_ids[i] not in zones_with_origin:
            continue
        origin_nodes = cells_in_zone.get(zone_ids[i], np.array([]))
        if orig_node_set is not None:
            origin_nodes = np.array(
                [n for n in origin_nodes if n in orig_node_set], dtype=origin_nodes.dtype)
        if len(origin_nodes) == 0:
            continue
        for j in cell_tier_dests[i]:
            dest_nodes = cells_in_zone.get(zone_ids[j], np.array([]))
            if len(dest_nodes) == 0:
                continue
            if dest_cell_node_set is not None:
                dest_set = {n for n in dest_nodes.tolist() if n in dest_cell_node_set}
            else:
                dest_set = set(dest_nodes.tolist())
            if not dest_set:
                continue
            for orig in origin_nodes:
                cells_to_cells[orig].update(dest_set)

    # --- Emit zones_to_zones ---
    zones_to_zones: defaultdict = defaultdict(set)
    zone_nodes_arr = np.array(zone_nodes, dtype=object)
    # Pre-compute per-zone-index "is this zone an eligible destination?" mask
    # if dest_zones filter is active.
    if dest_zone_node_set is not None:
        zone_is_dest = np.array(
            [zone_nodes[i] in dest_zone_node_set for i in range(n_zones)], dtype=bool)
    else:
        zone_is_dest = None
    for i in range(n_zones):
        if i and i % log_every == 0:
            logging.info(f"  zones_to_zones: {i:,} of {n_zones:,} origin zones")
        if zones_with_origin is not None and zone_ids[i] not in zones_with_origin:
            continue
        origin_zone_node = zone_nodes[i]
        if pd.isna(origin_zone_node):
            continue
        # cell-tier mask for origin i
        cell_mask = np.zeros(n_zones, dtype=bool)
        cell_mask[cell_tier_dests[i]] = True
        # zone-tier eligible (excluding cell-tier overlap)
        if use_regions:
            assert zone_tier_mat is not None
            ri = zone_region_idx[i]
            if ri < 0:
                continue
            same_region = zone_region_idx == ri
            cross_zt = np.zeros(n_zones, dtype=bool)
            valid = zone_region_idx >= 0
            cross_zt[valid] = zone_tier_mat[ri, zone_region_idx[valid]]
            zone_eligible = (same_region | cross_zt) & ~cell_mask
            zone_eligible[i] = False  # self handled at cell tier
        else:
            d = np.hypot(zone_xy[:, 0] - zone_xy[i, 0], zone_xy[:, 1] - zone_xy[i, 1])
            zone_eligible = (d < r_zones) & ~cell_mask
            zone_eligible[i] = False
        if zone_is_dest is not None:
            zone_eligible = zone_eligible & zone_is_dest
        dest_nodes_for_zone = zone_nodes_arr[zone_eligible]
        # Filter out NaN dest nodes.
        valid_dests = [
            n for n in dest_nodes_for_zone.tolist() if not (isinstance(n, float) and np.isnan(n))
        ]
        if valid_dests:
            zones_to_zones[origin_zone_node].update(valid_dests)

    # --- Emit zones_to_regions ---
    zones_to_regions: defaultdict = defaultdict(set)
    if use_regions and region_tier_mat is not None:
        for i in range(n_zones):
            if zones_with_origin is not None and zone_ids[i] not in zones_with_origin:
                continue
            origin_zone_node = zone_nodes[i]
            if pd.isna(origin_zone_node):
                continue
            ri = zone_region_idx[i]
            if ri < 0:
                continue
            for rj in np.where(region_tier_mat[ri])[0]:
                R_prime = region_ids[rj]
                dest_region_node = region_node_of.get(R_prime)
                if dest_region_node is None or pd.isna(dest_region_node):
                    continue
                if (dest_region_node_set is not None
                        and dest_region_node not in dest_region_node_set):
                    continue
                zones_to_regions[origin_zone_node].add(dest_region_node)

    return TieredODNodePairs(
        cells_to_cells={k: np.asarray(list(v)) for k, v in cells_to_cells.items()},
        zones_to_zones=(
            {k: np.asarray(list(v)) for k, v in zones_to_zones.items()}
            if zones is not None else None
        ),
        zones_to_regions=(
            {k: np.asarray(list(v)) for k, v in zones_to_regions.items()}
            if use_regions else None
        ),
    )


# ---------------------------------------------------------------------------
# Value lookups
# ---------------------------------------------------------------------------

def node_values(
    column: str,
    node_list: pd.Series | list | np.ndarray,
    df: pd.DataFrame,
    node_column: str,
) -> np.ndarray:
    """Single-tier lookup of `column` for every node in a list of node IDs."""
    if column not in df.columns:
        raise ValueError(f"`df` is missing column {column!r}.")
    df_lookup = _node_to_value_lookup(df, node_column, column)
    return np.array([df_lookup[node_id] for node_id in node_list])


def dest_values(
    column: str,
    pairs: TieredODPairs,
    cells: pd.DataFrame,
    node_column: str,
    zones: pd.DataFrame | None = None,
    regions: pd.DataFrame | None = None,
) -> TieredODPairs:
    """Look up `column` for every destination in `pairs`, tier by tier.

    Returns a `TieredODPairs` of value arrays paired position-wise with the input
    destination arrays. The third tier (`zones_to_regions`) looks values up in
    `regions[column]` at the region node IDs.

    Conservation invariant: if `column` is additive (e.g. population), is
    consistently aggregated through the cells → zones → regions hierarchy, and
    `r_regions` covers every region, then for each origin cell `i` the sum of
    values across all three tiers equals the total of `cells[column]` (no double-
    counting across tiers).
    """
    if column not in cells.columns:
        raise ValueError(f"`cells` is missing column {column!r}.")
    if pairs.zones_to_zones is not None:
        if zones is None:
            raise ValueError("`zones` is required because `pairs.zones_to_zones` is set.")
        if column not in zones.columns:
            raise ValueError(f"`zones` is missing column {column!r}.")
    if pairs.zones_to_regions is not None:
        if regions is None:
            raise ValueError("`regions` is required because `pairs.zones_to_regions` is set.")
        if column not in regions.columns:
            raise ValueError(f"`regions` is missing column {column!r}.")

    def _lookup_for(d: dict, lookup: dict) -> dict:
        return {origin: np.array([lookup.get(dest, np.nan) for dest in dests])
                for origin, dests in d.items()}

    cells_lookup = _node_to_value_lookup(cells, node_column, column)
    zones_out = None
    if pairs.zones_to_zones is not None:
        assert zones is not None
        zones_out = _lookup_for(pairs.zones_to_zones,
                                _node_to_value_lookup(zones, node_column, column))
    z_to_r_out = None
    if pairs.zones_to_regions is not None:
        assert regions is not None
        z_to_r_out = _lookup_for(pairs.zones_to_regions,
                                 _node_to_value_lookup(regions, node_column, column))
    return TieredODNodePairs(
        cells_to_cells=_lookup_for(pairs.cells_to_cells, cells_lookup),
        zones_to_zones=zones_out,
        zones_to_regions=z_to_r_out,
    )


# ---------------------------------------------------------------------------
# Geo-unit reindexing (node-keyed → geo-unit-keyed)
# ---------------------------------------------------------------------------

def _build_node_to_units_map(units: pd.DataFrame, node_column: str) -> dict:
    """Build `{node_id -> list of unit_ids whose `node_column` is that node}`.

    Units with NaN node IDs are dropped. The lists are returned in `units.index`
    order, which `reindex_by_geo_unit` then sorts when assembling per-origin
    dest arrays to give canonical ordering across modes.
    """
    valid = units[units[node_column].notna()]
    out: dict = {}
    for unit_id, node_id in zip(valid.index, valid[node_column]):
        out.setdefault(node_id, []).append(unit_id)
    return out


def _reindex_tier(
    tier_pairs: dict | None,
    tier_odm: dict | None,
    origin_units: pd.DataFrame,
    dest_units: pd.DataFrame,
    origin_node_column: str,
    dest_node_column: str,
) -> tuple[dict | None, dict | None]:
    """Reindex one tier from node-keyed to geo-unit-keyed.

    For each origin unit (cell/zone), look up its network node, find that
    node's per-origin entry in the node-keyed tier, then fan out each dest node
    to all dest units sharing it. Returns sorted-by-dest-id arrays for
    canonical ordering across modes (required for `aggregate_across_modes`).

    Args:
        tier_pairs: node-keyed `{origin_node -> dest_node_ids array}`.
        tier_odm: node-keyed `{origin_node -> values array}` aligned to
            `tier_pairs`. `None` when the caller only wants to reindex pairs.
        origin_units: DataFrame indexed by origin unit ID (cell or zone) with
            `origin_node_column` giving each unit's network node.
        dest_units: DataFrame indexed by dest unit ID, with `dest_node_column`
            giving each dest unit's network node. (Same as `origin_units` for
            same-tier reindexing.)

    Returns:
        `(new_pairs, new_odm)` — geo-keyed dicts of dest-unit-ID arrays and
        value arrays, sorted by dest-unit-ID per origin. `new_odm` is `None`
        iff `tier_odm` was `None`.
    """
    if tier_pairs is None:
        return None, None
    dest_node_to_units = _build_node_to_units_map(dest_units, dest_node_column)
    new_pairs: dict = {}
    new_odm: dict | None = {} if tier_odm is not None else None
    origin_valid = origin_units[origin_units[origin_node_column].notna()]
    for origin_unit, origin_node in zip(origin_valid.index, origin_valid[origin_node_column]):
        if origin_node not in tier_pairs:
            continue
        dest_node_arr = tier_pairs[origin_node]
        if tier_odm is not None:
            value_arr = np.asarray(tier_odm[origin_node])
        # Fan out: for each dest_node, emit one row per dest_unit sharing that node.
        out_dest_units: list = []
        out_values: list = []
        for i, dn in enumerate(dest_node_arr):
            units_at_node = dest_node_to_units.get(dn)
            if not units_at_node:
                continue
            for du in units_at_node:
                out_dest_units.append(du)
                if tier_odm is not None:
                    out_values.append(value_arr[i])
        if not out_dest_units:
            continue
        # Canonical sort by dest-unit ID — required for cross-mode alignment.
        dest_arr = np.asarray(out_dest_units)
        order = np.argsort(dest_arr, kind='stable')
        new_pairs[origin_unit] = dest_arr[order]
        if tier_odm is not None:
            assert new_odm is not None
            new_odm[origin_unit] = np.asarray(out_values)[order]
    return new_pairs, new_odm


@timeit
def reindex_by_geo_unit(
    pairs: TieredODNodePairs,
    odm: TieredODNodePairs | None,
    cells: pd.DataFrame,
    *,
    cell_node_column: str,
    zones: pd.DataFrame | None = None,
    zone_node_column: str | None = None,
    regions: pd.DataFrame | None = None,
    region_node_column: str | None = None,
) -> tuple[TieredODGeoPairs, TieredODGeoPairs | None]:
    """Convert a node-keyed (pairs, odm) pair into geo-unit-keyed form.

    Keys at each tier become:
        cells_to_cells   : cell_id (from `cells.index`) → cell_id dest array
        zones_to_zones   : zone_id (from `zones.index`) → zone_id dest array
        zones_to_regions : zone_id → region_id dest array

    Dest arrays are sorted by ID per origin — this canonical ordering enables
    cross-mode alignment in `aggregate_across_modes` (different modes produce
    different node-level snapping, but their geo-keyed forms align on the
    shared cell / zone / region ID universe).

    Fan-out: each (origin_node, dest_node) entry in the input expands to
    `|cells at origin_node| × |cells at dest_node|` entries at cell tier (same
    pattern at zone tier). Memory cost scales with average units-per-node.

    Args:
        pairs: node-keyed destination-ID table from `get_pairs`.
        odm: node-keyed cost / utility / value ODM aligned to `pairs`. `None`
            to reindex only `pairs` (returns `(new_pairs, None)`).
        cells: cell-level DataFrame, indexed by `cell_id`. Must have
            `cell_node_column`.
        cell_node_column: column on `cells` carrying the cell-tier network
            node ID.
        zones: optional zones DataFrame indexed by `zone_id`. Required iff
            `pairs.zones_to_zones` or `pairs.zones_to_regions` is set.
        zone_node_column: column on `zones` carrying the zone-tier network
            node ID. Required iff `zones` is given.
        regions: optional regions DataFrame indexed by `region_id`. Required
            iff `pairs.zones_to_regions` is set.
        region_node_column: column on `regions` carrying the region-tier
            network node ID. Required iff `regions` is given.

    Returns:
        `(new_pairs, new_odm)` — both `TieredODGeoPairs` (or
        `(new_pairs, None)` if `odm` was `None`). Tiers absent from `pairs`
        stay `None`.
    """
    if cell_node_column not in cells.columns:
        raise ValueError(f"`cells` is missing column {cell_node_column!r}.")
    if pairs.zones_to_zones is not None or pairs.zones_to_regions is not None:
        if zones is None or zone_node_column is None:
            raise ValueError(
                "`zones` and `zone_node_column` are required when `pairs` has "
                "zone-tier entries.")
        if zone_node_column not in zones.columns:
            raise ValueError(f"`zones` is missing column {zone_node_column!r}.")
    if pairs.zones_to_regions is not None:
        if regions is None or region_node_column is None:
            raise ValueError(
                "`regions` and `region_node_column` are required when `pairs` "
                "has region-tier entries.")
        if region_node_column not in regions.columns:
            raise ValueError(f"`regions` is missing column {region_node_column!r}.")

    cells_pairs, cells_odm = _reindex_tier(
        pairs.cells_to_cells,
        odm.cells_to_cells if odm is not None else None,
        cells, cells, cell_node_column, cell_node_column,
    )
    zones_pairs: dict | None = None
    zones_odm: dict | None = None
    if pairs.zones_to_zones is not None:
        assert zones is not None and zone_node_column is not None
        zones_pairs, zones_odm = _reindex_tier(
            pairs.zones_to_zones,
            odm.zones_to_zones if odm is not None else None,
            zones, zones, zone_node_column, zone_node_column,
        )
    z2r_pairs: dict | None = None
    z2r_odm: dict | None = None
    if pairs.zones_to_regions is not None:
        assert zones is not None and zone_node_column is not None
        assert regions is not None and region_node_column is not None
        z2r_pairs, z2r_odm = _reindex_tier(
            pairs.zones_to_regions,
            odm.zones_to_regions if odm is not None else None,
            zones, regions, zone_node_column, region_node_column,
        )

    new_pairs = TieredODGeoPairs(
        cells_to_cells=cells_pairs if cells_pairs is not None else {},
        zones_to_zones=zones_pairs,
        zones_to_regions=z2r_pairs,
    )
    if odm is None:
        return new_pairs, None
    new_odm = TieredODGeoPairs(
        cells_to_cells=cells_odm if cells_odm is not None else {},
        zones_to_zones=zones_odm,
        zones_to_regions=z2r_odm,
    )
    return new_pairs, new_odm


def dest_values_geo(
    column: str,
    pairs: TieredODGeoPairs,
    cells: pd.DataFrame,
    zones: pd.DataFrame | None = None,
    regions: pd.DataFrame | None = None,
) -> TieredODGeoPairs:
    """Look up `column` for every destination in a geo-keyed `pairs`, per tier.

    The geo-keyed twin of `dest_values`. Because destinations in
    `TieredODGeoPairs` are already individual geo-units (no node-level
    aggregation), each tier just looks up the value column on the matching
    DataFrame — no per-node summing. Structurally simpler and more honest
    than the node-keyed version: no implicit "many cells share a node, sum
    their values" assumption baked in.

    Args:
        column: name of the value column to look up. Must be present on
            `cells` (and on `zones`, `regions` for tiers that use those).
        pairs: geo-keyed destination-ID table (typically from
            `reindex_by_geo_unit`).
        cells: cell-level DataFrame indexed by `cell_id`.
        zones: optional zones DataFrame indexed by `zone_id`. Required iff
            `pairs.zones_to_zones` is set.
        regions: optional regions DataFrame indexed by `region_id`. Required
            iff `pairs.zones_to_regions` is set.

    Returns:
        `TieredODGeoPairs` of value arrays, paired position-wise with the
        input destination arrays.
    """
    if column not in cells.columns:
        raise ValueError(f"`cells` is missing column {column!r}.")
    if pairs.zones_to_zones is not None:
        if zones is None:
            raise ValueError("`zones` is required because `pairs.zones_to_zones` is set.")
        if column not in zones.columns:
            raise ValueError(f"`zones` is missing column {column!r}.")
    if pairs.zones_to_regions is not None:
        if regions is None:
            raise ValueError("`regions` is required because `pairs.zones_to_regions` is set.")
        if column not in regions.columns:
            raise ValueError(f"`regions` is missing column {column!r}.")

    def _lookup_for(d: dict, lookup: dict) -> dict:
        return {origin: np.array([lookup.get(dest, np.nan) for dest in dests])
                for origin, dests in d.items()}

    cells_lookup = cells[column].to_dict()
    zones_out = None
    if pairs.zones_to_zones is not None:
        assert zones is not None
        zones_out = _lookup_for(pairs.zones_to_zones, zones[column].to_dict())
    z_to_r_out = None
    if pairs.zones_to_regions is not None:
        assert regions is not None
        z_to_r_out = _lookup_for(pairs.zones_to_regions, regions[column].to_dict())
    return TieredODGeoPairs(
        cells_to_cells=_lookup_for(pairs.cells_to_cells, cells_lookup),
        zones_to_zones=zones_out,
        zones_to_regions=z_to_r_out,
    )


# ---------------------------------------------------------------------------
# Euclidean distances + summary
# ---------------------------------------------------------------------------

def _dists_for_dict(
    d: dict,
    nodes_xy: dict,
    dtype: np.dtype | type,
) -> dict:
    out: dict = {}
    for origin, dests in d.items():
        if origin not in nodes_xy:
            raise ValueError(f"Origin {origin!r} is not in `nodes`' xy map.")
        ox, oy = nodes_xy[origin]
        n = len(dests)
        if n == 0:
            out[origin] = np.empty(0, dtype=dtype)
            continue
        dx = np.empty(n, dtype=np.float64)
        dy = np.empty(n, dtype=np.float64)
        for i, dest in enumerate(dests):
            xy = nodes_xy.get(dest)
            if xy is None:
                raise ValueError(
                    f"Destination {dest!r} (origin {origin!r}) is not in `nodes`' xy map.")
            dx[i] = xy[0]
            dy[i] = xy[1]
        out[origin] = np.hypot(dx - ox, dy - oy).astype(dtype, copy=False)
    return out


# ---------------------------------------------------------------------------
# Cross-modal aggregation
# ---------------------------------------------------------------------------

def _aggregate_modes_tier(
    tier_arrays: list[np.ndarray],
    aggregator: str | Callable,
    scale: float,
) -> np.ndarray:
    """Apply the chosen aggregator across a stack of per-mode cost arrays.

    Stack shape: `(n_modes, n_dests)`. Returns a `(n_dests,)` array.
    """
    stacked = np.stack(tier_arrays, axis=0)
    if aggregator == 'min':
        # nanmin treats inf as a real value (unreachable mode → still the worst
        # finite value), but skips NaN (no observation for that mode).
        return np.nanmin(stacked, axis=0)
    if aggregator == 'logsum':
        # Log-sum-cost aggregation: -scale * ln Σ_m exp(-cost_m / scale).
        # Interpretation: cost_m is per-mode disutility (positive = bad).
        # Unreachable modes (inf) contribute exp(-inf) = 0; NaN is treated as
        # "no observation" and also contributes 0 (replace, don't propagate).
        exp_terms = np.exp(-stacked / scale)
        exp_terms = np.where(np.isnan(exp_terms), 0.0, exp_terms)
        sum_exp = exp_terms.sum(axis=0)
        # When all modes are unreachable, sum_exp = 0 → log = -inf → result =
        # +inf, which matches the "all unreachable" semantics from `min`.
        with np.errstate(divide='ignore'):
            return -scale * np.log(sum_exp)
    if callable(aggregator):
        return aggregator(stacked)
    raise ValueError(
        f"Unknown aggregator {aggregator!r}; expected 'min', 'logsum', or a callable.")


def aggregate_across_modes(
    odms: dict[str, tuple[TieredODGeoPairs, TieredODGeoPairs]],
    aggregator: str | Callable = 'min',
    *,
    scale: float = 1.0,
) -> tuple[TieredODGeoPairs, TieredODGeoPairs]:
    """Aggregate per-mode geo-keyed cost ODMs into a combined cost ODM.

    Enables cross-modal accessibility metrics where the aggregation across modes
    happens *inside* the accessibility computation rather than externally to it.
    Inputs must be `TieredODGeoPairs` — different modes typically live on
    different graphs (different node ID universes), but their geo-unit IDs are
    shared, so alignment is only possible in geo-unit space. Use
    `reindex_by_geo_unit` to lift per-mode node-keyed ODMs first.

    Each mode contributes `(pairs, costs)`:
      - `pairs`: geo-keyed `TieredODGeoPairs` of dest unit IDs.
      - `costs`: geo-keyed `TieredODGeoPairs` of cost values aligned to `pairs`.

    For each (origin, dest_unit) pair in the UNION across modes:
      - If a mode has the pair, use its cost.
      - If a mode is missing it (origin not in the mode, or dest not in the
        mode's per-origin dest array), fill with `+inf` ("unreachable by this
        mode").
    Then apply the aggregator across modes to produce a single combined cost.

    Three aggregator semantics:

    - **`'min'`** (default): per OD pair, take the minimum cost across modes.
      Use case: "how reachable is each destination under the fastest available
      mode?" Combine with `count_in_bins` for "destinations within X min by ANY
      mode"; with `gravity` or `nearest_k` for fastest-mode variants.

    - **`'logsum'`**: per OD pair, compute `-scale * ln Σ_m exp(-cost_m / scale)`
      — the discrete-choice log-sum-cost across modes. `scale` is the nest scale
      parameter (θ); defaults to 1.0, which gives the canonical `-ln Σ exp(-U)`
      when the per-mode cost is interpreted as utility (positive = disutility).
      Combine with `gravity(beta=1, family='exp')` to produce the canonical
      cross-modal logsum accessibility.

    - **Custom callable**: takes a `(n_modes, n_dests)` numpy array and returns
      a `(n_dests,)` array. Use for any aggregator not covered above (weighted
      average, max, etc.).

    Sign convention: per-mode costs should be positive disutilities (travel time,
    generalised cost, negated utility). For utility-as-benefit conventions
    (positive = attractive), negate before passing.

    Args:
        odms: `{mode_name -> (pairs, costs)}`. Must be non-empty. Both `pairs`
            and `costs` must be `TieredODGeoPairs`. Tier structure (which tiers
            are populated) must be consistent across modes.
        aggregator: `'min'`, `'logsum'`, or a callable.
        scale: nest scale parameter for `'logsum'` aggregation. Ignored for
            other aggregators.

    Returns:
        `(union_pairs, combined_costs)` — both `TieredODGeoPairs`. The
        `union_pairs` carries the per-origin union of dest IDs across modes
        (sorted canonically); `combined_costs` is aligned to it. NaN/inf are
        handled per-aggregator (`'min'` skips NaN, treats inf as finite-worst;
        `'logsum'` treats both as "mode contributes nothing to the sum").
    """
    if not odms:
        raise ValueError("`odms` must be non-empty.")
    mode_names = list(odms.keys())
    for m in mode_names:
        pairs_m, costs_m = odms[m]
        if not isinstance(pairs_m, TieredODGeoPairs):
            raise TypeError(
                f"Mode {m!r}: `pairs` must be a TieredODGeoPairs (got "
                f"{type(pairs_m).__name__}). Use `reindex_by_geo_unit` to lift "
                f"a node-keyed ODM into geo-unit space first.")
        if not isinstance(costs_m, TieredODGeoPairs):
            raise TypeError(
                f"Mode {m!r}: `costs` must be a TieredODGeoPairs (got "
                f"{type(costs_m).__name__}).")

    def _aggregate_tier(tier_name: str) -> tuple[dict | None, dict | None]:
        per_mode_pairs = [getattr(odms[m][0], tier_name) for m in mode_names]
        per_mode_costs = [getattr(odms[m][1], tier_name) for m in mode_names]
        if any(p is None for p in per_mode_pairs):
            if not all(p is None for p in per_mode_pairs):
                raise ValueError(
                    f"Tier {tier_name!r}: some modes populate it, others don't. "
                    f"Cross-modal aggregation requires consistent tier structure.")
            return None, None
        # Union of origin keys across modes.
        origin_union: set = set()
        for p in per_mode_pairs:
            origin_union.update(p.keys())
        out_pairs: dict = {}
        out_costs: dict = {}
        for origin in origin_union:
            # Union of dest IDs across modes for this origin.
            dest_union: set = set()
            for p in per_mode_pairs:
                if origin in p:
                    dest_union.update(p[origin].tolist())
            if not dest_union:
                continue
            dest_sorted = np.asarray(sorted(dest_union))
            # Build per-mode aligned cost arrays.
            aligned = []
            for p, c in zip(per_mode_pairs, per_mode_costs):
                if origin not in p:
                    aligned.append(np.full(len(dest_sorted), np.inf, dtype=float))
                    continue
                # Build a {dest_id -> cost} lookup for this mode's per-origin
                # entry, then look up each dest in the union.
                mode_dests = p[origin]
                mode_costs = np.asarray(c[origin], dtype=float)
                lookup = dict(zip(mode_dests.tolist(), mode_costs.tolist()))
                aligned.append(np.fromiter(
                    (lookup.get(d, np.inf) for d in dest_sorted),
                    dtype=float, count=len(dest_sorted)))
            out_pairs[origin] = dest_sorted
            out_costs[origin] = _aggregate_modes_tier(aligned, aggregator, scale)
        return out_pairs, out_costs

    c_pairs, c_costs = _aggregate_tier('cells_to_cells')
    z_pairs, z_costs = _aggregate_tier('zones_to_zones')
    r_pairs, r_costs = _aggregate_tier('zones_to_regions')

    union_pairs = TieredODGeoPairs(
        cells_to_cells=c_pairs if c_pairs is not None else {},
        zones_to_zones=z_pairs,
        zones_to_regions=r_pairs,
    )
    combined = TieredODGeoPairs(
        cells_to_cells=c_costs if c_costs is not None else {},
        zones_to_zones=z_costs,
        zones_to_regions=r_costs,
    )
    return union_pairs, combined


def get_euclidian_dists(
    nodes: pd.DataFrame | gpd.GeoDataFrame,
    pairs: TieredODPairs,
    dtype: np.dtype | type = np.float64,
) -> TieredODPairs:
    """Euclidean origin→destination distance for every pair in `pairs`, per tier.

    `nodes` must cover every node ID referenced anywhere in `pairs` (cell, zone,
    and region nodes). Distance is in the units of `nodes`' CRS.
    """
    nodes_xy = _build_node_xy_map(nodes)
    return TieredODNodePairs(
        cells_to_cells=_dists_for_dict(pairs.cells_to_cells, nodes_xy, dtype),
        zones_to_zones=(
            _dists_for_dict(pairs.zones_to_zones, nodes_xy, dtype)
            if pairs.zones_to_zones is not None else None
        ),
        zones_to_regions=(
            _dists_for_dict(pairs.zones_to_regions, nodes_xy, dtype)
            if pairs.zones_to_regions is not None else None
        ),
    )


