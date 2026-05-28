"""
Accessibility metrics computed against tiered OD tables.

Inputs are always:
    1. A cost `TieredODPairs` (subclass) — see "Key space" below.
    2. One or more pre-aggregated property weights, position-aligned with the
       cost ODM at each tier — a `dict[name -> TieredODPairs]`.
    3. A `cell_to_zone` mapping that gives each cell-tier origin its parent
       zone-tier key (for stitching the `zones_to_zones` far tier — the
       `cells_to_cells` and `cells_to_zones` tiers are already cell-keyed
       and don't need the indirection).

The per-origin stitching of `cells_to_cells / cells_to_zones / zones_to_zones`
tiers is done once, then reused across every (parameter × property) combination
— so adding more bins, decays, or k values is essentially free relative to a
single-parameter call.

**Key space — node-keyed vs geo-keyed.**

Two valid input shapes, distinguished by the costs/weights subclass:

- **`TieredODNodePairs`** (node-keyed): origins and dests are network node IDs.
  Each origin row in the output is a network node. Per-cell origin overhead
  cannot be applied here — the function returns per-NODE accessibilities. Build
  the `cell_to_zone` map from `od_pairs.build_cell_to_zone_node_map(cells,
  zones, node_column)` (cell-tier node → zone-tier node). Weights from
  `od_pairs.dest_values` (per-node sums).

- **`TieredODGeoPairs`** (geo-keyed): origins and dests are geo-unit IDs
  (cells_to_cells → cell_id; zones_to_zones → zone_id; etc.). Each origin row
  in the output is a cell. Per-cell origin overhead should be baked into the
  ODM *before* calling this function via `overhead.add_origin_cell_overhead`.
  Build the `cell_to_zone` map directly: `cells['zone_id'].to_dict()`. Weights
  from `od_pairs.dest_values_geo` (per-cell direct lookup, no implicit summing).

The same three functions (`cumulative_opportunities`, `gravity`, `nearest_k`) accept
either shape; output index name (`'node'` vs `'cell'`) reflects the input.

For cross-modal accessibility ("destinations within X min by ANY mode",
cross-modal logsum), combine per-mode `TieredODGeoPairs` cost ODMs with
`od_pairs.aggregate_across_modes` before passing here. Node-keyed cross-modal
is not supported — different modes live on different graphs.

For gravity in particular, the intrazonal-cost issue (cell-tier self-pairs
route at cost 0, which sends exp(0)=1 to maximum weight) is addressed by
calling `routing.floor_intrazonal_costs` on the cost ODM before passing here.

Provides:
    - `Bin` namedtuple — half-open `[lo, hi)` cost bin with a name.
    - `Decay` namedtuple — named callable for gravity-style cost decay.
    - `exp_decay`, `power_decay` — convenience constructors for common families.
    - `cumulative_opportunities` — sum each property's weights over destinations within each
      cost bin (cumulative-opportunity accessibility).
    - `gravity` — sum each property's weights weighted by f(cost), over all
      destinations, for one or more decay specs.
    - `nearest_k` — mean cost (or cost-at-k) to the k nearest weight-units,
      for one or more k values. Lower is better; canonical "mean travel time
      to the nearest k opportunities" formulation.
"""

from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODGeoPairs, TieredODPairs


def _require_cell_tier(costs: TieredODPairs) -> dict:
    """Narrow `costs.cells_to_cells` to a concrete `dict`, raising if None.

    Accessibility metrics require the cell-tier to be populated (origins are
    the cell-tier dict keys); raise a clear error if it isn't.
    """
    if costs.cells_to_cells is None:
        raise ValueError("`costs.cells_to_cells` is None; cell-tier is required.")
    return costs.cells_to_cells


class Bin(NamedTuple):
    """Half-open cost bin: `lo <= cost < hi`. `name` labels the output column.

    Bins should be mutually exclusive (the function does *not* check); a
    destination falling in multiple bins would be counted multiple times.
    """

    name: str
    lo: float
    hi: float


class Decay(NamedTuple):
    """Named cost-decay specification for `gravity`.

    `fn` is a vectorised callable mapping a cost array to a weight array; `name`
    labels the corresponding output column. Use `exp_decay` / `power_decay` for
    the common families, or construct directly with any user-defined callable.

    Multiple `Decay` specs can be passed to a single `gravity` call; the per-OD
    stitching is then amortised across all of them.
    """

    name: str
    fn: Callable[[np.ndarray], np.ndarray]


def exp_decay(name: str, beta: float) -> Decay:
    """Exponential decay: `f(c) = exp(-beta * c)`. `beta` > 0 for sensible decay."""
    return Decay(name, lambda c: np.exp(-beta * c))


def power_decay(name: str, beta: float) -> Decay:
    """Power-law decay: `f(c) = c ** (-beta)`. `beta` > 0; c = 0 yields `inf`,
    so callers should apply `routing.add_intrazonal_cost` first to replace
    self-pair cost-0 entries with a finite intrazonal cost.
    """
    return Decay(name, lambda c: np.power(c, -beta))


def _stitched_for(
    origin: int | str, costs: TieredODPairs, cell_to_zone_node: dict
) -> tuple[np.ndarray, slice, slice, slice]:
    """Per-origin stitched cost array + per-tier slices into it.

    The slices let callers stitch *other* per-tier arrays (e.g. each property's
    weights) into the same position-aligned 1-D layout without re-deriving the
    boundaries.

    Three tiers in order: cells_to_cells (cell-keyed, direct), cells_to_zones
    (cell-keyed, direct), zones_to_zones (zone-keyed, needs origin's parent
    zone). The returned slices have the same order.
    """
    cell_arr = _require_cell_tier(costs)[origin]
    c2z_arr = costs.cells_to_zones.get(origin) if costs.cells_to_zones is not None else None
    zone_node = cell_to_zone_node.get(origin)
    zone_arr = costs.zones_to_zones.get(zone_node) if costs.zones_to_zones is not None else None
    parts = [cell_arr]
    if c2z_arr is not None:
        parts.append(c2z_arr)
    if zone_arr is not None:
        parts.append(zone_arr)
    stitched = np.concatenate(parts) if len(parts) > 1 else cell_arr
    n_cell = len(cell_arr)
    n_c2z = len(c2z_arr) if c2z_arr is not None else 0
    n_zone = len(zone_arr) if zone_arr is not None else 0
    return (
        stitched,
        slice(0, n_cell),
        slice(n_cell, n_cell + n_c2z),
        slice(n_cell + n_c2z, n_cell + n_c2z + n_zone),
    )


def _stitched_weights(
    origin: int | str,
    zone_node: int | str | None,
    weights: TieredODPairs,
    n_cell: int,
    n_c2z: int,
    n_zone: int,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Stitch a single property's three-tier value arrays for one origin.

    Tiers that are `None` (or where the origin / zone has no entry) contribute
    zeros so the result is positionally aligned with the cost stitching from
    `_stitched_for`.
    """
    total = n_cell + n_c2z + n_zone
    out = np.zeros(total, dtype=dtype)
    cell_w = weights.cells_to_cells.get(origin) if weights.cells_to_cells is not None else None
    if cell_w is not None and n_cell:
        out[:n_cell] = cell_w
    if n_c2z and weights.cells_to_zones is not None:
        c2z_w = weights.cells_to_zones.get(origin)
        if c2z_w is not None:
            out[n_cell : n_cell + n_c2z] = c2z_w
    if n_zone and weights.zones_to_zones is not None:
        zw = weights.zones_to_zones.get(zone_node)
        if zw is not None:
            out[n_cell + n_c2z :] = zw
    return out


def _origin_index_name(costs: TieredODPairs) -> str:
    """Output-DataFrame index name based on the input ODM's key space."""
    return "cell" if isinstance(costs, TieredODGeoPairs) else "node"


def _sniff_dtype(costs: TieredODPairs) -> np.dtype:
    """Return the dtype of the first non-empty per-tier cost array.

    Used to make accessibility output dtype follow the input costs dtype
    (FP32 by default, FP64 if the caller opted in upstream) without an
    explicit kwarg on every accessibility function.
    """
    for tier in (costs.cells_to_cells, costs.cells_to_zones, costs.zones_to_zones):
        if tier is None:
            continue
        for arr in tier.values():
            return np.asarray(arr).dtype
    return np.dtype(np.float32)


def cumulative_opportunities(
    costs: TieredODPairs,
    weights: dict[str, TieredODPairs],
    cell_to_zone: dict,
    bins: list[Bin],
) -> pd.DataFrame:
    """Sum each property's weights over destinations whose cost falls in each bin.

    Args:
        costs: tiered travel costs. Subclass determines output indexing:
            `TieredODNodePairs` → per-node output; `TieredODGeoPairs` → per-cell
            output. Non-finite entries (`np.inf`, `np.nan`) won't match any
            finite bin and are silently dropped.
        weights: `{property_name -> TieredODPairs}`, position-aligned with
            `costs` per tier. Must share the costs' key space (node-keyed
            weights for node-keyed costs; geo-keyed for geo-keyed). Build via
            `od_pairs.dest_values` (node-keyed) or `od_pairs.dest_values_geo`
            (geo-keyed). Missing origins / tiers contribute zeros, not errors.
        cell_to_zone: `{cell_tier_key -> zone_tier_key}` map for tier
            stitching. Build from `od_pairs.build_cell_to_zone_node_map`
            (node-keyed: cell_node → zone_node) or directly from
            `cells['zone_id'].to_dict()` (geo-keyed: cell_id → zone_id).
        bins: half-open `[lo, hi)` cost bins. Should be mutually exclusive
            (not checked).

    Returns:
        DataFrame indexed by origin key with `(bin_name, property_name)`
        MultiIndex on columns. Order: bins outer, properties inner. Dtype
        follows the input `costs` ODM (FP32 by default, FP64 if the caller
        opted in upstream).

    Per-cell overhead: for `TieredODGeoPairs` inputs, bake per-cell origin
    overhead into the cost ODM upfront via `overhead.add_origin_cell_overhead`.
    """
    prop_names = list(weights.keys())
    origins = list(_require_cell_tier(costs).keys())
    columns = pd.MultiIndex.from_product(
        [[b.name for b in bins], prop_names], names=["bin", "property"]
    )
    dtype = _sniff_dtype(costs)
    out = np.zeros((len(origins), len(bins) * len(prop_names)), dtype=dtype)

    for i, origin in enumerate(origins):
        stitched_cost, cell_sl, c2z_sl, zone_sl = _stitched_for(origin, costs, cell_to_zone)
        n_cell = cell_sl.stop - cell_sl.start
        n_c2z = c2z_sl.stop - c2z_sl.start
        n_zone = zone_sl.stop - zone_sl.start
        zone_key = cell_to_zone.get(origin)

        prop_weights = np.empty((len(prop_names), len(stitched_cost)), dtype=dtype)
        for p, name in enumerate(prop_names):
            prop_weights[p] = _stitched_weights(
                origin, zone_key, weights[name], n_cell, n_c2z, n_zone, dtype=dtype
            )

        for b, bin_ in enumerate(bins):
            mask = (stitched_cost >= bin_.lo) & (stitched_cost < bin_.hi)
            if not mask.any():
                continue
            out[i, b * len(prop_names) : (b + 1) * len(prop_names)] = prop_weights[:, mask].sum(
                axis=1
            )

    return pd.DataFrame(
        out,
        index=pd.Index(origins, name=_origin_index_name(costs)),
        columns=columns,
    )


def gravity(
    costs: TieredODPairs,
    weights: dict[str, TieredODPairs],
    cell_to_zone: dict,
    decays: Decay | list[Decay],
) -> pd.DataFrame:
    """Gravity-based accessibility: sum each property's weights, weighted by
    `f(cost)`, over all destinations — for one or more decay specs in a single
    call.

    For each origin `i`, each property `w`, and each decay spec `f`::

        A_i^{f,w} = Σ_j w_j · f(cost_ij)

    Multiple decay specs share the per-OD stitching, so calling with a list of
    `Decay` specs is much cheaper than calling once per spec — useful for sensitivity
    analyses across decay-coefficient ranges.

    For utility-based accessibility, pass the per-OD utility values as the cost
    ODM and an exponential decay with the desired scale (β=1 gives the
    standard `Σ_j w_j · exp(-U_ij)` form, on which logsum accessibility is
    `ln(...)` of the same sum).

    Args:
        costs, weights, cell_to_zone: see `cumulative_opportunities`.
        decays: a single `Decay` or list of `Decay` specs. Output columns are
            MultiIndex `(decay_name, property_name)` with decay names outer.

    Returns:
        DataFrame indexed by origin key with MultiIndex columns
        `(decay, property)`. Dtype follows the input `costs` ODM
        (FP32 by default, FP64 if the caller opted in upstream).
    """
    if isinstance(decays, Decay):
        decays = [decays]
    if not decays:
        raise ValueError("`decays` must be a non-empty list of `Decay` specs.")

    prop_names = list(weights.keys())
    decay_names = [d.name for d in decays]
    origins = list(_require_cell_tier(costs).keys())
    columns = pd.MultiIndex.from_product([decay_names, prop_names], names=["decay", "property"])
    n_props = len(prop_names)
    dtype = _sniff_dtype(costs)
    out = np.zeros((len(origins), len(decays) * n_props), dtype=dtype)

    for i, origin in enumerate(origins):
        stitched_cost, cell_sl, c2z_sl, zone_sl = _stitched_for(origin, costs, cell_to_zone)
        n_cell = cell_sl.stop - cell_sl.start
        n_c2z = c2z_sl.stop - c2z_sl.start
        n_zone = zone_sl.stop - zone_sl.start
        zone_key = cell_to_zone.get(origin)

        prop_weights = np.empty((n_props, len(stitched_cost)), dtype=dtype)
        for p, name in enumerate(prop_names):
            prop_weights[p] = _stitched_weights(
                origin, zone_key, weights[name], n_cell, n_c2z, n_zone, dtype=dtype
            )

        finite_mask = np.isfinite(stitched_cost)
        if not finite_mask.any():
            continue
        cost_finite = stitched_cost[finite_mask]
        w_finite = prop_weights[:, finite_mask]
        for d, decay in enumerate(decays):
            decayed = decay.fn(cost_finite)
            # Defensive: a decay that produces non-finite at finite cost
            # (e.g. power with c=0 if intrazonal cost wasn't applied) should
            # not silently corrupt the sum. Drop those entries.
            if not np.all(np.isfinite(decayed)):
                decay_finite = np.isfinite(decayed)
                decayed = decayed[decay_finite]
                w_used = w_finite[:, decay_finite]
            else:
                w_used = w_finite
            out[i, d * n_props : (d + 1) * n_props] = (w_used * decayed).sum(axis=1)

    return pd.DataFrame(
        out,
        index=pd.Index(origins, name=_origin_index_name(costs)),
        columns=columns,
    )


def nearest_k(
    costs: TieredODPairs,
    weights: dict[str, TieredODPairs],
    cell_to_zone: dict,
    ks: int | float | list[int | float],
    *,
    aggregator: str = "cost_mean",
) -> pd.DataFrame:
    """Nearest-`k` accessibility: cost (mean, or at-`k`) over the `k` nearest
    weight-units.

    Each destination is treated as carrying `weight_j` opportunities at cost
    `cost_ij`. Destinations are sorted ascending by cost; the first `k`
    weight-units (with fractional contribution at the boundary) define the
    "nearest `k` opportunities". The aggregator decides what to return:

    - **`'cost_mean'`** (default): the mean cost over the first `k` weight-units,
        ``A_i^{k,w} = (Σ cost_j · weight_j contributed, fractional at the boundary) / k``.
        The canonical "mean travel cost to the nearest `k` opportunities"
        formulation — directly comparable across `k` values (k=3 and k=5 are on
        the same scale, in cost units).

    - **`'cost_at_k'`**: the cost of the `k`-th weight-unit — i.e., the cost
        at which the cumulative weight first reaches `k`. Answers
        "how far is the `k`-th nearest opportunity?".

    Both aggregators return a value in the same units as `costs`, with **lower
    values = better accessibility**. NaN is returned where the total available
    (finite-cost, positive-weight) opportunities at an origin is less than `k`
    — i.e., the `k`-th opportunity is unreachable in finite cost.

    Multiple `k` values share the per-OD sort, so a multi-`k` call is much
    cheaper than `k` individual calls.

    Args:
        costs, weights, cell_to_zone: see `cumulative_opportunities`.
        ks: a single `k` or list of `k`s; positive values, integer or float.
            Output columns are MultiIndex `(k, property_name)` with `k` outer.
        aggregator: `'cost_mean'` (default) or `'cost_at_k'`.

    Returns:
        DataFrame indexed by origin key with MultiIndex columns `(k, property)`.
        Dtype follows the input `costs` ODM (FP32 by default, FP64 if the
        caller opted in upstream). NaN where the `k`-th opportunity is
        unreachable.
    """
    if aggregator not in ("cost_mean", "cost_at_k"):
        raise ValueError(f"Unknown aggregator {aggregator!r}; expected 'cost_mean' or 'cost_at_k'.")
    if isinstance(ks, (int, float)):
        ks = [ks]
    if not ks:
        raise ValueError("`ks` must be a non-empty list of positive values.")
    if any(k <= 0 for k in ks):
        raise ValueError(f"All `k` values must be > 0; got {ks!r}.")

    prop_names = list(weights.keys())
    origins = list(_require_cell_tier(costs).keys())
    columns = pd.MultiIndex.from_product([ks, prop_names], names=["k", "property"])
    n_props = len(prop_names)
    dtype = _sniff_dtype(costs)
    out = np.full((len(origins), len(ks) * n_props), np.nan, dtype=dtype)
    ks_arr = np.asarray(ks, dtype=dtype)

    for i, origin in enumerate(origins):
        stitched_cost, cell_sl, c2z_sl, zone_sl = _stitched_for(origin, costs, cell_to_zone)
        n_cell = cell_sl.stop - cell_sl.start
        n_c2z = c2z_sl.stop - c2z_sl.start
        n_zone = zone_sl.stop - zone_sl.start
        zone_key = cell_to_zone.get(origin)

        prop_weights = np.empty((n_props, len(stitched_cost)), dtype=dtype)
        for p, name in enumerate(prop_names):
            prop_weights[p] = _stitched_weights(
                origin, zone_key, weights[name], n_cell, n_c2z, n_zone, dtype=dtype
            )

        finite_mask = np.isfinite(stitched_cost)
        if not finite_mask.any():
            continue
        costs_f = stitched_cost[finite_mask]
        sort_idx = np.argsort(costs_f)
        sorted_costs = costs_f[sort_idx]
        for p in range(n_props):
            w = prop_weights[p][finite_mask][sort_idx]
            # Non-finite or non-positive weights contribute nothing.
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            cum_w = np.cumsum(w)
            if cum_w.size == 0 or cum_w[-1] == 0.0:
                continue
            cum_cw = np.cumsum(sorted_costs * w) if aggregator == "cost_mean" else None
            total = cum_w[-1]
            for ki, k in enumerate(ks_arr):
                if total < k:
                    continue
                idx = int(np.searchsorted(cum_w, k, side="left"))
                if aggregator == "cost_at_k":
                    out[i, ki * n_props + p] = sorted_costs[idx]
                else:  # cost_mean
                    assert cum_cw is not None  # guaranteed by aggregator branch above
                    if idx == 0:
                        out[i, ki * n_props + p] = sorted_costs[0]
                    else:
                        full_cw = cum_cw[idx - 1]
                        partial = sorted_costs[idx] * (k - cum_w[idx - 1])
                        out[i, ki * n_props + p] = (full_cw + partial) / k

    return pd.DataFrame(
        out,
        index=pd.Index(origins, name=_origin_index_name(costs)),
        columns=columns,
    )


def flatten_index(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a 2-level column MultiIndex into single strings joined by `__`.

    Convenience for the accessibility outputs (which carry `(bin, property)`
    or `(decay, property)` MultiIndex columns) when downstream code prefers
    flat single-string column names (e.g. for CSV export). Mutates in place
    and also returns `df`.
    """
    df.columns = ["__".join(col).strip() for col in df.columns.values]
    return df
