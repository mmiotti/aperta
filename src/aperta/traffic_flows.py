"""Lightweight one-shot traffic-flow estimation via sampled betweenness centrality.

Estimates daily per-edge traffic volumes (interpretable as AADT once
calibrated) by simulating a quick three-step travel demand model:
trip generation (origin sampling weighted by population), trip distribution
(per-origin destination sampling weighted by a cost-decay function and
per-destination attractiveness), and route assignment (shortest-path routing
on the current edge weights, accumulating per-edge counts). Outputs can be
calibrated against ground-truth counter data via the helpers in
`aperta.calibration`.

**Scope and limitations.** This is a one-shot estimation pass: the routing
step uses the input edge weights without iterating toward congestion
equilibrium. It is intended for users who (1) want a per-edge traffic-flow
estimate to feed into travel-time calibration or as an accessibility feature,
and (2) do not already have detailed outputs from a full traffic-assignment
model (which would be a more rigorous alternative and could be plugged in
directly). The library reuses aperta's existing infrastructure — tiered OD
matrices, edge-weight calibration, scipy routing backend — to keep the
estimation cheap and consistent with the rest of the pipeline; it does not
aim to replace a dedicated traffic-assignment tool. An iterative
congestion-aware variant is theoretically possible as a future extension.

This module supplies the sampling primitive `nested_node_sample`. The
routing + per-edge accumulation itself lives in
`network_processing.get_nested_edge_betweenness`. A simpler alternative
for small study areas — radius-limited Brandes betweenness without
explicit OD sampling — also lives in `network_processing`. Downstream
callers apply their own normalisation of the raw sampled-betweenness
counts (e.g. scaling to an expected vehicle-kilometres total).
"""

from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

from aperta.od_pairs import TieredODPairs


@njit(cache=True)
def _weighted_sample_indices(weights: np.ndarray, rvals: np.ndarray) -> np.ndarray:
    """Sample `len(rvals)` indices into `weights` with probability ∝ weights.

    Equivalent to `np.random.choice(len(weights), len(rvals), replace=True, p=weights/weights.sum())`
    but JITted (cumsum + searchsorted) — fast for repeated calls with small-to-medium
    weight arrays. `rvals` is pre-drawn uniforms in [0, 1), so the caller controls
    the RNG (we don't use numba's random state).
    """
    cumsum = np.cumsum(weights)
    total = cumsum[-1]
    n = len(rvals)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = np.searchsorted(cumsum, rvals[i] * total)
    return out


def _zone_tier_dests_and_scores(
    pairs: TieredODPairs,
    weights: TieredODPairs,
    costs: TieredODPairs,
    cost_to_weight: Callable,
    mask: TieredODPairs | None = None,
) -> dict:
    """Pre-compute per-zone shared (zone_dests, zone_score) with optional mask
    applied. Done once per zone — reused across every cell in that zone during
    sampling, amortizing both the `cost_to_weight` call and the mask-filter
    step.

    Empty arrays when a zone has no zone-tier dests.

    Phase A note: previously also pre-computed region-tier dests/scores. The
    region tier has been replaced by `cells_to_zones` (cell-keyed origin,
    zone-keyed dest), which can't be amortised per-zone the same way — each
    cell has its own dest set. Re-integration of the cells_to_zones tier into
    `nested_node_sample` is a Phase B / D follow-up.
    """
    z2z_d = pairs.zones_to_zones or {}
    z2z_w = weights.zones_to_zones or {}
    z2z_c = costs.zones_to_zones or {}
    z2z_m = (mask.zones_to_zones if mask is not None else None) or {}
    empty_dest = np.empty(0, dtype=object)
    empty_score = np.empty(0)

    out: dict = {}
    for zn in z2z_d:
        if zn in z2z_d and len(z2z_d[zn]):
            zd, zw, zc = z2z_d[zn], z2z_w[zn], z2z_c[zn]
            if zn in z2z_m:
                m = z2z_m[zn]
                zd, zw, zc = zd[m], zw[m], zc[m]
            zone_dests = zd
            zone_score = zw * cost_to_weight(zc)
        else:
            zone_dests, zone_score = empty_dest, empty_score
        out[zn] = (zone_dests, zone_score)
    return out


def nested_node_sample(
    pairs: TieredODPairs,
    weights: TieredODPairs,
    costs: TieredODPairs,
    *,
    cell_to_zone_node: dict,
    orig_weights: np.ndarray | pd.Series | None,
    cost_to_weight: Callable,
    n_orig: int,
    n_dest: int,
    random_state: np.random.RandomState,
    mask: TieredODPairs | None = None,
    chosen: np.ndarray | None = None,
) -> dict:
    """Sample `n_dest` destinations for `n_orig` weighted-sampled origin cells,
    integrating all three tiers (cell, middle, far) into one combined pool.

    Per origin cell, the tier dest arrays are concatenated on the fly into one
    combined dest pool with per-pair scores `weight * cost_to_weight(cost)`.
    Sampling is then a single `np.random.choice`-equivalent (JITted) over the
    pool. Peak memory is bounded by the largest single per-origin concatenation,
    not by `n_orig × total_dests`.

    The per-zone shared scores (far tier) are computed once per zone (not per
    cell), so the `cost_to_weight` call is amortized across all cells in the
    zone. The middle tier (`cells_to_zones`) is keyed per cell — same dest
    *zones* across cells in a zone, but different per-cell costs — so it can't
    amortise the same way, but the per-cell cost is what makes the score
    correct.

    Args:
        pairs: destination IDs per tier.
        weights: destination weights per tier (e.g. populations), same shape as
            `pairs`. Typically the output of `od_pairs.lookup_dest_column_node`.
        costs: per-pair costs (e.g. line distances), same shape as `pairs`.
            Typically the output of `od_pairs.get_euclidean_dists`.
        cell_to_zone_node: `{cell_node -> zone_node}` mapping; build via
            `od_pairs.build_cell_to_zone_node_map`.
        orig_weights: per-origin sampling weights, aligned position-wise with
            `list(pairs.cells_to_cells.keys())`. Required when `chosen` is
            None; ignored when `chosen` is provided.
        cost_to_weight: monotone-decreasing function mapping a cost (e.g. distance
            in metres) to a per-pair weight. Vectorized — receives a 1-D array.
        n_orig, n_dest: number of origins to sample; number of destinations
            sampled PER origin-pick. Origin sampling is with replacement
            (popular origins can appear multiple times in the underlying
            `random_state.choice`); each duplicate pick generates its own
            batch of `n_dest` destinations, all i.i.d. from the same
            per-origin score distribution. So an origin picked `k` times
            ends up with a length-`k × n_dest` destination array — and
            contributes `k×` the flow downstream when
            `get_nested_edge_betweenness` walks predecessors, while still
            running only ONE Dijkstra from that origin (no wasted routing
            work; only the destination set grows).
            Total OD pairs sampled is exactly `n_orig × n_dest`
            regardless of duplicate distribution — useful for AADT scaling
            (denominator = `n_orig × n_dest`, no dedup correction needed).
            When `chosen` is provided, `n_orig` is ignored (sample size
            comes from `len(chosen)`); `n_dest` still controls
            destinations-per-pick.
        random_state: numpy RandomState; the only source of randomness.
        mask: optional boolean `TieredODPairs` (build via `od_pairs.make_mask`).
            Destinations where the mask is `False` are removed from the sampling
            pool. Missing origins or missing tiers in the mask are treated as
            "no filter" for that origin / tier.
        chosen: optional pre-sampled origin array (with replacement, so
            duplicates carry their `n_picks` weight). When provided, the
            internal `random_state.choice(origins, n_orig, True, p)` is
            skipped; `orig_weights` and `n_orig` are ignored. Useful when
            the caller pre-samples origins externally to restrict the
            upstream `tiered_path_costs` work to only origins that will
            actually contribute — every entry of `chosen` must be a key
            in `pairs.cells_to_cells`.

    Returns: `{origin_cell_node -> np.ndarray[dest_node]}` where each value
        array has length `n_picks × n_dest` (= `n_dest` for origins picked
        once, longer for origins picked multiple times by the
        with-replacement origin sampling).
    """
    if pairs.cells_to_cells is None:
        raise ValueError("`pairs.cells_to_cells` is None; cell-tier is required.")
    if costs.cells_to_cells is None or weights.cells_to_cells is None:
        raise ValueError("`costs` and `weights` must both have a populated cell-tier.")
    cell_pairs = pairs.cells_to_cells
    cell_costs_dict = costs.cells_to_cells
    cell_weights_dict = weights.cells_to_cells

    # Origin sampling: either internal (from `orig_weights`) or
    # caller-supplied (`chosen`). The caller-supplied path is the
    # standard way to restrict upstream `tiered_path_costs` work to
    # origins that will actually be sampled — by pre-selecting origins
    # and passing them here, callers can skip routing the long tail of
    # rarely-picked cells. Exactly one of the two paths must be used.
    if chosen is None and orig_weights is None:
        raise ValueError(
            "nested_node_sample: provide either `orig_weights` (for internal "
            "sampling) or `chosen` (for pre-sampled origins)."
        )
    if chosen is not None and orig_weights is not None:
        raise ValueError(
            "nested_node_sample: `orig_weights` and `chosen` are mutually "
            "exclusive — pass one or the other."
        )
    if chosen is None:
        origins = np.asarray(list(cell_pairs.keys()))
        p = np.asarray(orig_weights, dtype=float)
        p = p / p.sum()
        chosen = random_state.choice(origins, n_orig, True, p)
    else:
        chosen = np.asarray(chosen)
        # Validate: every pre-sampled origin must be a key in
        # `pairs.cells_to_cells`. Since `get_pairs` populates an entry
        # (possibly empty) for every valid origin, this catches genuine
        # user errors — pre-sampled origins not covered by
        # `get_pairs(orig_cells=...)`, or NaN values that slipped into
        # `chosen` from cells with unsnapped node IDs.
        missing = set(chosen.tolist()) - set(cell_pairs.keys())
        if missing:
            n_nan = sum(1 for x in missing if isinstance(x, float) and np.isnan(x))
            if n_nan:
                raise ValueError(
                    f"nested_node_sample: {len(missing)} entries in `chosen` "
                    f"are not present in `pairs.cells_to_cells`, of which "
                    f"{n_nan} are NaN. Pre-sampling from a cells DataFrame "
                    f"with NaN `node_id` values produces NaN draws — filter "
                    f"`cells = cells[cells['<node_column>'].notna()]` (and "
                    f"any other NaN-bearing columns used as weights or "
                    f"zone identifiers) before building `chosen`."
                )
            raise ValueError(
                f"nested_node_sample: {len(missing)} entries in `chosen` are "
                f"not present in `pairs.cells_to_cells`. Did you restrict "
                f"`get_pairs(orig_cells=...)` to cover the pre-sampled set? "
                f"Example missing: {sorted(missing)[:3]}."
            )

    # Pre-compute per-zone shared dest arrays + scores for the FAR tier
    # (zones_to_zones). Reused across every cell in that zone during sampling.
    z_combo = _zone_tier_dests_and_scores(pairs, weights, costs, cost_to_weight, mask)
    cell_mask_dict = (mask.cells_to_cells if mask is not None else None) or {}
    # Middle tier (cells_to_zones) is cell-keyed; pre-bind the dicts (or empty
    # fallbacks) so the inner loop doesn't keep checking for None.
    c2z_pairs = pairs.cells_to_zones or {}
    c2z_costs = costs.cells_to_zones or {}
    c2z_weights = weights.cells_to_zones or {}
    c2z_mask_dict = (mask.cells_to_zones if mask is not None else None) or {}
    empty_dest = np.empty(0, dtype=object)
    empty_score = np.empty(0)

    # Group sampled origins by zone — shared work (far tier) is done once per
    # zone-group. Count occurrences via `Counter` so duplicate picks in
    # `chosen` (with-replacement origin sampling lets popular origins appear
    # multiple times) generate proportionally more destination samples
    # downstream — each unique origin still runs one Dijkstra in the
    # consumer (`get_nested_edge_betweenness`), but its dest array gets
    # `n_picks × n_dest` entries instead of `n_dest`, so each pick
    # contributes its share of routing-effort to the flow estimate.
    chosen_counts: dict = Counter(chosen.tolist())
    chosen_by_zone: dict = defaultdict(list)
    for c in chosen_counts:
        chosen_by_zone[cell_to_zone_node.get(c)].append(c)

    out: dict = {}
    for zone_node, cells_here in chosen_by_zone.items():
        zone_dests, zone_score = z_combo.get(
            zone_node,
            (empty_dest, empty_score),
        )
        for c in cells_here:
            # Cell tier (cells_to_cells): per-cell origin + per-cell dest.
            # `get_pairs` populates an entry for every valid origin
            # (possibly empty array for isolated cells), so direct
            # lookups are safe — see the upfront validation against
            # `cell_pairs.keys()` above.
            cell_dests = cell_pairs[c]
            cell_costs = cell_costs_dict[c]
            cell_weights = cell_weights_dict[c]
            if c in cell_mask_dict:
                m = cell_mask_dict[c]
                cell_dests, cell_costs, cell_weights = cell_dests[m], cell_costs[m], cell_weights[m]
            cell_score = cell_weights * cost_to_weight(cell_costs)

            # Middle tier (cells_to_zones): per-cell origin → zone-node dest.
            # Cells in the same zone share dest IDs but have distinct per-cell
            # costs, so the score has to be re-computed per cell.
            if c in c2z_pairs:
                cz_dests = c2z_pairs[c]
                cz_costs_arr = c2z_costs[c]
                cz_weights_arr = c2z_weights[c]
                if c in c2z_mask_dict:
                    cm = c2z_mask_dict[c]
                    cz_dests = cz_dests[cm]
                    cz_costs_arr = cz_costs_arr[cm]
                    cz_weights_arr = cz_weights_arr[cm]
                cz_score = cz_weights_arr * cost_to_weight(cz_costs_arr)
            else:
                cz_dests, cz_score = empty_dest, empty_score

            all_dests = np.concatenate([cell_dests, cz_dests, zone_dests])
            all_score = np.concatenate([cell_score, cz_score, zone_score])
            # An origin with NO destinations across any tier (truly
            # isolated cell — no in-radius cells, no zones in [r_cells,
            # r_zones]) can't generate flow. Skip rather than crash on
            # the empty `cumsum` inside `_weighted_sample_indices`.
            if all_score.size == 0 or all_score.sum() <= 0:
                continue
            # Sample `n_picks × n_dest` destinations: this origin was
            # drawn `n_picks` times by the with-replacement origin
            # sampling at the top, so it generates that many "trips"-
            # worth of destinations. All drawn i.i.d. from the same
            # per-origin score distribution — see `n_dest_total` docstring.
            n_picks = chosen_counts[c]
            rvals = random_state.random(n_dest * n_picks)
            indices = _weighted_sample_indices(all_score, rvals)
            out[c] = all_dests[indices]
    return out


# ---------------------------------------------------------------------------
# Bin-adjusted destination weights — fix for the cost-weighted sampling bias
# at sparse-periphery origins. See memory `aperta-traffic-flow-sampling-bias-fix`
# for the design and motivation.
# ---------------------------------------------------------------------------


def percentile_bin_edges(
    survey_costs: np.ndarray | pd.Series,
    n_bins: int = 20,
) -> np.ndarray:
    """Equal-probability cost-bin edges from observed trip-cost data.

    Returns ``n_bins + 1`` edges such that each bin contains roughly ``1 / n_bins``
    of the survey data by count. Suitable as the ``bin_edges`` input to
    `bin_adjusted_dest_weights` for sampling that targets the empirical cost
    distribution non-parametrically (no need to fit a log-normal or similar).

    Args:
        survey_costs: observed trip costs (e.g. observed travel times). NaNs
            and non-finite values are dropped before percentile estimation.
        n_bins: number of equal-probability bins. Default 20 balances
            granularity vs. per-origin sample-budget headroom; 10–30 are
            reasonable choices.

    Returns:
        Sorted 1-D array of length ``n_bins + 1`` giving bin edges.
    """
    arr = np.asarray(survey_costs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("`survey_costs` is empty after dropping non-finite values.")
    return np.percentile(arr, np.linspace(0.0, 100.0, n_bins + 1))


def bin_adjusted_dest_weights(
    pairs: TieredODPairs,
    costs: TieredODPairs,
    dest_weights: TieredODPairs,
    bin_edges: np.ndarray,
    *,
    renormalize_per_origin: bool = True,
) -> TieredODPairs:
    """Per-origin per-bin reweight of destination weights, fixing the
    cost-weighted sampling bias at sparse-periphery origins.

    For each origin and each cost bin, the bin's target probability mass
    (``1 / n_bins``) is divided among the destinations that fall in that bin
    in proportion to their existing weight ``W(D)``. Bins with no destinations
    contribute nothing. The result is a per-origin adjusted weight array of
    the same shape as ``dest_weights``, which can be passed to
    `nested_node_sample` (or any weighted sampler) in place of the raw
    weights — replacing the ``cost_to_weight`` callable entirely.

    Destinations whose cost falls outside ``[bin_edges[0], bin_edges[-1]]``
    receive zero weight (treated as too rare to be informative).

    Args:
        pairs: destination IDs per tier (any of the three may be ``None``).
        costs: per-pair costs, same shape as ``pairs`` (e.g. travel times).
        dest_weights: base destination weights ``W(D)`` per pair, same shape
            as ``pairs`` (e.g. populations, employment counts).
        bin_edges: ``n_bins + 1`` sorted values, typically from
            `percentile_bin_edges` applied to a travel-survey cost column.
        renormalize_per_origin: when ``True`` (default), the adjusted weights
            for each origin are normalised to sum to 1, so each origin has
            the same total sampling weight regardless of how many cost bins
            its destinations populate. When ``False``, sparse-periphery
            origins end up with a smaller total weight (the empty bins'
            target mass is not redistributed) — this naturally reduces their
            effective trip count, useful when the bin adjustment is the only
            mechanism reducing trips from sparse origins. The default
            ``True`` matches the recommended decoupling: bin-adjustment fixes
            the cost distribution only; trip-generation count stays
            controlled separately at the ``orig_weights`` stage.

    Returns:
        Same ``TieredODPairs`` subclass as ``pairs`` with per-origin adjusted
        weight arrays. Origins whose destinations are entirely out of range
        (or whose ``dest_weights`` sum to zero) receive an all-zero array.
    """
    bin_edges = np.asarray(bin_edges, dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError(
            f"`bin_edges` must be a 1-D array of length >= 2; got shape {bin_edges.shape}."
        )
    if np.any(np.diff(bin_edges) < 0):
        raise ValueError("`bin_edges` must be non-decreasing.")
    n_bins = bin_edges.size - 1
    target_mass_per_bin = 1.0 / n_bins

    def _adjust_tier(
        pair_tier: dict | None,
        cost_tier: dict | None,
        weight_tier: dict | None,
    ) -> dict | None:
        if pair_tier is None or cost_tier is None or weight_tier is None:
            return None
        out: dict = {}
        for origin, dest_arr in pair_tier.items():
            cost_arr = np.asarray(cost_tier[origin], dtype=float)
            w_arr = np.asarray(weight_tier[origin], dtype=float)
            adjusted = np.zeros_like(w_arr, dtype=float)
            # Bin assignment: np.digitize(x, edges) returns 0 for x < edges[0],
            # n_bins+1 for x >= edges[-1], and i in 1..n_bins otherwise.
            # Subtract 1 to get 0..n_bins-1 for in-range, -1/n_bins for out.
            bin_idx = np.digitize(cost_arr, bin_edges) - 1
            in_range = (bin_idx >= 0) & (bin_idx < n_bins) & np.isfinite(cost_arr)
            if not in_range.any():
                out[origin] = adjusted
                continue
            # Per-bin total weight (vectorised with bincount).
            bin_idx_safe = np.where(in_range, bin_idx, 0)
            bin_sums = np.bincount(bin_idx_safe, weights=w_arr * in_range, minlength=n_bins)
            # Per-bin scaling factor: target_mass / available_mass, 0 for empty bins.
            scale = np.zeros(n_bins, dtype=float)
            populated = bin_sums > 0
            scale[populated] = target_mass_per_bin / bin_sums[populated]
            adjusted = w_arr * scale[bin_idx_safe] * in_range
            if renormalize_per_origin:
                total = adjusted.sum()
                if total > 0:
                    adjusted = adjusted / total
            out[origin] = adjusted
        return out

    return type(pairs)(
        cells_to_cells=_adjust_tier(
            pairs.cells_to_cells, costs.cells_to_cells, dest_weights.cells_to_cells
        ),
        cells_to_zones=_adjust_tier(
            pairs.cells_to_zones, costs.cells_to_zones, dest_weights.cells_to_zones
        ),
        zones_to_zones=_adjust_tier(
            pairs.zones_to_zones, costs.zones_to_zones, dest_weights.zones_to_zones
        ),
    )
