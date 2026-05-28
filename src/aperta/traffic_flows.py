"""Distance-weighted traffic-flow estimation via sampled betweenness centrality.

Estimates daily per-edge traffic volumes (AADT) by simulating a quick-and-dirty
3-step travel demand model: trip generation (origin sampling weighted by
population), trip distribution (per-origin destination sampling weighted by a
cost-decay function and per-destination attractiveness), and route assignment
(shortest-path routing on the current edge weights, accumulating per-edge
counts). Iterating the loop with congestion-aware edge-weight updates moves
the system toward equilibrium.

This module supplies the sampling primitives (`nested_node_sample`) and the
normalization step (`get`) that turns raw sampled-betweenness counts into a
per-edge volume calibrated against an expected total vehicle-kilometres figure.
The routing + per-edge accumulation itself lives in
`network_processing.get_nested_edge_betweenness`. A simpler alternative for
small study areas — radius-limited Brandes betweenness without explicit OD
sampling — also lives in `network_processing`.
"""

from collections import defaultdict
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from numba import njit

from aperta import network_processing, utils
from aperta.od_pairs import TieredODPairs


def get(
    g: nx.MultiGraph,
    routing_edge_weight: str,
    expected_km_driven: int | float,
    nested_node_sample: dict,
    *,
    cutoff: float | None = None,
) -> pd.Series:
    """Traffic-flow estimation from a nested OD sample, normalised to
    `expected_km_driven` total vehicle-kilometres.

    Routes each sampled (origin, dest) pair via scipy Dijkstra, accumulates
    per-edge usage counts (see `network_processing.get_nested_edge_betweenness`),
    then scales so that `sum(flow_e × length_e)` matches `expected_km_driven`.

    `cutoff` (optional): network-distance limit in `routing_edge_weight`
    units passed through to the per-origin Dijkstra. Set this to the
    upstream sampling radius (e.g. `r_zones` from `od_pairs.get_pairs`) —
    sampled destinations are guaranteed reachable within that radius, so
    the cutoff is correctness-preserving and gives a large speed-up on
    country-scale graphs. Default `None` = no cutoff.
    """
    bc = network_processing.get_nested_edge_betweenness(
        g, nested_node_sample, routing_edge_weight, cutoff=cutoff
    )
    lengths = nx.get_edge_attributes(g, "length")
    factor = expected_km_driven / sum(v * lengths[k] for k, v in bc.items())
    return bc * factor


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


@utils.timeit
def nested_node_sample(
    pairs: TieredODPairs,
    weights: TieredODPairs,
    costs: TieredODPairs,
    cell_to_zone_node: dict,
    orig_weights: np.ndarray | pd.Series,
    cost_to_weight: Callable,
    n_orig: int,
    n_dest: int,
    random_state: np.random.RandomState,
    *,
    mask: TieredODPairs | None = None,
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
            `pairs`. Typically the output of `od_pairs.dest_values`.
        costs: per-pair costs (e.g. line distances), same shape as `pairs`.
            Typically the output of `od_pairs.get_euclidian_dists`.
        cell_to_zone_node: `{cell_node -> zone_node}` mapping; build via
            `od_pairs.build_cell_to_zone_node_map`.
        orig_weights: per-origin sampling weights, aligned position-wise with
            `list(pairs.cells_to_cells.keys())`.
        cost_to_weight: monotone-decreasing function mapping a cost (e.g. distance
            in metres) to a per-pair weight. Vectorized — receives a 1-D array.
        n_orig, n_dest: number of origins to sample; number of destinations per
            sampled origin. Sampling is with replacement on both ends; repeats in
            the origin sample are deduped (each origin processed once).
        random_state: numpy RandomState; the only source of randomness.
        mask: optional boolean `TieredODPairs` (build via `od_pairs.make_mask`).
            Destinations where the mask is `False` are removed from the sampling
            pool. Missing origins or missing tiers in the mask are treated as
            "no filter" for that origin / tier.

    Returns: `{origin_cell_node -> np.ndarray[dest_node]}` of length `n_dest`.
    """
    if pairs.cells_to_cells is None:
        raise ValueError("`pairs.cells_to_cells` is None; cell-tier is required.")
    if costs.cells_to_cells is None or weights.cells_to_cells is None:
        raise ValueError("`costs` and `weights` must both have a populated cell-tier.")
    cell_pairs = pairs.cells_to_cells
    cell_costs_dict = costs.cells_to_cells
    cell_weights_dict = weights.cells_to_cells
    origins = np.asarray(list(cell_pairs.keys()))
    p = np.asarray(orig_weights, dtype=float)
    p = p / p.sum()
    chosen = random_state.choice(origins, n_orig, True, p)

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
    # zone-group. Dedupe so a duplicated origin in `chosen` doesn't trigger
    # redundant work (the original also overwrote duplicate dict keys silently).
    seen: set = set()
    chosen_by_zone: dict = defaultdict(list)
    for c in chosen:
        if c in seen:
            continue
        seen.add(c)
        chosen_by_zone[cell_to_zone_node.get(c)].append(c)

    out: dict = {}
    for zone_node, cells_here in chosen_by_zone.items():
        zone_dests, zone_score = z_combo.get(
            zone_node,
            (empty_dest, empty_score),
        )
        for c in cells_here:
            # Cell tier (cells_to_cells): per-cell origin + per-cell dest.
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
            rvals = random_state.random(n_dest)
            indices = _weighted_sample_indices(all_score, rvals)
            out[c] = all_dests[indices]
    return out
