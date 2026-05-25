"""
To approaches to estimate daily traffic volumes (AADT) using betweenness centrality:
- A one-shot approach using native betweenness centrality in a limited radius
- A more detailed approach using distance-weighted node sampling; essentially implementing a quick
  and dirty 3-step travel demand model (trip generation, trip distribution, iterative
  route assignment until equilibrium is reached). The mode is fixed.
"""

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
import networkx as nx
from numba import njit

from aperta import network_processing, utils
from aperta.od_pairs import TieredODPairs


def get(g: nx.MultiGraph,
        routing_edge_weight: str,
        expected_km_driven: int | float,
        cutoff: int | float | None = None,
        nodes: list[str | int] | None = None,
        nested_node_sample: dict | None = None) -> pd.Series:
    """Naive traffic flow estimation using betweenness centrality with a cutoff or a node sample."""

    if nested_node_sample:
        bc = network_processing.get_nested_edge_betweenness_using_igraph(g, nested_node_sample,
                                                                  True, routing_edge_weight)
    else:
        bc = network_processing.get_edge_betweenness_using_igraph(g, True, cutoff, routing_edge_weight,
                                                                  nodes, nodes)
    # Normalize betweenness
    lengths = nx.get_edge_attributes(g, 'length')
    factor = expected_km_driven / sum([v * lengths[k] for k, v in bc.items()])
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
    """Pre-compute per-zone shared (zone_dests, zone_score, region_dests, region_score)
    with optional mask applied. Done once per zone — reused across every cell in
    that zone during sampling, amortizing both the `cost_to_weight` call and the
    mask-filter step.

    Empty arrays when a zone has no zone- or zones-to-regions-tier dests.
    """
    z2z_d = pairs.zones_to_zones or {}
    z2r_d = pairs.zones_to_regions or {}
    z2z_w = weights.zones_to_zones or {}
    z2r_w = weights.zones_to_regions or {}
    z2z_c = costs.zones_to_zones or {}
    z2r_c = costs.zones_to_regions or {}
    z2z_m = (mask.zones_to_zones if mask is not None else None) or {}
    z2r_m = (mask.zones_to_regions if mask is not None else None) or {}
    empty_dest = np.empty(0, dtype=object)
    empty_score = np.empty(0)

    out: dict = {}
    for zn in set(z2z_d) | set(z2r_d):
        # Zone tier
        if zn in z2z_d and len(z2z_d[zn]):
            zd, zw, zc = z2z_d[zn], z2z_w[zn], z2z_c[zn]
            if zn in z2z_m:
                m = z2z_m[zn]
                zd, zw, zc = zd[m], zw[m], zc[m]
            zone_dests = zd
            zone_score = zw * cost_to_weight(zc)
        else:
            zone_dests, zone_score = empty_dest, empty_score
        # Region tier
        if zn in z2r_d and len(z2r_d[zn]):
            rd, rw, rc = z2r_d[zn], z2r_w[zn], z2r_c[zn]
            if zn in z2r_m:
                m = z2r_m[zn]
                rd, rw, rc = rd[m], rw[m], rc[m]
            region_dests = rd
            region_score = rw * cost_to_weight(rc)
        else:
            region_dests, region_score = empty_dest, empty_score
        out[zn] = (zone_dests, zone_score, region_dests, region_score)
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
    integrating cell, zone, and zones-to-regions tiers.

    Per origin cell, the three tier dest arrays are concatenated on the fly into
    one combined dest pool with per-pair scores `weight * cost_to_weight(cost)`.
    Sampling is then a single `np.random.choice`-equivalent (JITted) over the
    pool. Peak memory is bounded by the largest single per-origin concatenation,
    not by `n_orig × total_dests`.

    The per-zone shared scores (zone-tier + zones-to-regions tier) are computed
    once per zone (not per cell), so the `cost_to_weight` call is amortized
    across all cells in the zone.

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
    origins = np.asarray(list(pairs.cells_to_cells.keys()))
    p = np.asarray(orig_weights, dtype=float)
    p = p / p.sum()
    chosen = random_state.choice(origins, n_orig, True, p)

    # Pre-compute per-zone shared dest arrays + scores, with zone- and region-
    # tier masks applied (so the same filter is reused across all cells in the
    # zone — `cost_to_weight` and the mask filter both amortize).
    z_combo = _zone_tier_dests_and_scores(pairs, weights, costs, cost_to_weight, mask)
    cell_mask_dict = (mask.cells_to_cells if mask is not None else None) or {}
    empty_dest = np.empty(0, dtype=object)
    empty_score = np.empty(0)

    # Group sampled origins by zone — shared work is then done once per zone-group.
    # Dedupe so a duplicated origin in `chosen` doesn't trigger redundant work
    # (the original implementation also overwrote duplicate dict keys silently).
    seen: set = set()
    chosen_by_zone: dict = defaultdict(list)
    for c in chosen:
        if c in seen:
            continue
        seen.add(c)
        chosen_by_zone[cell_to_zone_node.get(c)].append(c)

    out: dict = {}
    for zone_node, cells_here in chosen_by_zone.items():
        zone_dests, zone_score, region_dests, region_score = z_combo.get(
            zone_node, (empty_dest, empty_score, empty_dest, empty_score),
        )
        for c in cells_here:
            cell_dests = pairs.cells_to_cells[c]
            cell_costs = costs.cells_to_cells[c]
            cell_weights = weights.cells_to_cells[c]
            if c in cell_mask_dict:
                m = cell_mask_dict[c]
                cell_dests, cell_costs, cell_weights = cell_dests[m], cell_costs[m], cell_weights[m]
            cell_score = cell_weights * cost_to_weight(cell_costs)
            all_dests = np.concatenate([cell_dests, zone_dests, region_dests])
            all_score = np.concatenate([cell_score, zone_score, region_score])
            rvals = random_state.random(n_dest)
            indices = _weighted_sample_indices(all_score, rvals)
            out[c] = all_dests[indices]
    return out
