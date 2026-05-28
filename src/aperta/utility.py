"""
Utility-based travel costs and accessibility.

Aperta supports linear utility specifications of the form::

    U_ij = constant
         + cost_coefficient * cost_ij
         + Σ_f f.coefficient * aggregated_route_feature_f(i, j)
         + Σ_o origin_features[o] * feature_o(i)
         + Σ_d destination_features[d] * feature_d(j)

where::

    cost_ij                       shortest-path cost on a chosen routing weight
    aggregated_route_feature(i,j) per-edge feature aggregated along the realised
                                  shortest path (sum / mean / etc.)
    feature(i), feature(j)        per-origin / per-destination scalar attributes
    constant, *_coefficient       per-mode parameters (typically calibrated from
                                  travel-survey data)

Computation is split in two steps to keep the routing pass cheap and to let
downstream cell-mode accessibility handle per-cell additions on its own terms:

1. `route_utility(...)` routes shortest paths once and computes the
   route-dependent components (cost + per-edge feature aggregations). Wraps
   `routing.tiered_path_aggregate`.

2. `add_endpoint_utility(...)` augments the route utility with the constant,
   origin, and destination components. Returns the full per-node-pair utility.

For **cell-mode** accessibility (per-cell overheads), the cell-overhead
contribution to utility (`β_cost · cell_walking_overhead`) is added by the
downstream accessibility function (`gravity`, `count_in_bins`, `nearest_k`)
via the existing `cell_overhead_column` mechanism. Precompute the per-cell
utility-overhead column in user code:

    cells['util_overhead'] = utility.cost_coefficient * cells['walk_overhead_s']

and pass it as `cell_overhead_column='util_overhead'`. The accessibility
function will add it to every cell's destination utilities — units match
(utils), no library changes needed.

For **logsum** (cross-modal) accessibility, combine per-mode utility ODMs with
`od_pairs.aggregate_across_modes(utilities_by_mode, aggregator='logsum')`,
then run gravity (with β = 1) or other downstream metrics on the combined
ODM. See the walkthrough notebook for a worked demonstration.

**Known limitations.**

*Self-pair utility under positive route-feature contributions.* For a
cell-to-itself OD pair, the realised shortest path has zero edges. The
route-feature contribution to utility is then 0 (see the handling inside
`route_utility` — this avoids a NaN-propagation bug where mean/min/max
aggregators over empty edge arrays would otherwise corrupt downstream
gravity / logsum sums).

The empty-path-as-zero treatment is the right call for utility specs where
route features contribute *negatively* to utility on net (e.g. perceived
road-class penalties, gradient costs). For utility specs where route
features contribute *positively* (e.g. greenery benefit, attractive-route
bonus), the self-pair can still appear less attractive than neighbouring
cells whose short routes earn a small positive route-feature contribution.
Semantically odd — the cell containing the opportunity shouldn't be
*worse* than its neighbours at accessing it — but mathematically correct
under the empty-path convention.

A more general fix would synthesise a representative within-cell route for
self-pairs (e.g. infer the typical route-feature value from neighbouring
cells, or apply the per-mode-defaults from the user's input). Not
currently implemented; flagged for future work.
"""

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import networkx as nx
import numpy as np
import pandas as pd

from aperta import od_pairs
from aperta.od_pairs import TieredODNodePairs, TieredODPairs
from aperta.routing import (
    PathAggregation,
    tiered_path_aggregate,
    tiered_path_costs,
)


class RouteFeature(NamedTuple):
    """A per-edge feature aggregated along the shortest path, with a utility
    coefficient. Used in `Utility.route_features`.

    During utility computation, the per-edge values are aggregated along the
    realised route (sum / mean / etc.) and then multiplied by `coefficient`
    to contribute `coefficient * aggregated_feature` to the OD-pair utility.

    `attribute`, `aggregator`: as in `routing.PathAggregation`. `coefficient`
    is the utility weight (β); typically negative for costs / penalties.
    """

    name: str
    attribute: str | Callable
    coefficient: float
    aggregator: str | Callable = "sum"


@dataclass
class Utility:
    """Linear utility specification.

    U_ij = constant
         + cost_coefficient * cost_ij
         + Σ_f f.coefficient * aggregated_f(i, j)             (route features)
         + Σ_o origin_features[o] * feature_o(i)              (origin features)
         + Σ_d destination_features[d] * feature_d(j)         (destination features)

    Coefficients can be positive or negative. Cost typically has a negative
    coefficient (cost reduces utility). The constant is the alternative-
    specific constant from a discrete-choice estimation; the cost coefficient
    and per-feature betas come from the same source.

    Cell-level features (cell-overhead, per-cell origin attributes) are NOT
    part of this spec — they are added by the downstream accessibility
    function via `cell_overhead_column`. See module docstring.
    """

    constant: float = 0.0
    cost_coefficient: float = 0.0
    route_features: list[RouteFeature] = field(default_factory=list)
    origin_features: dict[str, float] = field(default_factory=dict)
    destination_features: dict[str, float] = field(default_factory=dict)


def route_utility(
    pairs: TieredODPairs,
    graph: nx.Graph,
    cost_weight: str,
    utility: Utility,
    *,
    mask: TieredODPairs | None = None,
    dtype: np.dtype | type = np.float32,
) -> TieredODPairs:
    """Compute the route-dependent components of utility for every OD pair.

    For each OD pair (i, j):
        U_route(i, j) = cost_coefficient * cost(i, j)
                      + Σ_f f.coefficient * aggregated_f(i, j)

    where cost(i, j) is the shortest-path cost under `cost_weight` and the
    aggregations are over the edges of the realised (i, j) path.

    Origin, destination, and constant components are NOT included — add them
    via `add_endpoint_utility`. Cell-mode overhead is NOT included — handle
    that via the downstream accessibility function's `cell_overhead_column`.

    Internally calls `tiered_path_aggregate` (or `tiered_path_costs` when no
    route features are needed) so the routing pass is shared across the cost
    and all route features.

    Args:
        pairs: tiered destination IDs (typically from `od_pairs.get_pairs`).
        graph: routable networkx graph.
        cost_weight: edge attribute name used for routing AND as the cost
            contribution to utility (multiplied by `utility.cost_coefficient`).
        utility: the `Utility` spec.
        mask, dtype: as in `tiered_path_aggregate`.

    Returns:
        `TieredODPairs` of route-utility values (float64 by default).
        Unreachable / masked-out destinations are `np.nan` (NOT `np.inf` —
        utility is a signed quantity).
    """
    has_route_features = len(utility.route_features) > 0

    costs: TieredODPairs
    if has_route_features:
        path_aggs = [
            PathAggregation(name=rf.name, attribute=rf.attribute, aggregator=rf.aggregator)
            for rf in utility.route_features
        ]
        costs, aggs = tiered_path_aggregate(
            pairs,
            graph,
            cost_weight,
            edge_aggregations=path_aggs,
            mask=mask,
            dtype=dtype,
        )
    else:
        # No route features → skip path retrieval, faster.
        costs = tiered_path_costs(
            pairs,
            graph,
            cost_weight,
            mask=mask,
            dtype=dtype,
        )
        aggs = {}

    def _combine_per_tier(tier_attr: str) -> dict | None:
        cost_tier = getattr(costs, tier_attr)
        if cost_tier is None:
            return None
        out = {}
        for origin, cost_arr in cost_tier.items():
            # Treat inf costs as unreachable → NaN utility (signed-quantity convention).
            cost_finite = np.where(np.isfinite(cost_arr), cost_arr, np.nan)
            is_reachable = np.isfinite(cost_arr)
            u = utility.cost_coefficient * cost_finite
            for rf in utility.route_features:
                agg_arr = getattr(aggs[rf.name], tier_attr)[origin]
                contribution = rf.coefficient * agg_arr
                # Empty-path artifact: when cost is finite (typically 0 for a
                # self-pair) but the aggregator returned NaN (e.g. 'mean' over
                # zero edges is undefined), treat the route-feature contribution
                # as 0 — there's no route, so no route-feature utility. The
                # alternative — letting NaN propagate — would erase reachable
                # self-pair entries from downstream sums and produce visibly
                # wrong accessibility maps (the cell containing an opportunity
                # would lose its own contribution).
                contribution = np.where(
                    is_reachable & np.isnan(contribution),
                    0.0,
                    contribution,
                )
                u = u + contribution
            out[origin] = u.astype(dtype, copy=False)
        return out

    return TieredODNodePairs(
        cells_to_cells=_combine_per_tier("cells_to_cells"),
        cells_to_zones=_combine_per_tier("cells_to_zones"),
        zones_to_zones=_combine_per_tier("zones_to_zones"),
    )


def _origin_lookup(
    df: pd.DataFrame | None,
    column: str,
    node_column: str,
) -> dict | None:
    """Build a `{node_id -> value}` lookup for an origin feature.

    Returns `None` if `df` is missing or the column isn't present. Multiple
    rows sharing a node are mean-aggregated (typical for "many cells per
    network node" cases — take the mean as the per-node representative).
    """
    if df is None or column not in df.columns:
        return None
    return df.set_index(node_column)[column].groupby(level=0).mean().to_dict()


def add_endpoint_utility(
    route_utility: TieredODPairs,
    pairs: TieredODPairs,
    utility: Utility,
    *,
    cells: pd.DataFrame | None = None,
    zones: pd.DataFrame | None = None,
    node_column: str = "node_id",
) -> TieredODPairs:
    """Add constant, origin, and destination components to route utility.

    For each OD pair (i, j):
        U_full(i, j) = U_route(i, j)
                     + constant
                     + Σ_o origin_features[o] * feature_o(i)
                     + Σ_d destination_features[d] * feature_d(j)

    Origin features are looked up at the origin node. For the
    `cells_to_cells` and `cells_to_zones` tiers, origins are cell-tier
    nodes (`cells` is used); for `zones_to_zones`, origins are zone-tier
    nodes (`zones` is used). If multiple cells / zones map to the same
    network node, their feature values are averaged.

    Destination features are looked up via `od_pairs.dest_values` —
    cell-tier dests look up in `cells`, while `cells_to_zones` and
    `zones_to_zones` dests look up in `zones`. Missing feature columns at
    a given tier silently contribute zero to that tier (the OD pairs still
    get the other components).

    Cell-mode handling: this function operates at the network-node level
    (its origins are nodes). Per-cell origin features (different for two
    cells sharing a node) and per-cell-overhead utility contributions are
    NOT handled here. Compute them as a single per-cell column in user code:

        cells['util_overhead'] = (utility.cost_coefficient * cells['walk_overhead_s']
                                + utility.origin_features.get('density', 0) * cells['density'])

    and pass via `cell_overhead_column='util_overhead'` to the accessibility
    function. (`add_endpoint_utility` handles origin features at the
    cells-per-node-averaged level only.)

    Args:
        route_utility: per-OD route utility from `route_utility(...)`.
        pairs: tiered destination IDs (same as used in `route_utility`).
        utility: the `Utility` spec.
        cells, zones: per-unit DataFrames carrying the named features.
            Each must have `node_column` mapping to the network node ID
            for that tier.
        node_column: column name in cells/zones giving the network node.
            Default `'node_id'`.

    Returns:
        `TieredODPairs` of full per-OD utility (float64).
    """
    # Pre-compute destination-feature ODMs (tier-aware via dest_values).
    dest_value_odms: dict[str, tuple[TieredODPairs, float]] = {}
    for feature_col, beta in utility.destination_features.items():
        if cells is None or feature_col not in cells.columns:
            raise ValueError(
                f"Destination feature {feature_col!r} not in cells.columns "
                f"(needed for cell-tier destination lookups)."
            )
        d = od_pairs.dest_values(
            feature_col,
            pairs,
            cells,
            node_column,
            zones=zones if zones is not None and feature_col in zones.columns else None,
        )
        dest_value_odms[feature_col] = (d, beta)

    # Pre-compute origin-feature lookups, per tier.
    origin_cell_lookups: dict[str, tuple[dict, float]] = {}
    origin_zone_lookups: dict[str, tuple[dict, float]] = {}
    for feature_col, beta in utility.origin_features.items():
        cell_lu = _origin_lookup(cells, feature_col, node_column)
        if cell_lu is None:
            raise ValueError(
                f"Origin feature {feature_col!r} not in cells.columns "
                f"(needed for cell-tier origin lookups)."
            )
        origin_cell_lookups[feature_col] = (cell_lu, beta)
        zone_lu = _origin_lookup(zones, feature_col, node_column)
        if zone_lu is not None:
            origin_zone_lookups[feature_col] = (zone_lu, beta)

    def _combine_per_tier(
        tier_attr: str, origin_lookups: dict[str, tuple[dict, float]]
    ) -> dict | None:
        route_tier = getattr(route_utility, tier_attr)
        if route_tier is None:
            return None
        out: dict = {}
        for origin, u_arr in route_tier.items():
            u = u_arr + utility.constant
            for lookup, beta in origin_lookups.values():
                val = lookup.get(origin)
                if val is not None:
                    u = u + beta * float(val)
            for dest_odm, beta in dest_value_odms.values():
                d_arr = getattr(dest_odm, tier_attr)
                if d_arr is None:
                    continue
                d_vals = d_arr.get(origin)
                if d_vals is not None:
                    u = u + beta * d_vals
            out[origin] = u
        return out

    return TieredODNodePairs(
        cells_to_cells=_combine_per_tier("cells_to_cells", origin_cell_lookups),
        cells_to_zones=_combine_per_tier("cells_to_zones", origin_cell_lookups),
        zones_to_zones=_combine_per_tier("zones_to_zones", origin_zone_lookups),
    )
