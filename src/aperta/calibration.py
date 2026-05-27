"""Iterative calibration of per-edge weights against observed trip-time data.

`calibrate_edge_weights` fits a linear model relating observed point-to-point
trip times to features collected along the routed shortest path plus features
at trip endpoints. The same feature set defines both the per-edge weight
formula used for routing AND the regression — keeping the two consistent
(a subtle pitfall in earlier ad-hoc calibration code).

Model:

    time_trip = α · baseline_time
              + Σ_m coef_m · (baseline_time · length-weighted-avg of m along path)
              + Σ_a coef_a · (sum of a along path)
              + Σ_e coef_e · (endpoint value of e)
              + constant

where features come in three classes (matching how they enter the per-edge
duration formula in `examples/swiss/prepare/4_edge_weights.ipynb`):

- **multiplier**: scales baseline speed (so it multiplies baseline time per
  edge — appears in the regression as `baseline_time · feature_avg`).
  Examples: local density, traffic flow.
- **additive_route**: adds seconds per unit summed along the path. Examples:
  intersection counts (sec per intersection), elevation gain (sec per metre).
- **additive_endpoint**: adds seconds based on the value of a node attribute
  at the origin and at the destination. Examples: snap distance, local
  density.

Iteration (option A from the design discussion): re-route after each OLS fit,
since updated coefficients change edge weights and therefore the chosen path
+ feature aggregates. Cheap to repeat — usually converges in 2-3 passes.

This module does NOT compute betweenness / traffic flows itself. Treat the
traffic estimate as just another per-edge attribute the caller supplies (e.g.
via `network_processing.get_nested_edge_betweenness`). Then include it in
`multiplier_features` (if it scales duration like density) or
`additive_route_features` (if seconds-per-unit).
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from aperta import geo_mapping, geo_processing, network_processing, routing


# Used to convert km/h → m/s.
_KMH_TO_MS = 1.0 / 3.6


@dataclass
class CalibrationResult:
    """Outcome of `calibrate_edge_weights`.

    Attributes:
        coefficients: DataFrame indexed by feature name with columns
            `kind` (multiplier / additive_route / additive_endpoint / const /
            baseline), `coef` (fitted value), `p` (p-value),
            `mean_effect` (coef × mean of column, in seconds).
        r_squared: OLS R² on the held-in trips.
        n_used: number of ground-truth rows that survived snap + filter +
            successful routing.
        predicted_times: per-trip predicted time (Series indexed by trip_id);
            comparable to `ground_truth.loc[predicted_times.index, 'time_measured']`.
        observed_times: per-trip observed time (same index as predicted).
        routed_distances: per-trip routed distance (m); useful for
            distance-band breakdowns.
        rmse: overall RMSE on the held-in trips, in seconds.
        rmse_by_distance: Series of RMSE per distance band, indexed by
            band label (`'< 10 km'`, `'10-25 km'`, `'>= 25 km'`).
        edge_duration_attr: name of the per-edge attribute written to
            `graph` by the final iteration (default `'duration_calibrated'`).
            Downstream routing can use this as the cost.
        iter_log: DataFrame, one row per iteration, columns r_squared, rmse,
            n_used — useful to inspect convergence.
    """
    coefficients: pd.DataFrame
    r_squared: float
    n_used: int
    predicted_times: pd.Series
    observed_times: pd.Series
    routed_distances: pd.Series
    rmse: float
    rmse_by_distance: pd.Series
    edge_duration_attr: str
    iter_log: pd.DataFrame


def _baseline_edge_duration(graph: nx.MultiDiGraph,
                            baseline_speed_attr: str) -> dict:
    """Per-edge baseline duration in seconds: length / (speed / 3.6)."""
    out: dict = {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        speed = float(data[baseline_speed_attr])
        length = float(data['length'])
        if speed <= 0:
            raise ValueError(f"Edge ({u},{v},{k}) has non-positive "
                             f"{baseline_speed_attr}={speed!r}; calibration needs "
                             "a positive baseline speed.")
        out[(u, v, k)] = length / (speed * _KMH_TO_MS)
    return out


def _apply_edge_durations(graph: nx.MultiDiGraph,
                          baseline_duration: dict,
                          alpha: float,
                          multiplier_coefs: dict[str, float],
                          additive_route_coefs: dict[str, float],
                          out_attr: str,
                          min_edge_duration: float = 0.01) -> None:
    """Write per-edge predicted duration to `out_attr`.

        edge_duration = α · baseline + baseline · Σ_m coef_m · m_value
                                     + Σ_a coef_a · a_value

    `α` calibrates the overall baseline scale; multiplier coefs scale the
    baseline by per-feature amounts (added — not multiplied — to the α
    baseline term, since they enter as `baseline · feature` in the linear
    OLS model). Mirrors the per-edge formula in
    `examples/swiss/prepare/4_edge_weights.ipynb`, minus the floor / cap.

    Per-edge duration is clipped to `min_edge_duration` seconds. Without
    this, a single transient negative coefficient (mid-iteration OLS noise,
    common when starting from zero on a noisy feature) can produce a
    negative edge weight, which scipy's Dijkstra refuses to handle. The
    clip is wide of any physically meaningful duration so well-conditioned
    fits are unaffected.
    """
    for u, v, k, data in graph.edges(keys=True, data=True):
        base = baseline_duration[(u, v, k)]
        mult_term = base * sum(c * float(data.get(f, 0.0))
                               for f, c in multiplier_coefs.items())
        add_term = sum(c * float(data.get(f, 0.0))
                       for f, c in additive_route_coefs.items())
        data[out_attr] = max(alpha * base + mult_term + add_term,
                             min_edge_duration)


def _filter_and_snap_legs(legs: pd.DataFrame,
                          graph: nx.MultiDiGraph,
                          *,
                          min_trip_distance: float,
                          max_trip_distance: float,
                          max_dist_to_line_ratio: float,
                          snap_max_distance: float) -> pd.DataFrame:
    """Apply trip filters + snap origin / dest to nearest network nodes.

    Adds columns: `nx_node_orig`, `nx_node_dest`, `snap_dist_orig`,
    `snap_dist_dest`. Drops rows where either snap fails or distance
    is beyond `snap_max_distance`.
    """
    if 'dist_line' not in legs.columns:
        legs = legs.copy()
        dx = legs['dest_x'] - legs['orig_x']
        dy = legs['dest_y'] - legs['orig_y']
        legs['dist_line'] = np.hypot(dx, dy)

    n_in = len(legs)
    legs = legs[(legs['dist_line'] >= min_trip_distance)
                & (legs['dist_line'] < max_trip_distance)]
    if 'dist_measured' in legs.columns:
        ratio = legs['dist_measured'] / legs['dist_line']
        legs = legs[ratio < max_dist_to_line_ratio]
    logging.info(f"  Trip filter: {n_in:,} → {len(legs):,} legs after dist_line "
                 f"[{min_trip_distance:.0f}, {max_trip_distance:.0f}] m + "
                 f"dist_measured/dist_line < {max_dist_to_line_ratio}.")

    legs = legs.copy()
    for side in ('orig', 'dest'):
        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(legs[f'{side}_x'], legs[f'{side}_y']),
            index=legs.index,
        )
        node_ids, dists = network_processing.snap_to_network_nodes(
            points, graph, max_distance=snap_max_distance)
        legs[f'nx_node_{side}'] = node_ids
        legs[f'snap_dist_{side}'] = dists

    n_before = len(legs)
    legs = legs.dropna(subset=['nx_node_orig', 'nx_node_dest'])
    logging.info(f"  Snap filter: {n_before:,} → {len(legs):,} legs within "
                 f"{snap_max_distance:.0f} m of a network node.")
    return legs


def _join_endpoint_features(routed: pd.DataFrame,
                            legs: pd.DataFrame,
                            graph: nx.MultiDiGraph,
                            endpoint_features: list[str]) -> pd.DataFrame:
    """Add per-trip columns `<feature>_orig` and `<feature>_dest` from node attrs."""
    if not endpoint_features:
        return routed
    nodes_iter = graph.nodes(data=True)
    node_attrs = pd.DataFrame.from_dict(
        {n: {f: d.get(f, np.nan) for f in endpoint_features} for n, d in nodes_iter},
        orient='index',
    )
    aligned_legs = legs.loc[legs.index.intersection(routed.index)]
    for side in ('orig', 'dest'):
        node_ids = aligned_legs[f'nx_node_{side}']
        for f in endpoint_features:
            routed.loc[node_ids.index, f'{f}_{side}'] = (
                node_ids.map(node_attrs[f]).values)
    # Snap distance is on legs, not on graph nodes — propagate it through too.
    for side in ('orig', 'dest'):
        col = f'snap_dist_{side}'
        if col in legs.columns:
            routed.loc[aligned_legs.index, col] = (
                aligned_legs[col].reindex(routed.index).values)
    return routed


def _build_design_matrix(routed: pd.DataFrame,
                         multiplier_features: list[str],
                         additive_route_features: list[str],
                         additive_endpoint_features: list[str],
                         constant: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the OLS design matrix `X` and the list of feature column names.

    Multiplier features enter as `cost · feature_avg` (a velocity-like
    interaction term). Additive route features enter as raw sums. Endpoint
    features become two columns each: `<f>_orig` and `<f>_dest`.

    Returns `(X, feature_columns, kinds_per_column)`. The first column is
    always `cost` (baseline duration along the routed path) — its OLS
    coefficient is the calibrated multiplier on the per-edge baseline (the
    `α` term in the model docstring).
    """
    rows = {'baseline_time': routed['cost'].astype(float)}
    kinds = ['baseline']
    feat_cols = ['baseline_time']
    for f in multiplier_features:
        col = f'{f}__mult'
        rows[col] = (routed['cost'] * routed[f]).astype(float)
        kinds.append('multiplier')
        feat_cols.append(col)
    for f in additive_route_features:
        rows[f] = routed[f].astype(float)
        kinds.append('additive_route')
        feat_cols.append(f)
    for f in additive_endpoint_features:
        for side in ('orig', 'dest'):
            col = f'{f}_{side}'
            rows[col] = routed[col].astype(float)
            kinds.append('additive_endpoint')
            feat_cols.append(col)
    X = pd.DataFrame(rows)
    if constant:
        X.insert(0, 'const', 1.0)
        kinds.insert(0, 'const')
        feat_cols.insert(0, 'const')
    return X, feat_cols, kinds


def _rmse_by_distance(observed: pd.Series, predicted: pd.Series,
                      dist_line: pd.Series) -> pd.Series:
    """RMSE per distance band — `< 10 km` / `10-25 km` / `>= 25 km`."""
    bands = [('< 10 km', dist_line < 10_000),
             ('10-25 km', (dist_line >= 10_000) & (dist_line < 25_000)),
             ('>= 25 km', dist_line >= 25_000)]
    rows = {}
    for label, mask in bands:
        idx = observed.index[mask.reindex(observed.index, fill_value=False)]
        if len(idx) == 0:
            rows[label] = np.nan
        else:
            err = (predicted.loc[idx] - observed.loc[idx]).to_numpy()
            rows[label] = float(np.sqrt((err ** 2).mean()))
    return pd.Series(rows)


def calibrate_edge_weights(
    graph: nx.MultiDiGraph,
    ground_truth: pd.DataFrame,
    *,
    baseline_speed_attr: str = 'speed_kph',
    multiplier_features: dict[str, float] | None = None,
    additive_route_features: dict[str, float] | None = None,
    additive_endpoint_features: dict[str, float] | None = None,
    constant: bool = True,
    n_iterations: int = 3,
    snap_max_distance: float = 300.0,
    min_trip_distance: float = 3_000.0,
    max_trip_distance: float = 100_000.0,
    max_dist_to_line_ratio: float = 4.0,
    edge_duration_attr: str = 'duration_calibrated',
) -> CalibrationResult:
    """Iteratively calibrate per-edge durations against observed trip times.

    See module docstring for the model. Each iteration:
      1. Writes per-edge duration to `edge_duration_attr` from the current
         coefficients (or initial guesses on iteration 1).
      2. Routes each ground-truth trip on those weights, aggregating features
         along the path.
      3. Fits an OLS model of `time_measured` ~ baseline_time + features +
         endpoint terms.
      4. Updates coefficients to the OLS fit.

    Args:
        graph: routable networkx graph. Must carry `length` and
            `baseline_speed_attr` on every edge, plus every attribute named
            in the feature dicts.
        ground_truth: DataFrame with columns `orig_x`, `orig_y`, `dest_x`,
            `dest_y`, `time_measured` (seconds). Optional `dist_measured`
            enables the dist-ratio filter. Optional `dist_line` is computed
            from coords if not provided.
        baseline_speed_attr: per-edge speed in km/h (e.g. from
            `osmnx.add_edge_speeds`). Not modified by this function.
        multiplier_features: `{edge_attr: initial_coef}`. Each scales the
            baseline duration (`new_dur = old_dur · (1 + Σ coef · feat)`).
            Use for density-like features.
        additive_route_features: `{edge_attr: initial_coef}`. Each contributes
            `coef · feat_value` seconds per edge (summed along path). Use for
            intersection counts, elevation gain, etc.
        additive_endpoint_features: `{node_attr: initial_coef}`. Each adds
            `coef · value_at_origin + coef · value_at_destination` to total
            trip duration. Use for snap distance, local density at endpoints.
        constant: include an intercept in the OLS fit.
        n_iterations: number of route-fit cycles. 2-3 usually converges.
        snap_max_distance: drop trips where origin or destination is farther
            than this from any network node (metres).
        min_trip_distance: drop trips with `dist_line` below this (metres).
        max_trip_distance: drop trips with `dist_line` above this (metres).
        max_dist_to_line_ratio: if `dist_measured` is present, drop trips
            where `dist_measured / dist_line` exceeds this (long detours are
            usually data noise).
        edge_duration_attr: name of the per-edge duration attribute written
            on `graph` (overwritten each iteration).

    Returns:
        `CalibrationResult` — see its docstring.

    Raises:
        ValueError: if any required column is missing or every trip filters
            out before fitting.
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError(
            "calibrate_edge_weights needs `statsmodels` (install with "
            "`pip install 'aperta[examples]'` or `pip install statsmodels`)."
        ) from e

    multiplier_features = dict(multiplier_features or {})
    additive_route_features = dict(additive_route_features or {})
    additive_endpoint_features = dict(additive_endpoint_features or {})

    required = {'orig_x', 'orig_y', 'dest_x', 'dest_y', 'time_measured'}
    missing = required - set(ground_truth.columns)
    if missing:
        raise ValueError(f"`ground_truth` is missing required columns: {sorted(missing)}")

    logging.info(f"Calibration: {len(ground_truth):,} input trips, "
                 f"{len(multiplier_features)} multiplier + "
                 f"{len(additive_route_features)} additive-route + "
                 f"{len(additive_endpoint_features)} additive-endpoint features.")

    # 1. Pre-snap + filter once — depends only on the graph topology, not on
    #    the (iteratively changing) edge weights.
    legs = _filter_and_snap_legs(
        ground_truth, graph,
        min_trip_distance=min_trip_distance,
        max_trip_distance=max_trip_distance,
        max_dist_to_line_ratio=max_dist_to_line_ratio,
        snap_max_distance=snap_max_distance,
    )
    if len(legs) == 0:
        raise ValueError("No trips remain after snap + filter.")

    # 2. Baseline per-edge duration (length / speed) computed once — feeds
    #    into every iteration's edge-duration formula.
    baseline_duration = _baseline_edge_duration(graph, baseline_speed_attr)

    # 4. Iterate: apply current coefs → route → fit → update coefs.
    #    α (baseline scale) starts at 1.0; multiplier/additive/endpoint
    #    coefs start at user-supplied initial values. After each OLS fit,
    #    α + coefs are read directly from the fit (NO cumulative rescaling
    #    of baseline_duration — that was a confusing earlier design that
    #    made α drift across iterations even when the model was stable).
    alpha = 1.0
    cur_mult = dict(multiplier_features)
    cur_add = dict(additive_route_features)
    cur_end = dict(additive_endpoint_features)

    # Aggregation per feature: multiplier features get length-weighted-avg
    # (so they enter as a speed-like correction); additive route features
    # get summed along the path.
    edge_feature_aggs: dict[str, str] = {
        **{f: 'length_weighted' for f in cur_mult},
        **{f: 'sum' for f in cur_add},
    }

    iter_log_rows = []
    fit_result = None
    routed = None
    feat_cols: list[str] = []
    kinds: list[str] = []

    for iteration in range(1, n_iterations + 1):
        # 4a. Write per-edge duration into the graph (scipy CSR is rebuilt
        # per call inside shortest_path_metrics_one_to_one — cheap relative
        # to the actual Dijkstras).
        _apply_edge_durations(graph, baseline_duration, alpha, cur_mult,
                              cur_add, edge_duration_attr)

        # 4b. Route every trip + aggregate features along the path.
        routed = routing.shortest_path_metrics_one_to_one(
            graph, list(legs.index), legs['nx_node_orig'], legs['nx_node_dest'],
            weight=edge_duration_attr, length_attr='length',
            edge_features=edge_feature_aggs,
        )
        routed = _join_endpoint_features(routed, legs, graph, list(cur_end))

        # 4c. OLS fit.
        X, feat_cols, kinds = _build_design_matrix(
            routed, list(cur_mult), list(cur_add), list(cur_end), constant)
        y = legs.loc[routed.index, 'time_measured']
        valid = X.notna().all(axis=1) & y.notna()
        X_f, y_f = X[valid], y[valid]
        if len(y_f) < len(feat_cols) + 1:
            raise ValueError(
                f"Iteration {iteration}: only {len(y_f)} valid rows for "
                f"{len(feat_cols)} feature columns — calibration ill-posed.")
        fit_result = sm.OLS(y_f, X_f).fit()

        # 4d. Update coefficient state for next iteration directly from the
        #     OLS fit (no rescaling — coefs are in the units of the model
        #     equation; the per-edge formula in _apply_edge_durations uses
        #     them as-is).
        coefs = fit_result.params
        alpha = float(coefs['baseline_time'])
        cur_mult = {f: float(coefs[f'{f}__mult']) for f in cur_mult}
        cur_add = {f: float(coefs[f]) for f in cur_add}
        # Endpoint coefs are stored per (feature, side); the per-edge
        # formula doesn't use them. Average the two sides so cur_end stays
        # a single-value dict (only relevant for the next iteration's
        # design matrix, which rebuilds the per-side columns anyway).
        cur_end = {
            f: (float(coefs[f'{f}_orig']) + float(coefs[f'{f}_dest'])) / 2
            for f in cur_end
        }

        edge_feature_aggs = {
            **{f: 'length_weighted' for f in cur_mult},
            **{f: 'sum' for f in cur_add},
        }

        # 4e. Iteration log.
        pred = fit_result.fittedvalues
        rmse_iter = float(np.sqrt(((pred - y_f) ** 2).mean()))
        iter_log_rows.append({
            'iteration': iteration,
            'r_squared': float(fit_result.rsquared),
            'rmse': rmse_iter,
            'n_used': int(len(y_f)),
            'alpha': alpha,
        })
        logging.info(
            f"  Iter {iteration}/{n_iterations}: R²={fit_result.rsquared:.4f}, "
            f"RMSE={rmse_iter:.1f} s, n={len(y_f):,}, α={alpha:.3f}")

    assert fit_result is not None and routed is not None

    # 5. Write final per-edge duration (with the LAST iteration's coefs).
    _apply_edge_durations(graph, baseline_duration, alpha, cur_mult, cur_add,
                          edge_duration_attr)

    # 6. Assemble outputs.
    coef_df = pd.DataFrame({
        'kind': kinds,
        'coef': fit_result.params.values,
        'p': fit_result.pvalues.values,
    }, index=feat_cols)
    # Mean effect (coef × mean of column) — same convention as lumos.
    X, _, _ = _build_design_matrix(
        routed, list(multiplier_features), list(additive_route_features),
        list(additive_endpoint_features), constant)
    y = legs.loc[routed.index, 'time_measured']
    valid = X.notna().all(axis=1) & y.notna()
    coef_df['mean_effect'] = [
        float((X.loc[valid, c] * coef_df.loc[c, 'coef']).mean())
        for c in coef_df.index
    ]
    coef_df = coef_df.round(4)

    predicted = pd.Series(fit_result.fittedvalues, index=y[valid].index,
                          name='predicted_time')
    observed = y[valid].rename('observed_time')
    dist_routed = routed.loc[valid.index[valid], 'distance'].rename('routed_distance')
    rmse_overall = float(np.sqrt(((predicted - observed) ** 2).mean()))
    rmse_band = _rmse_by_distance(observed, predicted, legs['dist_line'])

    return CalibrationResult(
        coefficients=coef_df,
        r_squared=float(fit_result.rsquared),
        n_used=int(valid.sum()),
        predicted_times=predicted,
        observed_times=observed,
        routed_distances=dist_routed,
        rmse=rmse_overall,
        rmse_by_distance=rmse_band,
        edge_duration_attr=edge_duration_attr,
        iter_log=pd.DataFrame(iter_log_rows).set_index('iteration'),
    )


# --- Traffic-counter calibration -----------------------------------------
#
# Calibration of a modeled traffic-flow estimate (e.g. the road_stress
# output from `traffic_flows.nested_node_sample` + betweenness) against
# observed point counters. Two primitives:
#
#   * `snap_counters_to_edges` — assign each counter to the right network
#     edge using a bearing-aware nearest-line match. The "right edge"
#     part is critical: a counter sits next to two or more parallel
#     edges (opposite directions, service roads, frontage roads) and
#     naïve nearest-line picks the wrong one most of the time.
#
#   * `evaluate_against_counters` — compute correlation R², regression
#     slope, and RMSE between modeled and observed AADT on the snapped
#     edges. R² is scale-invariant (use it to pick distribution-shape
#     params); slope tells the caller how to rescale absolute volumes
#     (e.g. derive `trips_per_person_per_day`).
#
# Together these let a notebook do simple coordinate-descent calibration:
# vary one parameter at a time, re-simulate flows, evaluate, plot the
# error curve, user picks the minimum. The library doesn't ship a
# coordinate-descent driver — too project-specific (simulation cost,
# parameter set, stopping criterion vary too much).


def snap_counters_to_edges(
    counters: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    *,
    search_radius: float | pd.Series = 50.0,
    bearing_tol_deg: float = 20.0,
    bearing_column: str = 'bearing_deg',
    eligible_edges: Callable[[pd.Series, gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None,
    bidirectional: bool | None = None,
) -> pd.DataFrame:
    """Snap directional traffic counters to the correct network edges.

    Counters typically sit next to several parallel candidate edges (opposite
    directions on the same road; service roads; frontage roads), so naïve
    nearest-line matching picks the wrong edge most of the time. This
    function adds a **bearing tolerance** filter — only edges whose local
    bearing matches the counter's `bearing_deg` (within `bearing_tol_deg`)
    are eligible. For directed graphs the bearing comparison is directional
    (a counter at bearing 90° won't snap to an edge pointing at 270°),
    which correctly assigns the two counters of a two-way road to the two
    directional edges.

    Uses `d['geometry']` from every edge — guaranteed by
    `consolidate_intersections`. Edges without a `geometry` attribute
    (e.g. raw OSMnx graphs with `simplify=True`) are silently skipped;
    consolidate first or call `osmnx.graph_to_gdfs(..., fill_edge_geometry=True)`.

    Args:
        counters: GeoDataFrame of point geometries with a `bearing_column`
            (degrees, OSM/north-clockwise convention). Same CRS as the
            graph node coordinates.
        graph: routable nx graph. Edge attributes must include `geometry`
            (LineString) and whatever `eligible_edges` reads.
        search_radius: max cartesian distance for candidate edges (CRS
            units). Pass a scalar for one global radius or a `pd.Series`
            aligned to `counters.index` for per-counter radii (e.g. wider
            for highway counters which sit further from the carriageway).
        bearing_tol_deg: max angular difference between counter bearing
            and local edge bearing at the snap point.
        bearing_column: counter column holding the directional bearing
            (default `'bearing_deg'`).
        eligible_edges: optional `(counter_row, candidate_edges_gdf) -> subset`
            callback. Use to restrict matches by class — typically a
            highway counter only matches highway edges, a local counter
            only matches local edges. Forwarded to
            [[geo_mapping.map_points_to_filtered_lines]].
        bidirectional: how to compare bearings. `True` collapses opposite
            bearings (counter at 90° matches edges at 90° AND 270°) —
            correct for undirected graphs where one edge represents both
            directions of a road. `False` is directional — correct for
            directed graphs (the default `nx.MultiDiGraph`). `None`
            auto-detects from `graph.is_directed()`.

    Returns:
        DataFrame indexed like `counters` with columns:
          - `u`, `v`, `k`: matched edge ID (or `pd.NA` if no acceptable
            match within radius);
          - `snap_dist`: cartesian distance counter → edge (or `NaN`);
          - `dist_along`: along-edge distance from edge start to nearest
            point on edge (or `NaN`).

    Unmatched counters get all-NA rows — drop with `result.dropna(subset=['u'])`.
    """
    if bidirectional is None:
        bidirectional = not graph.is_directed()
    if bearing_column not in counters.columns:
        raise ValueError(
            f"`counters` is missing required column `{bearing_column!r}` "
            f"(have: {list(counters.columns)})")

    # Build edges GDF from graph. Drop edges without geometry — they can't
    # be snapped to anyway, and the caller is responsible for consolidating
    # / filling geometry beforehand.
    edge_records = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        geom = d.get('geometry')
        if geom is None:
            continue
        rec = dict(d)
        rec['u'], rec['v'], rec['k'] = u, v, k
        rec['geometry'] = geom
        edge_records.append(rec)
    if not edge_records:
        raise ValueError(
            "No edges have a `geometry` attribute. Consolidate the graph "
            "via `network_processing.consolidate_intersections` first, or "
            "use `osmnx.graph_to_gdfs(..., fill_edge_geometry=True)`.")
    edges_gdf = gpd.GeoDataFrame(edge_records, geometry='geometry', crs=counters.crs)
    # Linear integer index so `map_points_to_filtered_lines`'s `line_id`
    # outputs lift back to (u, v, k) via a single .iloc lookup.
    edges_gdf = edges_gdf.reset_index(drop=True)

    def _accept(counter_row, edge_row, ctx) -> bool:
        edge_bearing = geo_processing.line_segment_bearing_at(
            edge_row.geometry, ctx['dist_along'])
        if np.isnan(edge_bearing):
            return False
        diff = geo_processing.angular_diff_deg(
            counter_row[bearing_column], edge_bearing,
            undirected=bidirectional)
        return float(diff) <= bearing_tol_deg

    matches = geo_mapping.map_points_to_filtered_lines(
        counters, edges_gdf, search_radius=search_radius,
        eligible_lines=eligible_edges, accept=_accept,
    )

    # Lift `line_id` (positional row in edges_gdf) back to (u, v, k).
    out = pd.DataFrame(index=counters.index)
    matched = matches['line_id'].notna()
    out['u'] = pd.NA
    out['v'] = pd.NA
    out['k'] = pd.NA
    if matched.any():
        idxs = matches.loc[matched, 'line_id'].astype(int).to_numpy()
        out.loc[matched, 'u'] = edges_gdf.iloc[idxs]['u'].to_numpy()
        out.loc[matched, 'v'] = edges_gdf.iloc[idxs]['v'].to_numpy()
        out.loc[matched, 'k'] = edges_gdf.iloc[idxs]['k'].to_numpy()
    out['snap_dist'] = matches['distance']
    out['dist_along'] = matches['dist_along']
    n_match = int(matched.sum())
    logging.info(
        f"snap_counters_to_edges: {n_match:,} of {len(counters):,} counters "
        f"matched ({n_match / max(len(counters), 1) * 100:.1f}%); "
        f"bidirectional={bidirectional}, tol={bearing_tol_deg}°.")
    return out


def evaluate_against_counters(
    modeled: pd.Series,
    counters: pd.DataFrame,
    *,
    observed_column: str = 'traffic_cars',
) -> dict:
    """Compare modeled per-edge AADT against snapped counter observations.

    Args:
        modeled: per-edge modeled AADT, indexed by `(u, v, k)` tuples (the
            output of `traffic_flows.nested_node_sample` + betweenness +
            AADT scaling).
        counters: DataFrame with `u`, `v`, `k` columns (from
            `snap_counters_to_edges`) and an observed-AADT column. Rows
            with NA in `u`/`v`/`k` are dropped (unmatched counters).
        observed_column: name of the observed-AADT column (default
            `'traffic_cars'`, matching the Swiss counter schema).

    Returns:
        Dict with:
          - `r2`: Pearson correlation² between modeled and observed —
            **scale-invariant**, so use this to pick distribution-shape
            params (lognormal σ, μ).
          - `slope`: slope from a no-intercept regression
            `modeled = slope · observed`. Tells you how to rescale
            absolute volumes — e.g. multiply `trips_per_person_per_day`
            by `1 / slope` to bring the modeled total in line with
            counters.
          - `rmse`: root-mean-square error on the matched set, in
            counter-units (veh/day).
          - `n_matched`: number of counters used in the comparison.
          - `merged`: DataFrame with `observed`, `modeled`, `(u, v, k)`
            for every matched counter — convenient for scatter plots.
    """
    matched = counters.dropna(subset=['u', 'v', 'k']).copy()
    if observed_column not in matched.columns:
        raise ValueError(
            f"`counters` is missing observed column `{observed_column!r}` "
            f"(have: {list(matched.columns)})")
    # Build the index lookup. `modeled` may be a Series with a MultiIndex
    # or a tuple-keyed flat index — handle both via .reindex with tuples.
    keys = list(zip(matched['u'].astype(int),
                    matched['v'].astype(int),
                    matched['k'].astype(int)))
    if isinstance(modeled.index, pd.MultiIndex):
        modeled_values = modeled.reindex(keys).to_numpy()
    else:
        modeled_values = np.array([modeled.get(k, np.nan) for k in keys])
    matched['modeled'] = modeled_values
    matched['observed'] = matched[observed_column].astype(float)
    matched = matched.dropna(subset=['modeled', 'observed'])
    if len(matched) == 0:
        return {'r2': np.nan, 'slope': np.nan, 'rmse': np.nan,
                'n_matched': 0, 'merged': matched}

    obs = matched['observed'].to_numpy()
    mod = matched['modeled'].to_numpy()
    r = float(np.corrcoef(obs, mod)[0, 1]) if np.std(obs) > 0 and np.std(mod) > 0 else np.nan
    r2 = r ** 2 if not np.isnan(r) else np.nan
    # No-intercept regression: slope = Σ(x·y) / Σ(x²) with x=observed.
    denom = float((obs ** 2).sum())
    slope = float((obs * mod).sum() / denom) if denom > 0 else np.nan
    rmse = float(np.sqrt(((mod - obs) ** 2).mean()))
    return {'r2': r2, 'slope': slope, 'rmse': rmse,
            'n_matched': int(len(matched)),
            'merged': matched[['u', 'v', 'k', 'observed', 'modeled']]}
