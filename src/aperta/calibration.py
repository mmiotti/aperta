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
from dataclasses import dataclass
from typing import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from aperta import data_processing, geo_mapping, geo_processing, network_snap, routing

# Used to convert km/h → m/s.
_KMH_TO_MS = 1.0 / 3.6


@dataclass
class CalibrationResult:
    """Outcome of `calibrate_edge_weights`.

    Attributes:
        coefficients: DataFrame indexed by feature name with columns
            `coef` (fitted value) and `p` (p-value). Includes the OLS
            constant (`const`, if `constant` was set), `baseline_time`
            (the α scale on baseline duration), and one row per
            multiplier / additive_route / additive_endpoint feature.

        n_used: Number of ground-truth trips that survived snap +
            distance filters and entered the OLS fit.

        Three per-distance-band metrics frames are reported, each a
        DataFrame indexed by distance band (`"all"`, `"< 5 km"`,
        `"5-25 km"`, `">= 25 km"`) with columns `r2`, `rmse`, `bias`.
        Each measures fit between observed times and a different
        prediction mechanism on the same trip set:

          * `metrics_baseline` — predict via routing with
            `length / speed_kph` only (the un-calibrated graph). Invariant
            to the user's prior coefficient choices. Quantifies the lift
            the calibration provides over the raw speed_kph attribute.
          * `metrics_calibrated` — predict via routing on the
            CALIBRATED graph (final α + coefs applied to edge weights;
            Dijkstra re-runs to pick paths under those weights, and the
            sum of weights along the chosen path is the prediction). This
            is the production-relevant number — what you'd actually get
            if you deployed the calibrated graph for routing.
          * `metrics_regression` — OLS R² of the final iteration's
            linear-model fit. The OLS sits on the iteration-N routing's
            paths (NOT the final-coefs routing's paths), so this is an
            upper bound on calibrated R². The gap between regression and
            calibrated R² is a convergence diagnostic: small gap →
            calibration converged at this `n_iterations`; large gap →
            bumping `n_iterations` would tighten things.

        Quick overall-fit access: `result.metrics_calibrated.loc['all', 'r2']`.
    """

    coefficients: pd.DataFrame
    metrics_baseline: pd.DataFrame
    metrics_calibrated: pd.DataFrame
    metrics_regression: pd.DataFrame
    n_used: int


def apply_edge_durations(
    graph: nx.MultiGraph,
    *,
    multiplier_features: dict[str, float] | None = None,
    additive_route_features: dict[str, float] | None = None,
    alpha: float = 1.0,
    out_attr: str = "duration",
    baseline_duration_attr: str = "speed_kph",
    min_speed_kph: float = 1.0,
    max_speed_kph: float = 120.0,
) -> None:
    """Write per-edge duration to `out_attr` (mutates `graph` in place):

        edge_duration = α · base + base · Σ_m c_m · m_value + Σ_a c_a · a_value

    where `base = baseline_duration_attr` (usually: speed limit for
    cars, a fixed speed for active modes) before any features are added.

    Term semantics:
      - **α · base** — global scale on the network-derived baseline. `α=1.0`
        (default) is the prior; trust the network's `speed_kph` as-is.
        After calibration, pass the fitted `α` (from
        `CalibrationResult.coefficients` row `baseline_time`) to apply
        the calibrated weights to a fresh graph for downstream routing.
      - **base · multiplier_features[f] · edge[f]** — speed-like correction:
        per-feature coefficient scales the baseline by `f`'s value on
        that edge (e.g. `slope_climb · 8.0` on a 10 % climb adds 80 %
        to that edge's baseline time).
      - **additive_route_features[f] · edge[f]** — raw seconds per edge,
        independent of length / speed (e.g. `is_traffic_signal · 5.0`
        adds 5 s on signalised intersections).
    """
    multiplier_features = multiplier_features or {}
    additive_route_features = additive_route_features or {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        length = float(data["length"])
        base = float(data[baseline_duration_attr])
        mult_term = base * sum(c * float(data.get(f, 0.0)) for f, c in multiplier_features.items())
        add_term = sum(c * float(data.get(f, 0.0)) for f, c in additive_route_features.items())
        duration = alpha * base + mult_term + add_term
        max_duration = length / (min_speed_kph * _KMH_TO_MS)
        min_duration = length / (max_speed_kph * _KMH_TO_MS)
        data[out_attr] = max(min(duration, max_duration), min_duration)


def _build_predictors(
    routed: pd.DataFrame,
    baseline_edge_attr: str,
    multiplier_features: list[str],
    additive_route_features: list[str],
    additive_endpoint_features: list[str],
    constant: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the OLS design matrix `X` and the list of feature column names.

    Multiplier features enter as `cost · feature_avg` (a velocity-like
    interaction term). Additive route features enter as raw sums. Endpoint
    features become two columns each: `<f>_orig` and `<f>_dest`.

    Returns `(X, feature_columns, kinds_per_column)`. The first column is
    always `cost` (baseline duration along the routed path) — its OLS
    coefficient is the calibrated multiplier on the per-edge baseline (the
    `α` term in the model docstring).
    """
    rows = {"baseline_time": routed[baseline_edge_attr].astype(float)}
    kinds = ["baseline"]
    feat_cols = ["baseline_time"]
    for f in multiplier_features:
        rows[f] = (routed["cost"] * routed[f]).astype(float)
        kinds.append("multiplier")
        feat_cols.append(f)
    for f in additive_route_features:
        rows[f] = routed[f].astype(float)
        kinds.append("additive_route")
        feat_cols.append(f)
    for f in additive_endpoint_features:
        for side in ("orig", "dest"):
            col = f"{f}_{side}"
            rows[col] = routed[col].astype(float)
            kinds.append("additive_endpoint")
            feat_cols.append(col)
    X = pd.DataFrame(rows)
    if constant:
        X.insert(0, "const", 1.0)
        kinds.insert(0, "const")
        feat_cols.insert(0, "const")
    return X, feat_cols, kinds


def _metrics_by_distance(
    observed: pd.Series, predicted: pd.Series, dist_line: pd.Series
) -> pd.DataFrame:
    """Error metrics for all trips and per distance band."""

    bands = [
        ("all", dist_line > 0),
        ("< 5 km", dist_line < 5_000),
        ("5-25 km", (dist_line >= 5_000) & (dist_line < 25_000)),
        (">= 25 km", dist_line >= 25_000),
    ]
    rows = {}
    for label, mask in bands:
        if mask.sum() < 3:
            rows[label] = {
                "r2": np.nan,
                "rmse": np.nan,
                "bias": np.nan,
            }
        else:
            rows[label] = {
                "r2": r2_score(observed[mask], predicted[mask]),
                "rmse": root_mean_squared_error(observed[mask], predicted[mask]),
                "bias": predicted[mask].sum() / observed[mask].sum(),
                "n": mask.sum(),
            }
    return pd.DataFrame.from_dict(rows, orient="index")


def _apply_and_route(
    graph: nx.MultiDiGraph,
    trips: pd.DataFrame,
    multiplier_features: dict[str, float] | None = None,
    additive_route_features: dict[str, float] | None = None,
    endpoint_features: dict[str, float] | None = None,
    baseline_duration_attr: str = "__baseline_duration",
    min_speed_kph: float = 1.0,
    max_speed_kph: float = 120.0,
    edge_duration_attr: str = "duration_calibrated",
    edge_feature_aggs: dict[str, str] | None = None,
    alpha: float = 1.0,
    const: float = 0.0,
):

    # Apply edge weights
    apply_edge_durations(
        graph,
        multiplier_features=multiplier_features,
        additive_route_features=additive_route_features,
        alpha=alpha,
        out_attr=edge_duration_attr,
        baseline_duration_attr=baseline_duration_attr,
        min_speed_kph=min_speed_kph,
        max_speed_kph=max_speed_kph,
    )

    # Route and collect travel times + features
    routed = routing.shortest_path_metrics_one_to_one(
        graph,
        list(trips.index),
        trips["nx_node_orig"],
        trips["nx_node_dest"],
        weight=edge_duration_attr,
        length_attr="length",
        edge_features=edge_feature_aggs,
    )

    routed["cost_net"] = routed["cost"]
    routed["cost"] = routed["cost_net"] + const

    # Apply origin/destination costs
    if not endpoint_features:
        return routed
    routed = routed.join(
        trips[["nx_node_orig", "nx_node_dest", "snap_dist_orig", "snap_dist_dest"]]
    )
    nodes_iter = graph.nodes(data=True)
    node_attrs = pd.DataFrame.from_dict(
        {n: {f: d.get(f, np.nan) for f in endpoint_features} for n, d in nodes_iter},
        orient="index",
    )
    features = []
    values = []
    for side in ("orig", "dest"):
        for f in endpoint_features:
            features.append(f"{f}_{side}")
            values.append(endpoint_features[f])
            # Snap distance was read directly from trips table
            if f == "snap_dist":
                continue
            routed[f"{f}_{side}"] = routed[f"nx_node_{side}"].map(node_attrs[f]).values
    routed["cost"] = routed["cost"] + (routed[features] * np.array(values)[np.newaxis, :]).sum(
        axis=1
    )

    return routed


def calibrate_edge_weights(
    graph: nx.MultiDiGraph,
    ground_truth: pd.DataFrame,
    *,
    baseline_speed_attr: str = "speed_kph",
    multiplier_features: dict[str, float] | None = None,
    additive_route_features: dict[str, float] | None = None,
    additive_endpoint_features: dict[str, float] | None = None,
    min_speed_kph: float = 1.0,
    max_speed_kph: float = 120.0,
    constant: float | None = None,
    n_iterations: int = 3,
    max_distance: float = 300.0,
    max_dist_to_line_ratio: float = 4.0,
    edge_duration_attr: str = "duration_calibrated",
    eligible_node_ids=None,
    eligible_node_flag: str | None = None,
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
        min_speed_kph: minimum edge speed (including node effects) in km/h.
        max_speed_kph: maximum edge speed (including node effects) in km/h.
        constant: include an intercept in the OLS fit.
        n_iterations: number of route-fit cycles. 2-3 usually converges.
        max_distance: drop trips where origin or destination is farther
            than this from any network node (metres).
        max_dist_to_line_ratio: if `dist_measured` is present, drop trips
            where `dist_measured / dist_line` exceeds this (long detours are
            usually data noise).
        edge_duration_attr: name of the per-edge duration attribute written
            on `graph` (overwritten each iteration).
        eligible_node_ids: optional set / list / Index of node IDs to restrict
            trip-endpoint snap targets to. Forwarded to
            `snap_to_network_nodes`. Typically `prepared.snap_eligible_nodes`
            from `routing_prep.prepare_network` — prevents trips from
            snapping to trapped nodes and contaminating the calibration fit.
        eligible_node_flag: alternative to `eligible_node_ids` — name of a
            per-node bool attribute on `graph` marking eligible snap targets
            (e.g., `prepared.snap_eligible_flag`). Ignored if
            `eligible_node_ids` is also given.

    Returns:
        `CalibrationResult` — see its docstring.

    Raises:
        ValueError: if any required column is missing or every trip filters
            out before fitting.
    """

    import statsmodels.api as sm

    multiplier_features = dict(multiplier_features or {})
    additive_route_features = dict(additive_route_features or {})
    additive_endpoint_features = dict(additive_endpoint_features or {})
    r2_tolerance = 0.0001

    # 1: Data preparation
    required = {"orig_x", "orig_y", "dest_x", "dest_y", "time_measured"}
    missing = required - set(ground_truth.columns)
    n_in = len(ground_truth)
    if missing:
        raise ValueError(f"`ground_truth` is missing required columns: {sorted(missing)}")
    if n_iterations < 1:
        raise ValueError("`n_iterations` must be at least 1")
    trips = ground_truth
    if "dist_line" not in trips.columns:
        trips = data_processing.add_straight_line_dist(trips)
    if "dist_measured" in trips.columns and max_dist_to_line_ratio:
        ratio = trips["dist_measured"] / trips["dist_line"]
        trips = trips[ratio < max_dist_to_line_ratio]
    elif max_dist_to_line_ratio:
        raise ValueError(
            "`ground_truth` is missing `dist_measured` (required for `max_dist_to_line_ratio`)"
        )

    # 2: Snap origins and destinations
    for side in ("orig", "dest"):
        geom = gpd.points_from_xy(trips[f"{side}_x"], trips[f"{side}_y"])
        points = gpd.GeoDataFrame(geometry=geom, index=trips.index)
        node_ids, dists = network_snap.snap_to_network_nodes(
            points,
            graph,
            max_distance=max_distance,
            eligible_node_ids=eligible_node_ids,
            eligible_node_flag=eligible_node_flag,
        )
        trips[f"nx_node_{side}"] = node_ids
        trips[f"snap_dist_{side}"] = dists

    trips = trips.dropna(subset=["nx_node_orig", "nx_node_dest"])
    if len(trips) == 0:
        raise ValueError("No trips remain after snap + filter.")
    logging.info(f"  {n_in:,} → {len(trips):,} trips left after filters.")

    alpha = 1.0
    cur_mult = dict(multiplier_features)
    cur_add = dict(additive_route_features)
    cur_end = dict(additive_endpoint_features)
    const = constant if constant is not None else 0

    # 3: write baseline duration
    baseline_duration_attr = "__baseline_duration"
    length = nx.get_edge_attributes(graph, "length")
    baseline_speed = nx.get_edge_attributes(graph, baseline_speed_attr)
    baseline_duration = {
        k: length[k] / (np.minimum(max_speed_kph, np.maximum(min_speed_kph, v)) * _KMH_TO_MS)
        for k, v in baseline_speed.items()
    }
    nx.set_edge_attributes(graph, baseline_duration, baseline_duration_attr)

    # Aggregation per feature: multiplier features get length-weighted-avg
    # (so they enter as a speed-like correction); additive route features
    # get summed along the path.
    edge_feature_aggs: dict[str, str] = {
        **{baseline_duration_attr: "sum"},
        **{f: "length_weighted" for f in multiplier_features},
        **{f: "sum" for f in additive_route_features},
    }
    r2_prev = 0.0

    # `final_*` track the latest fit; updated every iteration. Guarantees a
    # valid result even when convergence stops on iter 1 (e.g. when the
    # initial guess produces R² below `r2_tolerance` and the conditional
    # branch below doesn't fire).
    final_coefs: pd.DataFrame | None = None
    final_m_baseline = final_m_calib = final_m_regr = None

    for iteration in range(1, n_iterations + 2):
        # 4.1: Apply edge weights based on current coefficient values and route trips
        routed = _apply_and_route(
            graph,
            trips,
            cur_mult,
            cur_add,
            cur_end,
            baseline_duration_attr,
            min_speed_kph,
            max_speed_kph,
            edge_duration_attr,
            edge_feature_aggs,
            alpha,
            const,
        )

        # 4.2: Build OLS design matrix
        X, feat_cols, kinds = _build_predictors(
            routed,
            baseline_duration_attr,
            list(cur_mult),
            list(cur_add),
            list(cur_end),
            constant is not None,
        )

        # 4.3: Run OLS
        y = trips.loc[routed.index, "time_measured"]
        valid = X.notna().all(axis=1) & y.notna()
        X_f, y_f = X[valid], y[valid]
        fit_result = sm.OLS(y_f, X_f).fit()

        # 4.4: Gather error metrics
        dist_line = ground_truth.loc[valid.index, "dist_line"]
        m_baseline = _metrics_by_distance(y_f, routed[baseline_duration_attr][valid], dist_line)
        m_calib_net = _metrics_by_distance(y_f, routed["cost_net"][valid], dist_line)
        m_calib = _metrics_by_distance(y_f, routed["cost"][valid], dist_line)
        m_regr = _metrics_by_distance(y_f, fit_result.fittedvalues, dist_line)

        # 4.5: Record this iteration's result unconditionally — the
        # convergence check below decides whether to *continue*, but a valid
        # result is always available.
        final_coefs = pd.DataFrame({"coef": fit_result.params, "p": fit_result.pvalues}).round(3)
        final_m_baseline, final_m_calib, final_m_regr = m_baseline, m_calib, m_regr

        # 4.6: Decide whether to update coefficients for the next iteration.
        if iteration <= n_iterations and m_calib.at["all", "r2"] >= r2_prev + r2_tolerance:
            logging.info(
                f"  Iter {iteration}/{n_iterations}: "
                f"R² (baseline) = {m_baseline.at['all', 'r2']:.3f}, "
                f"R² (calibration, a-priori, net) = {m_calib_net.at['all', 'r2']:.3f}, "
                f"R² (calibration, a-priori, gross) = {m_calib.at['all', 'r2']:.3f}, "
                f"R² (regression) = {m_regr.at['all', 'r2']:.3f}, n={len(y_f):,}"
            )
            r2_prev = m_calib.at["all", "r2"]
            c = fit_result.params
            if constant is not None:
                const = float(c["const"])
            alpha = c["baseline_time"]
            for name in cur_mult:
                cur_mult[name] = float(c[name])
            for name in cur_add:
                cur_add[name] = float(c[name])
            for name in cur_end:
                # Average origin and destination impact
                cur_end[name] = (float(c[f"{name}_orig"]) + float(c[f"{name}_dest"])) / 2
            for name, r in final_coefs.iterrows():
                if name == "baseline_time":
                    avg = routed[baseline_duration_attr].mean()
                elif name == "const":
                    avg = const
                else:
                    avg = routed[name].mean()
                logging.info(f"    {name:.<20s}: {r['coef']:7.3f} (p={r['p']:.3g}, avg={avg:.3g})")
        else:
            logging.info(f".  Calibration completed after {iteration} iterations")
            for band in m_calib.index:
                logging.info(
                    f"    {band:.<10s}: {m_baseline.at[band, 'r2']:.3f} → {m_calib.at[band, 'r2']:.3f}"
                )
            break

    return CalibrationResult(
        coefficients=final_coefs,
        metrics_baseline=final_m_baseline,
        metrics_calibrated=final_m_calib,
        metrics_regression=final_m_regr,
        n_used=int(valid.sum()),
    )


# --- Traffic-counter calibration -----------------------------------------
#
# Calibration of a modeled traffic-flow estimate (e.g. the flows
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
    max_distance: float | pd.Series = 50.0,
    bearing_tol_deg: float = 20.0,
    bearing_column: str = "bearing_deg",
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
        max_distance: max cartesian distance for candidate edges (CRS
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
            f"(have: {list(counters.columns)})"
        )

    # Build edges GDF from graph. Drop edges without geometry — they can't
    # be snapped to anyway, and the caller is responsible for consolidating
    # / filling geometry beforehand.
    edge_records = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        geom = d.get("geometry")
        if geom is None:
            continue
        rec = dict(d)
        rec["u"], rec["v"], rec["k"] = u, v, k
        rec["geometry"] = geom
        edge_records.append(rec)
    if not edge_records:
        raise ValueError(
            "No edges have a `geometry` attribute. Consolidate the graph "
            "via `network_processing.consolidate_intersections` first, or "
            "use `osmnx.graph_to_gdfs(..., fill_edge_geometry=True)`."
        )
    edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=counters.crs)
    # Linear integer index so `map_points_to_filtered_lines`'s `line_id`
    # outputs lift back to (u, v, k) via a single .iloc lookup.
    edges_gdf = edges_gdf.reset_index(drop=True)

    def _accept(counter_row, edge_row, ctx) -> bool:
        edge_bearing = geo_processing.line_segment_bearing_at(edge_row.geometry, ctx["dist_along"])
        if np.isnan(edge_bearing):
            return False
        diff = geo_processing.angular_diff_deg(
            counter_row[bearing_column], edge_bearing, undirected=bidirectional
        )
        return float(diff) <= bearing_tol_deg

    matches = geo_mapping.map_points_to_filtered_lines(
        counters,
        edges_gdf,
        max_distance=max_distance,
        eligible_lines=eligible_edges,
        accept=_accept,
    )

    # Lift `line_id` (positional row in edges_gdf) back to (u, v, k).
    out = pd.DataFrame(index=counters.index)
    matched = matches["line_id"].notna()
    out["u"] = pd.NA
    out["v"] = pd.NA
    out["k"] = pd.NA
    if matched.any():
        idxs = matches.loc[matched, "line_id"].astype(int).to_numpy()
        out.loc[matched, "u"] = edges_gdf.iloc[idxs]["u"].to_numpy()
        out.loc[matched, "v"] = edges_gdf.iloc[idxs]["v"].to_numpy()
        out.loc[matched, "k"] = edges_gdf.iloc[idxs]["k"].to_numpy()
    out["snap_dist"] = matches["distance"]
    out["dist_along"] = matches["dist_along"]
    n_match = int(matched.sum())
    logging.info(
        f"snap_counters_to_edges: {n_match:,} of {len(counters):,} counters "
        f"matched ({n_match / max(len(counters), 1) * 100:.1f}%); "
        f"bidirectional={bidirectional}, tol={bearing_tol_deg}°."
    )
    return out


def evaluate_against_counters(
    modeled: pd.Series,
    counters: pd.DataFrame,
    *,
    observed_column: str = "traffic_cars",
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
    matched = counters.dropna(subset=["u", "v", "k"]).copy()
    if observed_column not in matched.columns:
        raise ValueError(
            f"`counters` is missing observed column `{observed_column!r}` "
            f"(have: {list(matched.columns)})"
        )
    # Build the index lookup. `modeled` may be a Series with a MultiIndex
    # or a tuple-keyed flat index — handle both via .reindex with tuples.
    keys = list(zip(matched["u"].astype(int), matched["v"].astype(int), matched["k"].astype(int)))
    if isinstance(modeled.index, pd.MultiIndex):
        modeled_values = modeled.reindex(keys).to_numpy()
    else:
        modeled_values = np.array([modeled.get(k, np.nan) for k in keys])
    matched["modeled"] = modeled_values
    matched["observed"] = matched[observed_column].astype(float)
    matched = matched.dropna(subset=["modeled", "observed"])
    if len(matched) == 0:
        return {"r2": np.nan, "slope": np.nan, "rmse": np.nan, "n_matched": 0, "merged": matched}

    obs = matched["observed"].to_numpy()
    mod = matched["modeled"].to_numpy()
    r = float(np.corrcoef(obs, mod)[0, 1]) if np.std(obs) > 0 and np.std(mod) > 0 else np.nan
    r2 = r**2 if not np.isnan(r) else np.nan
    # No-intercept regression: slope = Σ(x·y) / Σ(x²) with x=observed.
    denom = float((obs**2).sum())
    slope = float((obs * mod).sum() / denom) if denom > 0 else np.nan
    rmse = float(np.sqrt(((mod - obs) ** 2).mean()))
    return {
        "r2": r2,
        "slope": slope,
        "rmse": rmse,
        "n_matched": int(len(matched)),
        "merged": matched[["u", "v", "k", "observed", "modeled"]],
    }
