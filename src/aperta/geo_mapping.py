"""Map geographic features from one GeoDataFrame onto another.

Four primitives, distinguished by the shape of the source/target geometries:

- `map_points_to_polygons`: for each point, find the containing polygon
  (with optional nearest-polygon fallback for points just outside any
  polygon).
- `map_polygons_to_points`: for each polygon, find points falling inside.
- `map_points_to_points`: nearest-neighbor matching between two point sets.
- `map_points_to_filtered_lines`: snap points to the nearest line within
  a candidate-line subset selected by a per-point callback (used for
  bearing-aware traffic-counter snapping in `calibration`).

Distinct from `geo_processing`, which handles geometry construction and
raster sampling rather than mapping data between layers.
"""

import logging
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta import data_processing
from aperta.errors import DataError


def _sjoin_with_nearest_fallback(
    left_df: gpd.GeoDataFrame,
    right_df: gpd.GeoDataFrame,
    *,
    predicate: str,
    allow_nearest: bool,
    max_distance: float | None,
    left_label: str,
    right_label: str,
) -> tuple[pd.Series, pd.Series]:
    """Shared `sjoin + nearest-fallback` implementation for the public mapping
    functions. For each row in `left_df`, find a row in `right_df` matching
    `predicate` (`'within'` or `'contains'`); if `allow_nearest`, fall back to
    the nearest right-row for unmatched left-rows.

    Returns `(ids, distances)` indexed by `left_df.index`. `left_label` /
    `right_label` are used only for the informational log lines.
    """
    index_name = right_df.index.name
    dist_col = index_name + "_distance"
    res = gpd.sjoin(left_df[["geometry"]], right_df[["geometry"]], how="left", predicate=predicate)
    res = data_processing.remove_duplicate_indices(res)
    # sjoin doesn't create a distance column; initialise to NaN so we have
    # a uniform shape. Containment matches (`within` / `contains`) then get
    # distance = 0 (the natural "contained = zero proximity" reading),
    # nearest-fallback matches get their finite distance from `sjoin_nearest`
    # below, and truly unmatched rows keep NaN — three distinguishable states.
    res[dist_col] = np.nan
    matched = pd.notnull(res[index_name])
    res.loc[matched, dist_col] = 0.0
    logging.info(
        f"{matched.sum():,} of {len(left_df):,} {left_label} "
        f"({matched.sum() / len(left_df) * 100:.1f}%) allocated to containing "
        f"{right_label}."
    )
    if allow_nearest:
        res_nearest = gpd.sjoin_nearest(
            left_df[~matched][["geometry"]],
            right_df[["geometry"]],
            how="left",
            max_distance=max_distance,
            distance_col=dist_col,
        )
        res_nearest = data_processing.remove_duplicate_indices(res_nearest)
        nearest_matched = pd.notnull(res_nearest[index_name])
        denom = max((~matched).sum(), 1)
        logging.info(
            f"{nearest_matched.sum():,} of {(~matched).sum():,} remaining "
            f"{left_label} ({nearest_matched.sum() / denom * 100:.1f}%) allocated "
            f"to nearest {right_label}."
        )
        res.loc[~matched, index_name] = res_nearest[index_name]
        res.loc[~matched, dist_col] = res_nearest[dist_col]
    unmatched = pd.isnull(res[index_name])
    denom = max((~matched).sum(), 1)
    logging.info(
        f"{unmatched.sum():,} {left_label} ({unmatched.sum() / denom * 100:.1f}%) not allocated."
    )
    if not res.index.equals(left_df.index):
        raise DataError(
            f"Index of spatial mapping result does not match original {left_label} index."
        )
    count_unique = len(res[index_name].unique())
    logging.info(
        f"{count_unique:,} ({count_unique / len(right_df) * 100:.1f}%) of "
        f"{right_label} indices present among matched {left_label}."
    )
    res[index_name] = res[index_name].astype(right_df.index.dtype)
    return res[index_name], res[dist_col]


def map_points_to_polygons(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    allow_nearest: bool = True,
    max_distance: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """For each point, find the polygon containing it.

    With `allow_nearest=True`, points not within any polygon fall back to the
    nearest polygon — capped at `max_distance` if given (CRS units).

    See also `map_polygons_to_points` for the inverse mapping (each polygon →
    a point it contains).

    Args:
        points: GeoDataFrame of points to assign. Output is indexed by
            `points.index`.
        polygons: GeoDataFrame of candidate polygons.
        allow_nearest: fallback to nearest polygon when a point isn't inside
            any polygon.
        max_distance: distance cap for the nearest-fallback (CRS units).
            `None` = no cap.

    Returns:
        `(ids, distances)` tuple, both indexed by `points.index`:
            - `ids`: the matched polygon ID for each point. NaN where no match
              (e.g. point outside all polygons and `allow_nearest=False`, or
              beyond `max_distance`).
            - `distances`: distance to the matched polygon (CRS units).
              `0.0` for points matched by "within" (inside the polygon —
              natural "contained = zero proximity" reading); finite
              positive values for points assigned via nearest-fallback;
              NaN only for truly unmatched points (`allow_nearest=False`
              or beyond `max_distance`).

    Points that match multiple polygons (overlapping) take the first match.
    """
    return _sjoin_with_nearest_fallback(
        left_df=points,
        right_df=polygons,
        predicate="within",
        allow_nearest=allow_nearest,
        max_distance=max_distance,
        left_label="points",
        right_label="polygon",
    )


def map_polygons_to_points(
    polygons: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    *,
    allow_nearest: bool = True,
    max_distance: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """For each polygon, find a point it contains.

    With `allow_nearest=True`, polygons that contain no point fall back to the
    nearest point — capped at `max_distance` if given (CRS units). Polygons
    containing multiple points keep the first match.

    See also `map_points_to_polygons` for the inverse mapping (each point →
    the polygon containing it).

    Args:
        polygons: GeoDataFrame of polygons to assign. Output is indexed by
            `polygons.index`.
        points: GeoDataFrame of candidate points.
        allow_nearest: fallback to nearest point when a polygon contains no
            point.
        max_distance: distance cap for the nearest-fallback (CRS units).
            `None` = no cap.

    Returns:
        `(ids, distances)` tuple, both indexed by `polygons.index`:
            - `ids`: the matched point ID for each polygon. NaN where no
              match (no contained point and either `allow_nearest=False` or
              beyond `max_distance`).
            - `distances`: distance from the polygon to the matched point
              (CRS units). `0.0` for polygons matched by "contains" (point
              inside the polygon — natural "contained = zero proximity"
              reading); finite positive values for polygons assigned via
              nearest-fallback; NaN only for truly unmatched polygons
              (`allow_nearest=False` or beyond `max_distance`).
    """
    return _sjoin_with_nearest_fallback(
        left_df=polygons,
        right_df=points,
        predicate="contains",
        allow_nearest=allow_nearest,
        max_distance=max_distance,
        left_label="polygons",
        right_label="point",
    )


def map_points_to_points(
    left_points: gpd.GeoDataFrame | gpd.GeoSeries,
    right_points: gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    max_distance: float | None = None,
    verbose: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """For each point in `left_points`, find the nearest point in `right_points`.

    For the common special case where `right_points` are the nodes of a
    `networkx` graph (or similar), use
    `aperta.network_snap.snap_to_network_nodes` instead — it takes the
    graph directly, skipping the boilerplate of extracting node `(x, y)`
    coordinates and wrapping them in a GeoDataFrame.

    Args:
        left_points: GeoDataFrame or GeoSeries of query points. Output is
            indexed by `left_points.index`.
        right_points: GeoDataFrame or GeoSeries of candidate points.
        max_distance: distance cap (CRS units). `None` = no cap. Query points
            with no match within `max_distance` get NaN ID and NaN distance.

    Returns:
        `(ids, distances)` tuple, both indexed by `left_points.index`:
            - `ids`: the nearest-point ID from `right_points`. NaN if no match
              within `max_distance`.
            - `distances`: distance to the matched point (CRS units). NaN if no match.
    """
    # Coerce GeoSeries → single-column GeoDataFrame so the `df[["geometry"]]`
    # calls below work uniformly (on a GeoSeries, `[["geometry"]]` is a
    # row-label lookup and raises KeyError).
    if isinstance(left_points, gpd.GeoSeries):
        left_points = left_points.to_frame(name="geometry")
    if isinstance(right_points, gpd.GeoSeries):
        right_points = right_points.to_frame(name="geometry")
    index_name = right_points.index.name
    dist_col = index_name + "_distance"
    res = gpd.sjoin_nearest(
        left_points[["geometry"]],
        right_points[["geometry"]],
        how="left",
        max_distance=max_distance,
        distance_col=dist_col,
    )
    res = data_processing.remove_duplicate_indices(res)
    f = pd.notnull(res[index_name])
    count_unique = len(res[index_name].unique())
    if verbose:
        logging.info(
            f"{f.sum():,} of {len(left_points):,} points in `left_points` found a match "
            f"in `right_points`."
        )
        logging.info(
            f"{count_unique:,} ({count_unique / len(right_points) * 100:.1f}%) of points in "
            f"`right_points` are present in `left_points`."
        )
    return res[index_name], res[dist_col]


def map_points_to_filtered_lines(
    points: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    *,
    max_distance: float | pd.Series,
    eligible_lines: Callable[[pd.Series, gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None,
    accept: Callable[[pd.Series, pd.Series, dict], bool] | None = None,
) -> pd.DataFrame:
    """For each point, find the nearest line that passes per-point filters.

    Matching is sequential per point: candidate lines within `max_distance`
    are optionally pre-filtered, sorted by cartesian distance to the point,
    then walked in increasing-distance order. The first line `accept` returns
    `True` for is the match. If `accept` is `None`, the nearest eligible line
    is matched.

    Args:
        points: GeoDataFrame of point geometries.
        lines: GeoDataFrame of LineString / MultiLineString geometries. Must
            be in the same CRS as `points`.
        max_distance: max distance (CRS units) for candidate lines. Scalar
            applies to all points; pass a `pd.Series` aligned to `points.index`
            for per-point radii (e.g. a wider radius for highway counters).
        eligible_lines: optional per-point pre-filter
            `(point_row, candidate_lines) -> eligible_lines`. Receives only
            candidates already within `max_distance` of the point; should
            return a subset (typically `candidate_lines[mask]`).
        accept: optional per-candidate acceptance test
            `(point_row, line_row, ctx) -> bool`, where `ctx` is a dict with
            `'distance'` (cartesian point-to-line), `'dist_along'` (arclength
            along the line to the point closest to `point`), and
            `'nearest_point'` (the shapely Point on the line nearest `point`).
            Walked in order of increasing `distance`; the first `True` is the
            match. `None` = accept the nearest eligible line.

    Returns:
        DataFrame indexed like `points` with three columns:
          - `line_id`: index of matched line (or `pd.NA` if no match)
          - `distance`: cartesian distance to matched line (or `NaN`)
          - `dist_along`: distance along matched line to nearest point (or `NaN`)
    """
    if isinstance(max_distance, (int, float)):
        radii = pd.Series(float(max_distance), index=points.index)
    else:
        radii = max_distance.astype(float)
        if not radii.index.equals(points.index):
            raise DataError("`max_distance` Series must share index with `points`.")
    if points.crs != lines.crs:
        raise DataError(f"`points` CRS {points.crs} does not match `lines` CRS {lines.crs}.")

    sindex = lines.sindex
    pt_indices, line_ids, dists, alongs = [], [], [], []

    for pt_idx, pt_row in points.iterrows():
        pt_geom = pt_row.geometry
        r = radii.loc[pt_idx]
        # All lines within `r` of the point (cartesian).
        pos = sindex.query(pt_geom, predicate="dwithin", distance=r)
        matched_id, matched_dist, matched_along = pd.NA, np.nan, np.nan
        if len(pos):
            candidates = lines.iloc[pos]
            if eligible_lines is not None:
                candidates = eligible_lines(pt_row, candidates)
            if len(candidates):
                cand_dists = candidates.geometry.distance(pt_geom).values
                order = np.argsort(cand_dists, kind="stable")
                for k in order:
                    d = float(cand_dists[k])
                    line_row = candidates.iloc[k]
                    line_geom = line_row.geometry
                    dist_along = float(line_geom.project(pt_geom))
                    if accept is not None:
                        ctx = {
                            "distance": d,
                            "dist_along": dist_along,
                            "nearest_point": line_geom.interpolate(dist_along),
                        }
                        if not accept(pt_row, line_row, ctx):
                            continue
                    matched_id = candidates.index[k]
                    matched_dist = d
                    matched_along = dist_along
                    break
        pt_indices.append(pt_idx)
        line_ids.append(matched_id)
        dists.append(matched_dist)
        alongs.append(matched_along)

    out = pd.DataFrame(
        {"line_id": line_ids, "distance": dists, "dist_along": alongs},
        index=pd.Index(pt_indices, name=points.index.name),
    )
    n_matched = out["line_id"].notna().sum()
    logging.info(
        f"Matched {n_matched:,} of {len(points):,} points "
        f"({n_matched / max(len(points), 1) * 100:.1f}%)."
    )
    return out
