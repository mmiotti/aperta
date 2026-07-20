"""Geometric and raster-sampling utilities used across aperta.

Four broad concerns:

1. **Geometric helpers** — building hectare / H3 grids, generating point or
   polygon geometries from coordinate data, projecting between coordinate
   reference systems, computing per-cell spatial neighborhoods. Distinct
   from `geo_mapping`, which is specifically about mapping data between
   geo units.

2. **Geometry math** — `line_bearings_deg`, `line_segment_bearing_at`,
   `angular_diff_deg`, `simplify_geometry`. Operations whose inputs and
   outputs are geometries (or geometry-derived numerics like bearings).

3. **Raster sampling** — `sample_raster_at_points` reads values from a
   raster file at given point coordinates (used for elevation lookups).
   Requires the optional `topo` extra (`rasterio`).

4. **DEM fetch** — `fetch_copernicus_dem` downloads, mosaics, clips, and
   optionally reprojects Copernicus GLO-30 DEM tiles for a given polygon.
   Requires the optional `topo` extras (`rasterio`, `requests`).

For purely tabular helpers (column transforms, aggregation rules, metric
discovery), see `data_processing`.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


def _line_segments(line) -> np.ndarray | None:
    """Stack of `(dx, dy, length)` rows for every segment, across all parts.

    Handles both `LineString` and `MultiLineString` (multipart parts are
    concatenated in stored order). Empty / degenerate inputs return `None`.
    """
    if line is None or line.is_empty:
        return None
    parts = list(line.geoms) if hasattr(line, "geoms") else [line]
    rows = []
    for part in parts:
        if part is None or part.is_empty:
            continue
        coords = np.asarray(part.coords)
        if len(coords) < 2:
            continue
        deltas = np.diff(coords[:, :2], axis=0)
        seg_lens = np.hypot(deltas[:, 0], deltas[:, 1])
        rows.append(np.column_stack([deltas, seg_lens]))
    if not rows:
        return None
    return np.vstack(rows)


def _bearing_at(segs: np.ndarray, dist_along: float) -> float:
    """Compass bearing of the segment containing `dist_along` in a `_line_segments` array."""
    seg_lens = segs[:, 2]
    total = seg_lens.sum()
    if total == 0:
        return float("nan")
    pos = max(0.0, min(float(dist_along), total))
    i = int(np.searchsorted(np.cumsum(seg_lens), pos, side="right"))
    i = min(i, len(seg_lens) - 1)
    dx, dy = segs[i, 0], segs[i, 1]
    if dx == 0 and dy == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def line_bearings_deg(geoms: gpd.GeoSeries) -> pd.Series:
    """Compass bearing (degrees, [0, 360)) at each line's midpoint.

    For each (Multi)LineString, the polyline segment containing the
    half-arclength point is identified, and the bearing of *that segment* is
    returned — i.e. the local direction of travel at the midpoint, not the
    first→last chord. For straight single-segment lines the two are identical;
    for curved polylines they diverge. MultiLineString parts are treated as if
    concatenated end-to-end in their stored order.

    `0°` = +y (North), increasing clockwise (`90°` = East, `180°` = South,
    `270°` = West). The bearing reflects the line's stored vertex order — the
    line A→B has the opposite bearing of B→A. For undirected geometries take
    `bearings % 180`.

    Lines must be in a projected CRS with x = easting, y = northing (e.g.
    LV95 / EPSG:2056); a geographic CRS would introduce longitude-convergence
    error. Degenerate or empty geometries (no length, missing, or fewer than 2
    vertices) return `NaN`.
    """

    def _bearing(line) -> float:
        segs = _line_segments(line)
        if segs is None:
            return float("nan")
        return _bearing_at(segs, segs[:, 2].sum() / 2)

    return pd.Series([_bearing(g) for g in geoms], index=geoms.index, dtype="float64")


def line_segment_bearing_at(line, dist_along: float) -> float:
    """Compass bearing of the polyline segment containing an arbitrary `dist_along`.

    Same convention as `line_bearings_deg` but at any arclength position, not
    just the midpoint. Useful after snapping a point to a line, when you want
    the line's *local* direction at the snap point (e.g. to compare a traffic
    counter's bearing against the local edge bearing).

    `dist_along` is in the line's CRS units (meters in LV95). Clamped to
    `[0, line.length]`. MultiLineString parts are treated as concatenated
    end-to-end. Empty / degenerate inputs return `NaN`.
    """
    segs = _line_segments(line)
    if segs is None:
        return float("nan")
    return _bearing_at(segs, dist_along)


def angular_diff_deg(a, b, undirected: bool = False):
    """Smallest unsigned difference (in degrees) between two compass bearings.

    Default (`undirected=False`): treats inputs as directions of travel —
    returns the circular distance in `[0, 180]`. E.g. `angular_diff_deg(45, 350)`
    is `55`, not `305`; opposing directions (90° vs 270°) give `180`.

    `undirected=True`: treats inputs as orientations of an undirected line —
    additionally folds across 180°, so A→B and B→A are equivalent. Returns
    `[0, 90]`. Use this when matching e.g. a directed traffic counter to an
    undirected road segment that carries traffic both ways: a counter bearing
    `90°` aligns with edges at *both* `90°` and `270°`.

    Accepts scalars or aligned numpy/pandas arrays; return type follows
    `np.minimum` (scalar in → scalar-like out).
    """
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    diff = np.minimum(d, 360.0 - d)
    if undirected:
        diff = np.minimum(diff, 180.0 - diff)
    return diff


def sum_within_radius(
    gdf: gpd.GeoDataFrame,
    cols: list[str],
    radius: int | float,
    *,
    return_densities: bool = False,
    add_filled_densities: bool = False,
) -> pd.DataFrame:
    """Same-set neighbourhood sum: for each cell in `gdf`, sum the values
    of `cols` over all cells (including itself) within `radius`.

    For the cross-set case (`targets × sources` with two different
    GeoDataFrames) use [[cross_sum_within_radius]] instead.

    The regular sum, divided by the circular query area when
    `return_densities=True`, gives per-km² density. The "filled" variant
    additionally divides by the *occupied* area (cells where at least one
    value in `cols` is > 0) — useful where occupancy is sparse and the
    circular-area normaliser dilutes the signal (e.g. employment
    concentrated at an airport).

    The CRS of `gdf` must represent meters if densities are returned.
    Densities are returned per km².

    Implementation: builds a `scipy.spatial.KDTree` on cell centroids and
    constructs a sparse 0/1 adjacency matrix from `query_ball_point(radius)`;
    each cell's own row is implicitly included (since `dist(p, p) = 0 ≤ radius`).
    Per-cell sums are then `adj @ values` in one vectorised pass.
    """
    if not return_densities and add_filled_densities:
        raise ValueError("`add_filled_densities` can only be True when `return_densities` is True.")

    from scipy.sparse import csr_matrix
    from scipy.spatial import KDTree

    centroids = gdf.geometry.centroid
    xy = np.column_stack([centroids.x.to_numpy(dtype=float), centroids.y.to_numpy(dtype=float)])
    tree = KDTree(xy)
    idx_lists = tree.query_ball_point(xy, r=radius)

    # Sparse 0/1 adjacency matrix: row i has 1s at the neighbour positions
    # of cell i (within `radius`, inclusive of i itself). `adj @ values`
    # then computes per-cell spatial sums in one vectorised pass.
    n = len(gdf)
    row_lens = np.fromiter((len(nbrs) for nbrs in idx_lists), dtype=np.int64, count=n)
    cols_idx = np.concatenate([np.asarray(nbrs, dtype=np.int64) for nbrs in idx_lists])
    rows = np.repeat(np.arange(n, dtype=np.int64), row_lens)
    data = np.ones(len(rows), dtype=float)
    adj = csr_matrix((data, (rows, cols_idx)), shape=(n, n), dtype=float)

    values = gdf[cols].to_numpy(dtype=float)
    spatial_sums = np.asarray(adj @ values)

    res = pd.DataFrame(index=gdf.index)
    circular_area_km2 = (np.pi * radius**2) / 1e6
    out_values = spatial_sums / circular_area_km2 if return_densities else spatial_sums
    df = pd.DataFrame(out_values, index=gdf.index, columns=[f"{col}_r{radius}" for col in cols])
    res = res.join(df)

    if add_filled_densities:
        # "Filled" densities: divide each cell's spatial sum by the total
        # area of *occupied* cells within its neighbourhood, instead of the
        # circular query area. Useful where occupancy is sparse — e.g.
        # employment concentrated at an airport with empty cells around.
        cell_areas_km2 = gdf.geometry.area.to_numpy(dtype=float) / 1e6
        if not np.any(cell_areas_km2 > 0):
            raise ValueError(
                "`add_filled_densities=True` requires polygon geometries with "
                "non-zero area. Got geometries with all zero area — likely "
                "points. Compute filled densities on a polygon layer (e.g. "
                "cells), then aggregate the result onto the point layer "
                "(e.g. via snap) if needed."
            )
        if len(np.unique(cell_areas_km2.round(decimals=4))) > 1:
            raise NotImplementedError(
                "Cells must currently be same size to calculate filled densities."
            )
        is_filled = (gdf[cols].sum(axis=1) > 0).to_numpy()
        # `adj @ is_filled` = per-cell count of filled neighbours.
        n_filled_neighbours = np.asarray(adj @ is_filled.astype(float))
        fill_cell_areas = cell_areas_km2 * n_filled_neighbours
        f = fill_cell_areas > 0
        filled_values = spatial_sums.copy()
        filled_values[f, :] = filled_values[f, :] / fill_cell_areas[f, np.newaxis]
        filled_values[~f, :] = 0
        df_filled = pd.DataFrame(
            filled_values, index=gdf.index, columns=[f"{col}_r{radius}_filled" for col in cols]
        )
        res = res.join(df_filled)

    res = res.fillna(0)
    return res


def count_within_radius(
    gdf: gpd.GeoDataFrame,
    radius: int | float,
) -> pd.Series:
    """Same-set neighbourhood count: for each cell in `gdf`, the number of
    cells (including itself) within `radius`. Useful as a denominator
    for neighbourhood-mean / variance computations and as a
    "neighbourhood crowding" feature in its own right.

    Mirrors [[sum_within_radius]]'s machinery — builds a
    `scipy.spatial.KDTree` on cell centroids and queries
    `query_ball_point(radius)`; the per-cell count is the length of each
    returned neighbour list. Each cell's own row is always counted
    (since `dist(p, p) = 0 ≤ radius`), so the minimum returned value is 1.

    Returns a `pd.Series` of `int64` counts indexed by `gdf.index`,
    named `count_r{radius}`.
    """
    from scipy.spatial import KDTree

    centroids = gdf.geometry.centroid
    xy = np.column_stack([centroids.x.to_numpy(dtype=float), centroids.y.to_numpy(dtype=float)])
    tree = KDTree(xy)
    idx_lists = tree.query_ball_point(xy, r=radius)

    counts = np.fromiter(
        (len(nbrs) for nbrs in idx_lists),
        dtype=np.int64,
        count=len(gdf),
    )
    return pd.Series(counts, index=gdf.index, name=f"count_r{radius}")


def cross_sum_within_radius(
    targets: gpd.GeoDataFrame | gpd.GeoSeries,
    sources: gpd.GeoDataFrame | gpd.GeoSeries,
    radius: float,
    *,
    weight_column: str | None = None,
    return_density: bool = False,
    name: str = "aggregate",
) -> pd.Series:
    """Cross-set neighbourhood sum: for each target geometry, count source
    geometries (or sum a `weight_column` over them) within `radius` of the
    target's centroid.

    For the same-set case (one `gdf`, queried against itself with
    self-inclusion, multiple value columns at once, optional "filled-area"
    densities) use [[sum_within_radius]] instead. The two helpers are
    deliberately separate: same-set and cross-set queries differ in input
    shape (one gdf vs two) and in optimal backend (single sparse-matrix
    multiply vs per-target KDTree query).

    Backend: scipy `cKDTree` on source coordinates — O(N_src log N_src) one-
    time build, O(log N_src + k) per query.

    Both `targets` and `sources` must share a metric CRS for densities (and
    the radius) to be meaningful in real-world units. The function uses the
    centroid of each geometry, so non-point inputs (polygons, lines) work
    fine — their centroid is queried.

    Args:
        targets: GeoDataFrame or GeoSeries. Centroid of each geometry is
            the query point; `targets.index` becomes the output Series index.
        sources: GeoDataFrame or GeoSeries. Centroid of each source is the
            point that's counted / summed. If `weight_column` is given,
            `sources` must be a GeoDataFrame carrying that column.
        radius: query radius in CRS units (typically metres).
        weight_column: name of the column on `sources` to sum. `None`
            (default) counts source geometries within radius instead.
        return_density: if True, divide aggregates by `π · radius²` (the
            circular query area). Caller is responsible for ensuring CRS
            units are metric.
        name: name for the returned Series.

    Returns:
        `pd.Series` indexed by `targets.index`. Targets with no sources in
        range get `0` (or `0.0` density).
    """
    from scipy.spatial import KDTree

    src_centroids = sources.geometry.centroid
    src_xy = np.column_stack(
        [src_centroids.x.to_numpy(dtype=float), src_centroids.y.to_numpy(dtype=float)]
    )
    tgt_centroids = targets.geometry.centroid
    tgt_xy = np.column_stack(
        [tgt_centroids.x.to_numpy(dtype=float), tgt_centroids.y.to_numpy(dtype=float)]
    )
    tree = KDTree(src_xy)

    if weight_column is None:
        # `return_length=True` (scipy ≥1.6) skips materialising the per-query
        # index lists — much faster for large source sets.
        agg = np.asarray(
            tree.query_ball_point(tgt_xy, r=radius, return_length=True),
            dtype=float,
        )
    else:
        if not isinstance(sources, gpd.GeoDataFrame) or weight_column not in sources.columns:
            raise ValueError(
                f"`sources` must be a GeoDataFrame containing column "
                f"{weight_column!r} when `weight_column` is given."
            )
        weights = sources[weight_column].to_numpy(dtype=float)
        idx_lists = tree.query_ball_point(tgt_xy, r=radius)
        agg = np.fromiter(
            (float(weights[idxs].sum()) if len(idxs) else 0.0 for idxs in idx_lists),
            dtype=float,
            count=len(idx_lists),
        )

    if return_density:
        agg = agg / (np.pi * radius**2)
    return pd.Series(agg, index=targets.index, name=name)


def sample_raster_at_points(
    points: gpd.GeoDataFrame | gpd.GeoSeries,
    raster_path,
    *,
    band: int = 1,
    method: str = "bilinear",
    name: str = "raster_value",
) -> pd.Series:
    """Sample raster values at point centroids — vectorized, single-pass.

    Reads the raster once and indexes into the array with either bilinear
    interpolation (default — right for continuous fields like elevation;
    smooths pixel-boundary quantization noise) or nearest-neighbour (for
    categorical rasters like land cover or zoning, where blending values
    is meaningless).

    For ~100k+ points this is orders of magnitude faster (and far lighter
    on memory) than `rasterio.Dataset.sample`, which does a GDAL
    `RasterIO()` call per point and is the bottleneck inside
    `osmnx.elevation.add_node_elevations_raster`.

    Points whose `(row, col)` falls outside the raster — or whose pixel
    equals the raster's `nodata` value — get `NaN`. (For `'bilinear'`,
    any of the 4 surrounding pixels being nodata propagates to `NaN`.)
    The caller is responsible for ensuring `points.crs` matches the
    raster CRS; the function does not reproject. Polygon / line inputs
    are sampled at their centroid.

    Args:
        points: GeoDataFrame or GeoSeries; sampling location is each
            geometry's centroid; `points.index` becomes the output index.
        raster_path: path to a single-band raster (GeoTIFF, VRT, etc.).
        band: 1-based band index.
        method: `'bilinear'` (default) interpolates the 4 surrounding
            pixels weighted by fractional distance; `'nearest'` returns
            the value of the pixel containing the point.
        name: name for the returned Series.

    Returns:
        `pd.Series` of floats indexed by `points.index`.
    """
    import rasterio
    from rasterio.windows import Window, from_bounds

    # Skip the `.centroid` op when inputs are already points — the centroid
    # of a Point is the Point itself, so the op is a no-op, but on geographic
    # CRSes (WGS84) geopandas warns about it unconditionally. False positive
    # for points; suppress by short-circuiting.
    geom = points.geometry
    if (geom.geom_type == "Point").all():
        sample_geom = geom
    else:
        sample_geom = geom.centroid
    xs = sample_geom.x.to_numpy(dtype=float)
    ys = sample_geom.y.to_numpy(dtype=float)
    if len(xs) == 0:
        return pd.Series([], index=points.index, name=name, dtype=float)

    # Windowed read: only load the raster area covering the query bounding
    # box. For DEMs that cover a much larger area than the query set
    # (regional / national / global rasters used to sample a local point
    # set), this drops `arr` from a large fraction of the file to just
    # what's needed. When the file already matches the query extent, the
    # window covers the whole raster — same as the historic full read,
    # no overhead.
    with rasterio.open(raster_path) as src:
        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.any():
            xs_f, ys_f = xs[finite], ys[finite]
            requested = from_bounds(
                float(xs_f.min()),
                float(ys_f.min()),
                float(xs_f.max()),
                float(ys_f.max()),
                src.transform,
            )
            # Snap to integer pixel boundaries — `from_bounds` returns a
            # fractional window, and `src.read(window=...)` /
            # `src.window_transform(...)` behave more predictably with
            # integer offsets. Floor the offset, ceil the far edge, then
            # clip to the actual raster extent.
            col_start = max(0, int(np.floor(requested.col_off)))
            row_start = max(0, int(np.floor(requested.row_off)))
            col_stop = min(src.width, int(np.ceil(requested.col_off + requested.width)))
            row_stop = min(src.height, int(np.ceil(requested.row_off + requested.height)))
            if col_stop > col_start and row_stop > row_start:
                window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
            else:
                # Bounding box is entirely outside the raster — read nothing.
                window = Window.from_slices((0, 0), (0, 0))
        else:
            # All query coords NaN → no valid samples; read nothing.
            window = Window.from_slices((0, 0), (0, 0))

        arr = src.read(band, window=window).astype(float)
        arr_transform = src.window_transform(window)
        inv_transform = ~arr_transform
        nodata = src.nodata

    height, width = arr.shape

    # Mask nodata up-front so both branches see NaN — bilinear's weighted
    # sum will then naturally propagate NaN when any contributing pixel
    # is invalid.
    if nodata is not None and arr.size > 0:
        arr = np.where(arr == nodata, np.nan, arr)

    # Affine `~transform` maps (x, y) -> (col_float, row_float). The +/-0.5
    # shift converts from pixel-corner to pixel-centre coordinates, which
    # is what bilinear interpolation expects: a point at the pixel's
    # centre should return that pixel's value exactly, not a blend.
    cols_f, rows_f = inv_transform * (xs, ys)

    out = np.full(len(points), np.nan, dtype=float)
    if method == "nearest":
        cols = np.floor(cols_f).astype(int)
        rows = np.floor(rows_f).astype(int)
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if valid.any():
            out[valid] = arr[rows[valid], cols[valid]]
    elif method == "bilinear":
        from scipy.ndimage import map_coordinates

        # `map_coordinates` expects (row, col) at pixel CENTRES (an integer
        # coord = that pixel's centre). Subtract 0.5 to convert from corner-
        # based pixel coords. mode='constant', cval=nan returns NaN out-of-
        # bounds; NaN inputs propagate to NaN outputs via the weighted sum.
        valid = (cols_f >= 0) & (cols_f <= width) & (rows_f >= 0) & (rows_f <= height)
        if valid.any():
            coords = np.vstack([rows_f[valid] - 0.5, cols_f[valid] - 0.5])
            out[valid] = map_coordinates(arr, coords, order=1, mode="constant", cval=np.nan)
    else:
        raise ValueError(f"method must be 'bilinear' or 'nearest', got {method!r}")

    return pd.Series(out, index=points.index, name=name)


def build_h3_grid(
    polygon,
    resolution: int,
    *,
    polygon_crs: str = "EPSG:4326",
    target_crs: str | None = None,
    id_column: str = "cell_id",
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of H3 hex cells covering `polygon`.

    One row per H3 cell whose centre lies inside `polygon`. Geometry is the
    closed hex polygon. The H3 library operates in EPSG:4326 (WGS84); the
    input polygon is reprojected if `polygon_crs != 'EPSG:4326'`, and the
    output is optionally reprojected to `target_crs`.

    For nested tiers (e.g. cells at H3 res 10, zones at res 8), call once
    per resolution OR derive the coarser tier by `h3.cell_to_parent` on the
    finer one (cheaper, and the nesting is exact). Pure replacement of the
    private
    `_h3_cells_for_polygon` / `_h3_cell_to_polygon` helpers seen across
    aperta example notebooks.

    Args:
        polygon: input area as a shapely Polygon or MultiPolygon.
        resolution: H3 resolution (0–15). Lower = larger cells.
        polygon_crs: CRS of the input polygon (default `'EPSG:4326'`).
        target_crs: optional output CRS (cells are reprojected if given).
        id_column: name for the H3 cell-ID index (default `'cell_id'`).

    Returns:
        GeoDataFrame indexed by H3 cell ID (string), with a `geometry`
        column. CRS is `target_crs` if given, else `'EPSG:4326'`.
    """
    import h3
    from shapely.geometry import Polygon as ShPolygon

    if polygon_crs != "EPSG:4326":
        polygon = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs("EPSG:4326").iloc[0]

    parts = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    cell_ids: list[str] = []
    for part in parts:
        # shapely uses (lng, lat); H3 wants (lat, lng).
        exterior = [(lat, lng) for lng, lat in part.exterior.coords]
        interiors = [[(lat, lng) for lng, lat in ring.coords] for ring in part.interiors]
        h3_poly = h3.LatLngPoly(exterior, *interiors)
        cell_ids.extend(h3.h3shape_to_cells(h3_poly, resolution))

    cell_ids = sorted(set(cell_ids))
    geoms = [ShPolygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cell_ids]
    gdf = gpd.GeoDataFrame(
        {"geometry": geoms},
        index=pd.Index(cell_ids, name=id_column),
        crs="EPSG:4326",
    )
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf


def get_hectare_geometries(df: pd.DataFrame, crs: str) -> gpd.GeoSeries:
    """Get 100m hectare square geometries.

    `crs` is the CRS of the (x, y) coordinates in `df.index` (e.g. 'EPSG:2056' for
    Swiss LV95). Library function — does not assume a particular country/CRS key.
    """
    geom = gpd.points_from_xy(
        df.index.get_level_values(0),
        df.index.get_level_values(1),
        crs=crs,
    ).buffer(50, cap_style=3)
    return geom


def simplify_geometry(
    geometry: shapely.LineString,
    target_size: int,
    maximum_size: int,
    max_precision_m: float = 10.0,
    min_precision_m: float = 250.0,
) -> tuple[shapely.LineString, int, float]:
    """Simplify `geometry` toward `target_size` vertices using a per-line precision.

    Returns `(simplified, new_size, new_size / old_size)`. Precision is bounded
    between `max_precision_m` and `min_precision_m`. Logs a warning if the
    result still exceeds `maximum_size`.

    Assumes input is in a lat/lon CRS (EPSG:4326). Uses the 0.00001-degree ≈ 1 m
    approximation appropriate for central-European latitudes.
    """
    old_size = len(geometry.xy[0])
    total_length_m_approx = geometry.length / 0.00001
    target_precision_m = max(
        max_precision_m, min(total_length_m_approx / target_size, min_precision_m)
    )
    precision = 0.00001 * target_precision_m
    new_geometry = geometry.simplify(precision)
    new_size = len(new_geometry.xy[0])
    if new_size > maximum_size:
        logging.warning(
            f"Could not reduce geometry to target size with precision "
            f"{precision:.5f}: {maximum_size} ({new_size:,})"
        )
    return new_geometry, new_size, new_size / old_size


_COPERNICUS_DEM_URL_TEMPLATE = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def _copernicus_tile_name(lat: int, lng: int) -> str:
    """Build a Copernicus GLO-30 DEM tile name using its SW-corner
    convention. Hemispheres are encoded as letter prefixes:
    `N`/`S` for lat ≥ 0 / lat < 0, `E`/`W` for lng ≥ 0 / lng < 0.
    Magnitudes are zero-padded (lat to 2 digits, lng to 3).

    Examples:
        lat=51, lng=0   → 'Copernicus_DSM_COG_10_N51_00_E000_00_DEM'
        lat=51, lng=-1  → 'Copernicus_DSM_COG_10_N51_00_W001_00_DEM'
        lat=-1, lng=5   → 'Copernicus_DSM_COG_10_S01_00_E005_00_DEM'
    """
    lat_hemi = "N" if lat >= 0 else "S"
    lng_hemi = "E" if lng >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{lat_hemi}{abs(lat):02d}_00_{lng_hemi}{abs(lng):03d}_00_DEM"


def fetch_copernicus_dem(
    polygon,
    out_path,
    *,
    polygon_crs: str = "EPSG:4326",
    target_crs: str | None = None,
    cache_tile_dir=None,
    cleanup_tiles: bool = True,
    verbose: bool = True,
) -> Path:
    """Download + mosaic + clip + reproject Copernicus GLO-30 DEM for a polygon.

    Workflow:

    1. Compute the 1° × 1° tile bounding box covering `polygon` in EPSG:4326.
    2. Download each tile from AWS Open Data (skipped if already present in
       `cache_tile_dir`).
    3. Mosaic the tiles with `rasterio.merge`.
    4. Clip to `polygon` (in EPSG:4326).
    5. Optionally reproject to `target_crs`.
    6. Write the result as a compressed GeoTIFF to `out_path`.
    7. Optionally clean up the raw tile files.

    If `out_path` already exists, the function is a no-op and returns
    `out_path` immediately — caller is responsible for invalidating the
    cache (e.g., delete the file) when the underlying request changes.

    Args:
        polygon: shapely Polygon or MultiPolygon covering the area of
            interest.
        out_path: destination GeoTIFF path. Parent directory must exist.
        polygon_crs: CRS of `polygon` (default `'EPSG:4326'`).
        target_crs: CRS of the output GeoTIFF (e.g. `'EPSG:2056'` for
            Swiss LV95). `None` keeps EPSG:4326.
        cache_tile_dir: directory for downloaded raw tiles. `None`
            defaults to the parent of `out_path`. Tiles already present
            here are not re-downloaded.
        cleanup_tiles: if `True` (default), the raw per-tile `.tif`
            files are deleted after the mosaic / clip / reproject; set
            `False` to keep them for reuse.
        verbose: print per-tile download progress + final summary.

    Returns:
        `Path` to the saved GeoTIFF (same as `out_path`).

    Raises:
        ImportError: if `rasterio` or `requests` is not installed.
        requests.HTTPError: on tile-download failure.
    """
    try:
        import rasterio
        import requests
        from rasterio.io import MemoryFile
        from rasterio.mask import mask as raster_mask
        from rasterio.merge import merge as raster_merge
        from rasterio.warp import (
            Resampling,
            calculate_default_transform,
            reproject,
        )
    except ImportError as e:
        raise ImportError(
            "fetch_copernicus_dem needs the `topo` extras "
            "(`pip install 'aperta[topo]'`) — missing: " + str(e)
        ) from None

    out_path = Path(out_path)
    if out_path.exists():
        return out_path

    cache_tile_dir = Path(cache_tile_dir) if cache_tile_dir else out_path.parent
    cache_tile_dir.mkdir(parents=True, exist_ok=True)

    # 1°-tile bounding box in EPSG:4326.
    if polygon_crs != "EPSG:4326":
        polygon_4326 = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs("EPSG:4326").iloc[0]
    else:
        polygon_4326 = polygon
    minx, miny, maxx, maxy = polygon_4326.bounds
    lats = range(int(np.floor(miny)), int(np.ceil(maxy)))
    lngs = range(int(np.floor(minx)), int(np.ceil(maxx)))
    tile_names = [_copernicus_tile_name(lat, lng) for lat in lats for lng in lngs]
    if verbose:
        print(f"Downloading {len(tile_names)} Copernicus DEM tile(s)...")

    raw_tiles: list[Path] = []
    for tname in tile_names:
        local = cache_tile_dir / f"{tname}.tif"
        if not local.exists():
            url = _COPERNICUS_DEM_URL_TEMPLATE.format(name=tname)
            if verbose:
                print(f"  {tname} ...", end="", flush=True)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            local.write_bytes(r.content)
            if verbose:
                print(f" {len(r.content) / 1e6:.0f} MB")
        raw_tiles.append(local)

    # Mosaic.
    srcs = [rasterio.open(p) for p in raw_tiles]
    mosaic, mosaic_transform = raster_merge(srcs)
    mosaic_meta = srcs[0].meta.copy()
    mosaic_meta.update(
        {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
            "compress": "lzw",
        }
    )

    # Clip (in EPSG:4326).
    with MemoryFile() as memfile:
        with memfile.open(**mosaic_meta) as tmp:
            tmp.write(mosaic)
        with memfile.open() as tmp:
            clipped, clipped_transform = raster_mask(
                tmp, [polygon_4326.__geo_interface__], crop=True
            )
            clipped_meta = tmp.meta.copy()
    clipped_meta.update(
        {
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": clipped_transform,
            "compress": "lzw",
        }
    )

    for s in srcs:
        s.close()

    # Optionally reproject.
    if target_crs is None or target_crs == clipped_meta["crs"]:
        final_arr = clipped
        final_meta = clipped_meta
    else:
        src_crs = clipped_meta["crs"]
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            target_crs,
            clipped_meta["width"],
            clipped_meta["height"],
            *rasterio.transform.array_bounds(
                clipped_meta["height"], clipped_meta["width"], clipped_meta["transform"]
            ),
        )
        final_arr = np.empty((1, dst_height, dst_width), dtype=clipped.dtype)
        reproject(
            source=clipped[0],
            destination=final_arr[0],
            src_transform=clipped_meta["transform"],
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        final_meta = clipped_meta.copy()
        final_meta.update(
            {
                "crs": target_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "compress": "lzw",
            }
        )

    with rasterio.open(out_path, "w", **final_meta) as dst:
        dst.write(final_arr)

    if cleanup_tiles:
        for p in raw_tiles:
            p.unlink(missing_ok=True)

    if verbose:
        h, w = final_arr.shape[1], final_arr.shape[2]
        valid = final_arr[final_arr != final_meta.get("nodata", None)]
        if valid.size:
            print(
                f"DEM saved to {out_path}: {h} × {w} pixels; "
                f"elevation range {valid.min():.0f}–{valid.max():.0f} m."
            )
        else:
            print(f"DEM saved to {out_path}: {h} × {w} pixels.")

    return out_path
