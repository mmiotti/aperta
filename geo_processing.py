import logging
import warnings

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from aperta import utils


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


def remove_duplicate_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate indices from DataFrame or Series. Often useful after spatial join."""
    df = df.loc[~df.index.duplicated(keep='first'), :]
    return df


def _line_segments(line) -> np.ndarray | None:
    """Stack of `(dx, dy, length)` rows for every segment, across all parts.

    Handles both `LineString` and `MultiLineString` (multipart parts are
    concatenated in stored order). Empty / degenerate inputs return `None`.
    """
    if line is None or line.is_empty:
        return None
    parts = list(line.geoms) if hasattr(line, 'geoms') else [line]
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
        return float('nan')
    pos = max(0.0, min(float(dist_along), total))
    i = int(np.searchsorted(np.cumsum(seg_lens), pos, side='right'))
    i = min(i, len(seg_lens) - 1)
    dx, dy = segs[i, 0], segs[i, 1]
    if dx == 0 and dy == 0:
        return float('nan')
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
            return float('nan')
        return _bearing_at(segs, segs[:, 2].sum() / 2)

    return pd.Series([_bearing(g) for g in geoms], index=geoms.index, dtype='float64')


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
        return float('nan')
    return _bearing_at(segs, dist_along)


def add_lat_lon(
    df: pd.DataFrame,
    prefix: str,
    from_crs: str,
    to_crs: str = 'EPSG:4326',
) -> pd.DataFrame:
    """Add `<prefix>_lat` and `<prefix>_lon` columns by transforming `<prefix>_x/y`."""
    transformer = Transformer.from_crs(from_crs, to_crs)
    lat, lon = transformer.transform(df[f'{prefix}_x'].to_numpy(),
                                     df[f'{prefix}_y'].to_numpy())
    df[f'{prefix}_lat'] = lat
    df[f'{prefix}_lon'] = lon
    return df


def add_straight_line_dist(
    df: pd.DataFrame,
    orig_prefix: str = 'orig',
    dest_prefix: str = 'dest',
    out_col: str = 'dist_line',
) -> pd.DataFrame:
    """Add euclidean origin→destination distance, in CRS units (typically metres)."""
    dx = df[f'{orig_prefix}_x'] - df[f'{dest_prefix}_x']
    dy = df[f'{orig_prefix}_y'] - df[f'{dest_prefix}_y']
    df[out_col] = np.hypot(dx, dy)
    return df


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
            f"{precision:.5f}: {maximum_size} ({new_size:,})")
    return new_geometry, new_size, new_size / old_size


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


@utils.timeit
def custom_spatial_lag(
    gdf: gpd.GeoDataFrame,
    cols: list[str],
    radius: int | float,
    return_densities: bool,
    add_filled_densities: bool,
) -> pd.DataFrame:
    """Calculate regular and filled averages of columns `cols' in gdf `gdf' within `radius` around each cell.

    The regular average for a column representing the population in each cell will be the population density per km².
    The filled average is the population density per km² area of gdf where at least one value in cols is larger than
    zero, that is, the population density per occupied area.

    The CRS of `gdf` must represent meters if densities are returned. Densities are returned per km².
    """

    if not return_densities and add_filled_densities:
        raise ValueError("`add_filled_densities` can only be True when `return_densities` is True.")

    w = libpysal.weights.DistanceBand.from_dataframe(
        gdf, threshold=radius, binary=True, silence_warnings=True,
    )

    # Each cell itself (focal) should be included when calculating its own spatial average.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The weights matrix is not fully connected")
        w = libpysal.weights.fill_diagonal(w, 1.0)
    # We are not transforming row weights to yield average
    # w.transform = 'r'

    res = pd.DataFrame(index=gdf.index)
    circular_area = (radius ** 2 * 3.14159) / 1e6
    cell_areas = gdf.geometry.area / 1e6

    spatial_sums = np.array(libpysal.weights.lag_spatial(w, gdf[cols]))
    values = spatial_sums / circular_area if return_densities else spatial_sums
    df = pd.DataFrame(values, index=gdf.index, columns=[f'{col}_r{radius}' for col in cols])
    res = res.join(df)

    # Calculate filled cell areas (total area within radius around each cell where at least one column is > 0)
    if add_filled_densities:
        # Note 1: "filled" densities currently include any cell that has at least something one it (regardless of column)
        # Note 2: "filled" densities currently lead to some dramatically high values around e.g. Zurich airport, where
        # a lot of jobs are allocated to very few cells.
        is_filled = gdf.index[gdf[cols].sum(axis=1) > 0]
        if len(np.unique(cell_areas.round(decimals=4))) > 1:
            raise NotImplementedError("Cells must currently be same size to calculate filled densities.")
        fill_cell_areas = []
        for cell_area, idx in zip(cell_areas, gdf.index):
            # Count number of cells that have something on it (at least one variable is > 0)
            fill_cell_areas.append(cell_area * len([1 for cell_idx in w[idx].keys() if cell_idx in is_filled]))
        fill_cell_areas = np.array(fill_cell_areas)
        f = fill_cell_areas > 0
        spatial_sums[f, :] = spatial_sums[f, :] / fill_cell_areas[f, np.newaxis]
        spatial_sums[~f, :] = 0
        df = pd.DataFrame(spatial_sums, index=gdf.index, columns=[f'{col}_r{radius}_filled' for col in cols])
        res = res.join(df)

    res = res.fillna(0)
    return res


@utils.timeit
def aggregate_within_radius(
    targets: gpd.GeoDataFrame | gpd.GeoSeries,
    sources: gpd.GeoDataFrame | gpd.GeoSeries,
    radius: float,
    *,
    weight_column: str | None = None,
    return_density: bool = False,
    name: str = 'aggregate',
) -> pd.Series:
    """For each target geometry, aggregate over source geometries within `radius`.

    Cross-set buffer aggregation. For each target's centroid, finds source
    centroids within `radius` and either counts them (default) or sums a
    chosen column. Returns a Series indexed by `targets.index`.

    Backend: scipy `cKDTree` on source coordinates — O(N_src log N_src) one-
    time build, O(log N_src + k) per query. Much faster (and bounded-memory)
    versus a libpysal weights matrix for the cross-set case.

    For SAME-set spatial lag (`gdf × gdf` with reusable weights, multiple
    columns at once, optional "filled-area" densities) use
    [[custom_spatial_lag]] instead. The two helpers are deliberately separate:
    same-set lag and cross-set buffer queries are distinct operations with
    different optimal backends.

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
    src_xy = np.column_stack([src_centroids.x.to_numpy(dtype=float),
                              src_centroids.y.to_numpy(dtype=float)])
    tgt_centroids = targets.geometry.centroid
    tgt_xy = np.column_stack([tgt_centroids.x.to_numpy(dtype=float),
                              tgt_centroids.y.to_numpy(dtype=float)])
    tree = KDTree(src_xy)

    if weight_column is None:
        # `return_length=True` (scipy ≥1.6) skips materialising the per-query
        # index lists — much faster for large source sets.
        agg = np.asarray(
            tree.query_ball_point(tgt_xy, r=radius, return_length=True),
            dtype=float,
        )
    else:
        if (not isinstance(sources, gpd.GeoDataFrame)
                or weight_column not in sources.columns):
            raise ValueError(
                f"`sources` must be a GeoDataFrame containing column "
                f"{weight_column!r} when `weight_column` is given.")
        weights = sources[weight_column].to_numpy(dtype=float)
        idx_lists = tree.query_ball_point(tgt_xy, r=radius)
        agg = np.fromiter(
            (float(weights[idxs].sum()) if len(idxs) else 0.0
             for idxs in idx_lists),
            dtype=float, count=len(idx_lists),
        )

    if return_density:
        agg = agg / (np.pi * radius ** 2)
    return pd.Series(agg, index=targets.index, name=name)


@utils.timeit
def sample_raster_at_points(
    points: gpd.GeoDataFrame | gpd.GeoSeries,
    raster_path,
    *,
    band: int = 1,
    method: str = 'bilinear',
    name: str = 'raster_value',
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

    with rasterio.open(raster_path) as src:
        arr = src.read(band).astype(float)
        inv_transform = ~src.transform
        nodata = src.nodata
        height, width = src.height, src.width

    # Mask nodata up-front so both branches see NaN — bilinear's weighted
    # sum will then naturally propagate NaN when any contributing pixel
    # is invalid.
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    centroids = points.geometry.centroid
    xs = centroids.x.to_numpy(dtype=float)
    ys = centroids.y.to_numpy(dtype=float)
    # Affine `~transform` maps (x, y) -> (col_float, row_float). The +/-0.5
    # shift converts from pixel-corner to pixel-centre coordinates, which
    # is what bilinear interpolation expects: a point at the pixel's
    # centre should return that pixel's value exactly, not a blend.
    cols_f, rows_f = inv_transform * (xs, ys)

    out = np.full(len(points), np.nan, dtype=float)
    if method == 'nearest':
        cols = np.floor(cols_f).astype(int)
        rows = np.floor(rows_f).astype(int)
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if valid.any():
            out[valid] = arr[rows[valid], cols[valid]]
    elif method == 'bilinear':
        from scipy.ndimage import map_coordinates
        # `map_coordinates` expects (row, col) at pixel CENTRES (an integer
        # coord = that pixel's centre). Subtract 0.5 to convert from corner-
        # based pixel coords. mode='constant', cval=nan returns NaN out-of-
        # bounds; NaN inputs propagate to NaN outputs via the weighted sum.
        valid = ((cols_f >= 0) & (cols_f <= width)
                 & (rows_f >= 0) & (rows_f <= height))
        if valid.any():
            coords = np.vstack([rows_f[valid] - 0.5, cols_f[valid] - 0.5])
            out[valid] = map_coordinates(arr, coords, order=1,
                                         mode='constant', cval=np.nan)
    else:
        raise ValueError(f"method must be 'bilinear' or 'nearest', got {method!r}")

    return pd.Series(out, index=points.index, name=name)


def build_h3_grid(
    polygon,
    resolution: int,
    *,
    polygon_crs: str = 'EPSG:4326',
    target_crs: str | None = None,
    id_column: str = 'cell_id',
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of H3 hex cells covering `polygon`.

    One row per H3 cell whose centre lies inside `polygon`. Geometry is the
    closed hex polygon. The H3 library operates in EPSG:4326 (WGS84); the
    input polygon is reprojected if `polygon_crs != 'EPSG:4326'`, and the
    output is optionally reprojected to `target_crs`.

    For nested tiers (e.g. cells at H3 res 10, zones at res 8, regions at
    res 6 — a common multi-scale convention), call once per resolution OR
    derive the coarser tiers by `h3.cell_to_parent` on the finest tier
    (cheaper, and the nesting is exact). Pure replacement of the private
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

    if polygon_crs != 'EPSG:4326':
        polygon = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs('EPSG:4326').iloc[0]

    parts = (list(polygon.geoms) if polygon.geom_type == 'MultiPolygon'
             else [polygon])
    cell_ids: list[str] = []
    for part in parts:
        # shapely uses (lng, lat); H3 wants (lat, lng).
        exterior = [(lat, lng) for lng, lat in part.exterior.coords]
        interiors = [[(lat, lng) for lng, lat in ring.coords]
                     for ring in part.interiors]
        h3_poly = h3.LatLngPoly(exterior, *interiors)
        cell_ids.extend(h3.h3shape_to_cells(h3_poly, resolution))

    cell_ids = sorted(set(cell_ids))
    geoms = [ShPolygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)])
             for c in cell_ids]
    gdf = gpd.GeoDataFrame(
        {'geometry': geoms},
        index=pd.Index(cell_ids, name=id_column),
        crs='EPSG:4326',
    )
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf
