"""Topographic data fetchers for aperta.

So far one helper: `fetch_copernicus_dem`, which downloads, mosaics, clips,
and (optionally) reprojects Copernicus GLO-30 DEM tiles from AWS Open Data
for a given polygon. The Copernicus GLO-30 is the de facto open global 30 m
DEM and is the right default for accessibility analyses that need
elevation-aware edge weights.

Heavy optional dependencies (`rasterio`, `requests`) are imported lazily —
the rest of `aperta` works without them.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np

COPERNICUS_DEM_URL_TEMPLATE = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
COPERNICUS_DEM_TILE_NAME_TEMPLATE = "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lng:03d}_00_DEM"


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
    tile_names = [
        COPERNICUS_DEM_TILE_NAME_TEMPLATE.format(lat=lat, lng=lng) for lat in lats for lng in lngs
    ]
    if verbose:
        print(f"Downloading {len(tile_names)} Copernicus DEM tile(s)...")

    raw_tiles: list[Path] = []
    for tname in tile_names:
        local = cache_tile_dir / f"{tname}.tif"
        if not local.exists():
            url = COPERNICUS_DEM_URL_TEMPLATE.format(name=tname)
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
