"""Plot helpers shared across `examples/extended/` notebooks.

Pure presentation code — graph + data → matplotlib figure. Lives here
rather than inline in each notebook so the notebook flow stays focused
on the substantive (`what aperta does, and how`) bits. Project-specific
styling (highway-tier line widths, capacity table) that isn't generic
enough to belong in `aperta.visualization`.

Generic primitives — `plot_edge_values`, `add_styled_colorbar` — are in
`aperta.visualization` and used by the wrappers here.

Location-specific parameters (crop centre, zoom width, label) are
passed in by the calling notebook rather than hardcoded here, so the
showcase retargets cleanly when the seed location changes.

Underscore prefix on the module name flags it as project-internal —
not a tutorial example to read, just helpers the notebooks call.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from aperta import visualization as viz
from aperta.network_processing import OSM_HIGHWAY_RANKS


# Harmonised font sizes for paper-figure output. Imported by every
# /extended notebook (via `import _figures as figures`) so the rcParams
# stick globally for matplotlib.
TITLE_SIZE  = 12   # axes titles
LABEL_SIZE  = 12   # axes labels + colour-bar labels (match title size)
LEGEND_SIZE = 10   # legend text and any in-figure annotation labels
TICK_SIZE   = 10   # tick labels (axis + colour-bar)

plt.rcParams['axes.titlesize']  = TITLE_SIZE
plt.rcParams['axes.labelsize']  = LABEL_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE


# Paper-figure export — all figures from /extended notebooks save here.
# Caller-relative path: notebooks run from `extended/`, so this resolves
# to `extended/results/figures_highres/`.
import os as _os
from pathlib import Path as _Path
PAPER_FIGURES_DIR = _Path('results/figures_highres')


def save_figure(fig, name: str, *, ext: str = 'png', dpi: int = 300,
                bbox_inches: str = 'tight'):
    """Save `fig` to `PAPER_FIGURES_DIR / f'{name}.{ext}'` at high DPI.

    Defaults: PNG at 300 DPI with `bbox_inches='tight'` — the right
    choice for raster map content (network plots, choropleths, OSM
    basemap underlays). Pass `ext='pdf'` for vector-friendly content
    (scatter plots, text-heavy figures) where scaling cleanly matters.
    """
    out = PAPER_FIGURES_DIR / f'{name}.{ext}'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches)
    print(f'Saved {out}')


# Per-highway-tier line widths for network maps. Motorway/trunk thickest,
# residential thinnest — makes road class readable independent of the
# colour scale.
HWY_WIDTH = {
    'motorway': 3.5, 'motorway_link': 2.0,
    'trunk': 3.0,    'trunk_link': 1.5,
    'primary': 2.4,  'primary_link': 1.2,
    'secondary': 1.8, 'secondary_link': 1.0,
    'tertiary': 1.4, 'tertiary_link': 0.9,
    'unclassified': 1.0, 'residential': 0.8,
    'living_street': 0.6, 'service': 0.5, 'road': 0.5, 'busway': 0.5,
}

# Rough literature values for per-lane daily capacity (veh/lane/day).
# Sources: HCM 2010 + assorted urban-planning rules of thumb, rounded.
# Used to compute (V/C)² as a BPR-style congestion feature.
CAPACITY_PER_LANE = {
    'motorway': 35000, 'motorway_link': 25000,
    'trunk': 28000,    'trunk_link': 20000,
    'primary': 22000,  'primary_link': 16000,
    'secondary': 18000, 'secondary_link': 14000,
    'tertiary': 14000, 'tertiary_link': 11000,
    'unclassified': 10000, 'residential': 9000,
    'living_street': 6000, 'service': 6000,
    'road': 9000, 'busway': 6000,
}
DEFAULT_CAPACITY = 9000

def edge_highway(d) -> str | None:
    """Flatten OSM `highway` tag (may be list-valued post-merge) to a single str."""
    hwy = d.get('highway')
    if isinstance(hwy, list):
        return hwy[0] if hwy else None
    return hwy


def crop_to_polygon(polygon, *, buffer_frac: float = 0.05
                    ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return `(xlim, ylim)` for a plot cropped to a polygon's bbox.

    `buffer_frac` shrinks the bbox by that fraction on each side
    (default 5 %, i.e. show 90 % of the bbox).
    """
    minx, miny, maxx, maxy = polygon.bounds
    dx, dy = maxx - minx, maxy - miny
    xlim = (minx + buffer_frac * dx, maxx - buffer_frac * dx)
    ylim = (miny + buffer_frac * dy, maxy - buffer_frac * dy)
    return xlim, ylim


def plot_network_map(
    ax,
    graph: nx.MultiDiGraph,
    values: dict | pd.Series,
    *,
    cmap='Reds',
    vmin: float = 0.0,
    vmax: float | None = None,
    vmax_quantile: float = 0.99,
    cbar_label: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    basemap: bool = False,
    crs=None,
):
    """Draw a per-edge network map with highest-tier roads on top.

    Wraps `aperta.visualization.plot_edge_values` with the Swiss
    aesthetic: per-tier line widths from `HWY_WIDTH`, sorted by
    `OSM_HIGHWAY_RANKS` ascending (motorway/trunk land on top of the
    residential mesh — without that, thin gray edges visually mask
    the busiest roads at junctions), height-matched colour bar, square
    aspect, hidden ticks, optional bbox crop.

    Args:
        ax: target matplotlib axes.
        graph: nx graph; each edge should have `geometry` (LineString)
            and `highway` for proper styling.
        values: per-edge value mapping `(u, v, k) -> float`.
        cmap, vmin: matplotlib colour-scale settings.
        vmax: explicit colour-scale ceiling. If `None`, derived from
            `vmax_quantile` of the positive values in `values`.
        vmax_quantile: quantile used to auto-clip the colour scale.
            Extreme bottlenecks compress the rest beyond P99 / P95
            etc. — `0.99` is the usual choice for this notebook.
        cbar_label, title: colour-bar label, axes title.
        xlim, ylim: optional bbox crop tuples (use `crop_to_polygon`).
        basemap: render CartoDB Positron basemap underneath. When False
            (default), the axes have a white facecolor — preserves the
            historical look. When True, the basemap shows through, so
            the white facecolor is dropped.
        crs: CRS for the basemap. Defaults to `graph.graph['crs']`.
    """
    vals = np.asarray(list(values.values()) if isinstance(values, dict)
                      else values.to_numpy())
    if vmax is None:
        pos = vals[vals > 0]
        vmax = float(np.quantile(pos, vmax_quantile)) if pos.size else 1.0

    edge_widths = {
        (u, v, k): HWY_WIDTH.get(edge_highway(d), 0.5)
        for u, v, k, d in graph.edges(keys=True, data=True)
    }

    viz.plot_edge_values(
        graph, values, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        edge_widths=edge_widths,
        sort_key=lambda key, d: OSM_HIGHWAY_RANKS.get(edge_highway(d), -1),
    )
    if not basemap:
        ax.set_facecolor('white')
    viz.add_styled_colorbar(ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str)
                            else cmap,
                            vmin=vmin, vmax=vmax, label=cbar_label)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if basemap:
        _add_positron_basemap(ax, crs or graph.graph.get('crs', 'EPSG:2056'))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Polygon- / basemap-aware figures (city focus + neighborhood zoom + AOI)
# ---------------------------------------------------------------------------
# Three figure types share the same look:
#   - `plot_aoi_overview`   — full AOI silhouette, for overview / prep plots
#   - `plot_city_focus`     — city-shaped clip, for results (accessibility, flows)
#   - `plot_neighborhood_zoom` — square crop, building-level (deep-zoom story)
#
# All three render onto a CartoDB Positron-no-labels basemap by default —
# the grey backdrop carries the geographic context that makes the data
# cells read as a *city silhouette* rather than an arbitrary collection
# of coloured hexes. Pass `basemap=False` to skip (useful for network
# figures where the basemap would duplicate rendered roads).

_POI_DEFAULT_STYLE = {
    'supermarket':       dict(color='red',       markersize=5, edgecolor='black', linewidth=0.25),
    'groceries':         dict(color='red',       markersize=5, edgecolor='black', linewidth=0.25),
    # `darkgreen` reads against viridis green-yellow more clearly than
    # default green (#008000); the contrast against yellow-zone hexes is
    # what made hiking dots fade in the previous render.
    'hiking':            dict(color='darkgreen', markersize=5, edgecolor='black', linewidth=0.25),
}


def fetch_city_polygon(name: str, *, target_crs=None,
                       clip_geom=None,
                       smooth: bool = True,
                       simplify_m: float = 150.0,
                       smooth_m: float = 80.0):
    """Fetch a city / admin-area polygon via `osmnx.geocode_to_gdf`.

    Optional `clip_geom` intersects the fetched polygon — useful when a
    municipal boundary has awkward narrow appendages that dominate the
    bbox (Bern, for instance, extends ~7 km west of its main body in a
    thin strip). Pass a bbox via `shapely.geometry.box(minx, miny, maxx,
    maxy)` or any polygon. `smooth=True` then simplifies + morphological-
    closes to clean up jagged edges.

    Args:
        name: place name for `ox.geocode_to_gdf` (e.g. `'Bern, Switzerland'`).
        target_crs: optional CRS to reproject to (e.g. the cells' CRS).
        clip_geom: optional polygon (in `target_crs`) to intersect with.
        smooth: apply simplification + morphological closing.
        simplify_m, smooth_m: simplification tolerance and morphological
            closing radius, in target-CRS units (typically metres).
    """
    import osmnx as ox
    gdf = ox.geocode_to_gdf(name)
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    poly = gdf.geometry.iloc[0]
    if clip_geom is not None:
        poly = poly.intersection(clip_geom)
    if smooth:
        if simplify_m and simplify_m > 0:
            poly = poly.simplify(simplify_m)
        if smooth_m and smooth_m > 0:
            # Closing + opening — fills small concavities (gaps) AND
            # rounds off small convex protrusions (spikes). Closing
            # alone leaves jagged tooth-edges on the outside.
            poly = poly.buffer(smooth_m).buffer(-2 * smooth_m).buffer(smooth_m)
    return poly


def _add_positron_basemap(ax, crs, *, attribution: bool = False):
    """Add CartoDB Positron-no-labels basemap, @2x for sharp paper rendering.

    Defaults to `attribution=False` for clean panels; flip to `True` when
    rendering for publication."""
    import contextily as ctx
    ctx.add_basemap(
        ax,
        crs=crs,
        source=ctx.providers.CartoDB.PositronNoLabels(r='@2x'),
        attribution='© OpenStreetMap, © CartoDB' if attribution else False,
        attribution_size=6,
    )


def _set_polygon_extent(ax, polygon, pad_frac: float = 0.05):
    """Set ax xlim/ylim to a polygon's bbox + a small fractional pad."""
    minx, miny, maxx, maxy = polygon.bounds
    dx, dy = maxx - minx, maxy - miny
    ax.set_xlim(minx - pad_frac * dx, maxx + pad_frac * dx)
    ax.set_ylim(miny - pad_frac * dy, maxy + pad_frac * dy)


def _plot_pois(ax, pois, *, clip_polygon=None):
    """Plot a POI dict — `{label: gdf, ...}` — with default styling.

    `clip_polygon` (optional): restrict POIs to those inside the polygon
    so the overlay matches the data extent rather than spilling beyond.
    Pass an explicit `(gdf, kwargs)` tuple to override per layer."""
    if not pois:
        return
    for label, spec in pois.items():
        if isinstance(spec, tuple):
            gdf, kwargs = spec
        else:
            gdf = spec
            style = next((v for k, v in _POI_DEFAULT_STYLE.items()
                          if k in label.lower()), None)
            kwargs = style if style else dict(color='red', markersize=7)
        if clip_polygon is not None:
            gdf = gdf[gdf.geometry.within(clip_polygon)]
        gdf.plot(ax=ax, **kwargs, label=label, zorder=5)


def plot_aoi_overview(
    ax,
    polygon,
    *,
    cells: gpd.GeoDataFrame | None = None,
    values: pd.Series | None = None,
    cmap='viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    basemap: bool = True,
    title: str = '',
    label: str = '',
):
    """AOI-shaped overview map.

    Polygon outline drawn on top of (optionally) a per-cell choropleth and
    a Positron basemap. Useful for prep figures (DEM coverage, raw data
    extent) and methodology figures (the full study area silhouette).

    Args:
        ax: target matplotlib axes.
        polygon: AOI polygon, in the cells' CRS.
        cells, values: optional per-cell choropleth (same convention as
            `plot_cell_focus`).
        cmap, vmin, vmax: colour-scale settings.
        basemap: render CartoDB Positron basemap underneath.
        title, label: axes title and colourbar label.
    """
    if cells is not None and values is not None:
        plot_values = values.dropna() if hasattr(values, 'dropna') else values
        cells_plot = cells.loc[cells.index.isin(plot_values.index)]
        if vmin is None:
            vmin = float(plot_values.min())
        if vmax is None:
            vmax = float(plot_values.max())
        viz.plot_cell_values(
            cells_plot, plot_values, ax=ax, cmap=cmap,
            vmin=vmin, vmax=vmax, legend=False,
        )
        viz.add_styled_colorbar(
            ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
            vmin=vmin, vmax=vmax, label=label,
            size='4%', extend='neither',
        )

    aoi_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=getattr(cells, 'crs', None))
    aoi_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=6)
    _set_polygon_extent(ax, polygon, pad_frac=0.05)

    if basemap and cells is not None:
        _add_positron_basemap(ax, cells.crs)

    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)


def plot_city_focus(
    ax,
    city_polygon,
    cells: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    pois: dict | None = None,
    cmap='viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    basemap: bool = True,
    clip_to_polygon: bool = True,
    crop_center_xy: tuple[float, float] | None = None,
    crop_half_m: float | None = None,
    title: str = '',
    label: str = '',
):
    """City-shaped focus map with per-cell values and optional POI overlays.

    Filters `cells` to those whose centroid lies inside `city_polygon`
    (when `clip_to_polygon=True`) so the rendered data fills the city
    silhouette. Surrounding area shows through to the Positron basemap.

    Two extent modes:
      - `crop_center_xy=None`: extent = polygon bbox + 5 % pad. Aspect
        follows the polygon's bbox (may be non-square).
      - `crop_center_xy=(cx, cy)` + `crop_half_m=r`: square crop, `2r`
        wide and tall, centred on `(cx, cy)`. Parts of the city polygon
        outside the crop get visually clipped at the panel edges.

    Args:
        ax: target matplotlib axes.
        city_polygon: clip polygon (e.g. via `fetch_city_polygon`).
        cells: cell-indexed GeoDataFrame; must share CRS with `city_polygon`.
        values: per-cell values (Series indexed by cell_id).
        pois: optional `{label: gdf}` (default styling per label keyword:
            'supermarket'/'groceries' → red, 'hiking' → green), or
            `{label: (gdf, plot_kwargs)}` for full per-layer control.
        cmap, vmin, vmax: colour-scale settings; `vmin`/`vmax` auto-derived
            from the filtered values when `None`.
        basemap: render CartoDB Positron basemap underneath.
        clip_to_polygon: filter cells (and POIs) to those inside the polygon.
        crop_center_xy, crop_half_m: optional square crop — see two-modes
            note above.
        title, label: axes title and colourbar label.
    """
    # Filter to in-polygon cells (centroid test) — gives the silhouette.
    if clip_to_polygon:
        inside = cells.geometry.centroid.within(city_polygon)
        cells_plot = cells.loc[inside]
    else:
        cells_plot = cells
    vals = values.loc[values.index.isin(cells_plot.index)].dropna() if hasattr(values, 'dropna') else values
    cells_plot = cells_plot.loc[cells_plot.index.isin(vals.index)]

    if vmin is None:
        vmin = float(vals.min()) if len(vals) else 0.0
    if vmax is None:
        vmax = float(vals.max()) if len(vals) else 1.0

    viz.plot_cell_values(
        cells_plot, vals, ax=ax, cmap=cmap,
        vmin=vmin, vmax=vmax, legend=False,
    )

    # City boundary on top — dark-ish grey, partially opaque, clearly
    # visible as a silhouette outline without going as hard as black.
    city_gdf = gpd.GeoDataFrame(geometry=[city_polygon], crs=cells.crs)
    city_gdf.boundary.plot(ax=ax, color='0.5', linewidth=1.0, alpha=0.6, zorder=4)

    _plot_pois(ax, pois, clip_polygon=city_polygon if clip_to_polygon else None)

    if crop_center_xy is not None and crop_half_m is not None:
        cx, cy = crop_center_xy
        ax.set_xlim(cx - crop_half_m, cx + crop_half_m)
        ax.set_ylim(cy - crop_half_m, cy + crop_half_m)
    else:
        _set_polygon_extent(ax, city_polygon, pad_frac=0.05)
    if basemap:
        _add_positron_basemap(ax, cells.crs)

    ax.set_aspect('equal')
    # Hide ticks WITHOUT disabling the axis — leaves the spines + ylabel
    # mechanism intact for downstream callers that want to add row labels.
    ax.tick_params(left=False, bottom=False, top=False, right=False,
                   labelleft=False, labelbottom=False,
                   labeltop=False, labelright=False)
    ax.set_axis_on()
    ax.set_frame_on(True)
    # Thin black frame around each panel — on top of basemap + data.
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('black')
        s.set_linewidth(0.8)
        s.set_zorder(10)
    if title:
        ax.set_title(title)
    viz.add_styled_colorbar(
        ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        vmin=vmin, vmax=vmax, label=label,
        size='4%', extend='neither',
    )


def plot_routes_on_basemap(
    ax,
    graph: nx.Graph,
    paths: list[list],
    edge_attribute,
    *,
    crop_center_xy: tuple[float, float],
    crop_half_m: float,
    cmap='RdYlGn',
    vmin: float | None = None,
    vmax: float | None = None,
    linewidth: float = 2.6,
    origin_xy: tuple[float, float] | None = None,
    dest_xys: list[tuple[float, float]] | None = None,
    city_polygon=None,
    weight_for_min_parallel: str = 'bike_time_s',
    basemap: bool = True,
    title: str = '',
    label: str = '',
    crs=None,
):
    """Render realised routes on a basemap, edges coloured by an attribute.

    Visualises the path-aggregation mechanism: each path is a list of
    node IDs (output of `nx.shortest_path` or `scipy.csgraph.dijkstra`
    + predecessor-walk); per-edge attribute values are extracted and
    rendered as a coloured `LineCollection`. Use the same `cmap` /
    `vmin` / `vmax` as a per-origin aggregation map next to it to make
    the "we walk the path and average" mechanism visually literal.

    Args:
        ax: target matplotlib axes.
        graph: the routable graph the paths came from.
        paths: list of node-ID sequences. Empty or single-node paths
            are skipped.
        edge_attribute: callable `(u, v, data) → float` or a string
            edge-attribute name. Resolved per edge along each path.
        crop_center_xy, crop_half_m: square crop window (same convention
            as `plot_city_focus`).
        cmap, vmin, vmax, linewidth: line-colour settings; pass the
            same vmin/vmax as the companion aggregation map.
        origin_xy, dest_xys: optional marker coordinates (CRS-units).
            Origin gets a distinctive marker, destinations small dots.
        city_polygon: optional outline (e.g. the same polygon used by
            `plot_city_focus`) drawn faintly for geographic context.
        weight_for_min_parallel: for MultiDiGraph parallels, the edge
            attribute used to pick the min-weight parallel (matches
            the router's choice).
        basemap: render CartoDB Positron backdrop.
        title, label: axes title and colour-bar label.
        crs: CRS to use for the basemap. Defaults to `graph.graph['crs']`
            if present.
    """
    import networkx as nx_  # local alias for type checks
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if isinstance(edge_attribute, str):
        attr_fn = lambda u, v, d: float(d[edge_attribute])
    else:
        attr_fn = edge_attribute

    is_multi = isinstance(graph, (nx_.MultiGraph, nx_.MultiDiGraph))

    segments: list = []
    colors: list = []
    for path in paths:
        if path is None or len(path) < 2:
            continue
        for u, v in zip(path[:-1], path[1:]):
            if v not in graph[u]:
                continue
            if is_multi:
                d = min(graph[u][v].values(),
                        key=lambda dd: dd.get(weight_for_min_parallel, np.inf))
            else:
                d = graph[u][v]
            color_val = attr_fn(u, v, d)
            geom = d.get('geometry')
            if geom is not None and hasattr(geom, 'coords'):
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    segments.append((coords[i], coords[i + 1]))
                    colors.append(color_val)
            else:
                xu, yu = graph.nodes[u]['x'], graph.nodes[u]['y']
                xv, yv = graph.nodes[v]['x'], graph.nodes[v]['y']
                segments.append(((xu, yu), (xv, yv)))
                colors.append(color_val)

    cx, cy = crop_center_xy
    ax.set_xlim(cx - crop_half_m, cx + crop_half_m)
    ax.set_ylim(cy - crop_half_m, cy + crop_half_m)

    if city_polygon is not None:
        crs_for_poly = crs or graph.graph.get('crs')
        gpd.GeoDataFrame(geometry=[city_polygon], crs=crs_for_poly).boundary.plot(
            ax=ax, color='0.35', linewidth=0.6, alpha=0.6, zorder=3,
        )

    if basemap:
        basemap_crs = crs or graph.graph.get('crs', 'EPSG:2056')
        _add_positron_basemap(ax, basemap_crs)

    lc = LineCollection(
        segments, cmap=cmap, linewidth=linewidth,
        norm=Normalize(vmin=vmin, vmax=vmax),
        zorder=5,
    )
    lc.set_array(np.asarray(colors))
    ax.add_collection(lc)

    if dest_xys:
        xs = [xy[0] for xy in dest_xys]
        ys = [xy[1] for xy in dest_xys]
        ax.scatter(xs, ys, s=16, color='red', edgecolor='black',
                   linewidth=0.3, zorder=7)
    if origin_xy is not None:
        ax.scatter([origin_xy[0]], [origin_xy[1]], s=90, marker='o',
                   facecolor='white', edgecolor='black', linewidth=1.2, zorder=8)

    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, top=False, right=False,
                   labelleft=False, labelbottom=False)
    ax.set_axis_on()
    ax.set_frame_on(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('black')
        s.set_linewidth(0.8)
        s.set_zorder(10)
    if title:
        ax.set_title(title)
    viz.add_styled_colorbar(
        ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        vmin=vmin, vmax=vmax, label=label,
        size='4%', extend='neither',
    )


def plot_neighborhood_zoom(
    ax,
    buildings: gpd.GeoDataFrame,
    *,
    center_xy: tuple[float, float],
    half_m: float,
    values: pd.Series | None = None,
    network: nx.Graph | None = None,
    cmap='viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    network_kwargs: dict | None = None,
    basemap: bool = True,
    title: str = '',
    label: str = '',
):
    """Deep-zoom map at building granularity.

    Square crop centred on `center_xy` with half-width `half_m`. Buildings
    coloured by `values` (a per-building Series, indexed like `buildings`);
    optional network overlay in white on top.

    Used for the "we can attribute accessibility down to individual
    buildings" story — the per-cell logsum joined back to the building
    layer that fed into the cells in prep.

    Args:
        ax: target matplotlib axes.
        buildings: building-indexed GeoDataFrame.
        center_xy: `(x, y)` centre in the buildings' CRS.
        half_m: half-width of the square crop.
        values: per-building values (Series indexed like `buildings`).
        network: optional networkx graph; edges drawn in white as overlay.
        cmap, vmin, vmax: colour-scale settings.
        network_kwargs: override styling for the network overlay; defaults
            to thin white.
        basemap: render CartoDB Positron basemap underneath.
        title, label: axes title and colourbar label.
    """
    cx, cy = center_xy
    xlim = (cx - half_m, cx + half_m)
    ylim = (cy - half_m, cy + half_m)

    # Filter to buildings whose bbox intersects the crop window — keeps
    # rendering cost bounded on large building layers.
    bx = buildings.geometry.bounds
    in_window = ((bx['maxx'] >= xlim[0]) & (bx['minx'] <= xlim[1]) &
                 (bx['maxy'] >= ylim[0]) & (bx['miny'] <= ylim[1]))
    sub = buildings.loc[in_window]

    if values is not None:
        vals = values.loc[values.index.isin(sub.index)].dropna() if hasattr(values, 'dropna') else values
        sub = sub.loc[sub.index.isin(vals.index)].copy()
        sub['_v'] = vals.reindex(sub.index).values
        if vmin is None:
            vmin = float(sub['_v'].min())
        if vmax is None:
            vmax = float(sub['_v'].max())
        sub.plot(ax=ax, column='_v', cmap=cmap, vmin=vmin, vmax=vmax,
                 edgecolor='none', linewidth=0)
    else:
        sub.plot(ax=ax, color='dimgrey', edgecolor='none')

    if network is not None:
        # Edges intersecting the crop window. Light white lines over the
        # buildings so the path structure is visible without dominating.
        nk = dict(color='white', linewidth=0.6, alpha=0.9) | (network_kwargs or {})
        edge_records = []
        for u, v, d in network.edges(data=True):
            geom = d.get('geometry')
            if geom is None:
                # Synthesize from node coords if no explicit geometry.
                from shapely.geometry import LineString
                xu, yu = network.nodes[u]['x'], network.nodes[u]['y']
                xv, yv = network.nodes[v]['x'], network.nodes[v]['y']
                geom = LineString([(xu, yu), (xv, yv)])
            minx_e, miny_e, maxx_e, maxy_e = geom.bounds
            if (maxx_e >= xlim[0] and minx_e <= xlim[1]
                    and maxy_e >= ylim[0] and miny_e <= ylim[1]):
                edge_records.append(geom)
        if edge_records:
            edges_gdf = gpd.GeoDataFrame(geometry=edge_records, crs=buildings.crs)
            edges_gdf.plot(ax=ax, **nk, zorder=3)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    if basemap:
        _add_positron_basemap(ax, buildings.crs)

    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_title(title)
    if values is not None:
        viz.add_styled_colorbar(
            ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
            vmin=vmin, vmax=vmax, label=label,
            size='4%', extend='neither',
        )


def plot_cell_map_cropped(
    ax,
    cells: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    crop_center_xy: tuple[float, float],
    crop_half_m: float,
    cmap='viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    title: str,
    label: str = '',
):
    """Plot per-cell `values` on a square-cropped window.

    Square + framed (axes spines visible, ticks hidden), height-matched
    colour bar. Wraps `viz.plot_cell_values` with the accessibility-
    notebook aesthetic.

    Args:
        ax: target matplotlib axes.
        cells: GeoDataFrame indexed by cell_id with `geometry`.
        values: per-cell values (Series indexed by cell_id).
        crop_center_xy: `(x, y)` centre of the crop in the cells' CRS.
        crop_half_m: half-width of the square crop, in the cells' CRS
            distance units. `(crop_half_m * 2)²` is the visible area.
        cmap: matplotlib colormap.
        vmin, vmax: explicit colour range. If both `None`, auto-derived
            from `values`.
        symmetric: if True, use `[-max(|v|), +max(|v|)]` (good for
            diverging metrics like percent change).
        title, label: axes title and colour-bar label.
    """
    if symmetric:
        m = float(values.abs().max() or 1.0)
        vmin, vmax = -m, m
    else:
        if vmin is None:
            vmin = float(values.min())
        if vmax is None:
            vmax = float(values.max())

    viz.plot_cell_values(
        cells, values, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        legend=False,
    )
    # Restore frame, hide ticks, square aspect, crop to requested window.
    ax.set_axis_on()
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    cx, cy = crop_center_xy
    ax.set_xlim(cx - crop_half_m, cx + crop_half_m)
    ax.set_ylim(cy - crop_half_m, cy + crop_half_m)
    ax.set_title(title)
    viz.add_styled_colorbar(ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str)
                            else cmap,
                            vmin=vmin, vmax=vmax, label=label,
                            size='4%', extend='neither')
