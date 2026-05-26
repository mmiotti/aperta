"""Plot helpers shared across `examples/extended/` notebooks.

Pure presentation code — graph + data → matplotlib figure. Lives here
rather than inline in each notebook so the notebook flow stays focused
on the substantive (`what aperta does, and how`) bits. Project-specific
styling (Swiss highway-tier line widths, capacity table, Bern crop) that
isn't generic enough to belong in `aperta.visualization`.

Generic primitives — `plot_edge_values`, `add_styled_colorbar` — are in
`aperta.visualization` and used by the wrappers here.

Underscore prefix on the module name flags it as project-internal —
not a tutorial example to read, just helpers the notebooks call.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from aperta import visualization as viz
from aperta.network_processing import HIGHWAY_RANKS


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

# Standard Bern crop window for accessibility cell maps (LV95, metres).
BERN_CX, BERN_CY = 2_600_000, 1_199_000
BERN_ZOOM_HALF = 3_500  # 7 × 7 km window


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
):
    """Draw a per-edge network map with highest-tier roads on top.

    Wraps `aperta.visualization.plot_edge_values` with the Swiss
    aesthetic: per-tier line widths from `HWY_WIDTH`, sorted by
    `HIGHWAY_RANKS` ascending (motorway/trunk land on top of the
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
        sort_key=lambda key, d: HIGHWAY_RANKS.get(edge_highway(d), -1),
    )
    ax.set_facecolor('white')
    viz.add_styled_colorbar(ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str)
                            else cmap,
                            vmin=vmin, vmax=vmax, label=cbar_label)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def plot_bern_cell_map(
    ax,
    cells: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    cmap='viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    title: str,
    label: str = '',
):
    """Plot per-cell `values` on a Bern-cropped 7 × 7 km window.

    Square + framed (axes spines visible, ticks hidden), height-matched
    colour bar, deterministic crop. Wraps `viz.plot_cell_values` with
    the accessibility-notebook aesthetic.

    Args:
        ax: target matplotlib axes.
        cells: GeoDataFrame indexed by cell_id with `geometry`.
        values: per-cell values (Series indexed by cell_id).
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
    # Restore frame, hide ticks, square aspect, crop to Bern window.
    ax.set_axis_on()
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_xlim(BERN_CX - BERN_ZOOM_HALF, BERN_CX + BERN_ZOOM_HALF)
    ax.set_ylim(BERN_CY - BERN_ZOOM_HALF, BERN_CY + BERN_ZOOM_HALF)
    ax.set_title(title)
    viz.add_styled_colorbar(ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str)
                            else cmap,
                            vmin=vmin, vmax=vmax, label=label,
                            size='4%', extend='neither')
