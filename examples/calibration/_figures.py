"""Plot helpers shared across `examples/calibration/` notebooks.

Pure presentation code — graph + data → matplotlib figure. Lives here
rather than inline in each notebook so the notebook flow stays focused
on the substantive (`what aperta does, and how`) bits. Project-specific
styling (highway-tier line widths) that isn't generic enough to belong
in `aperta.visualization`.

Generic primitives — `plot_edge_values`, `add_styled_colorbar` — are in
`aperta.visualization` and used by the wrappers here.

Location-specific parameters (crop centre, zoom width, label) are
passed in by the calling notebook rather than hardcoded here, so the
showcase retargets cleanly when the seed location changes.

Underscore prefix on the module name flags it as project-internal —
not a tutorial example to read, just helpers the notebooks call.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from aperta import visualization as viz


# Inlined to keep the examples self-contained (aperta is OSM-agnostic;
# OSM rank conventions are owned by aperta-atlas, which we deliberately
# don't import from in aperta's example notebooks). Mirrored from
# `aperta_atlas.osm.OSM_HIGHWAY_RANKS` as of v0.3.0a0.
OSM_HIGHWAY_RANKS: dict[str, int] = {
    "motorway": 7, "motorway_link": 7,
    "trunk": 6, "trunk_link": 6,
    "primary": 5, "primary_link": 5,
    "secondary": 4, "secondary_link": 4,
    "tertiary": 3, "tertiary_link": 3,
    "residential": 2, "road": 2,
    "living_street": 1, "pedestrian": 1,
    "unclassified": -1, "service": -1, "busway": -1,
    "cycleway": -1, "footway": -1, "path": -1, "track": -1,
    "steps": -1, "crossing": -1, "disused": -1,
}


# Harmonised font sizes for paper-figure output. Imported by every
# /calibration notebook (via `import _figures as figures`) so the rcParams
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


# Paper-figure export — all figures from /calibration notebooks save here.
# Caller-relative path: notebooks run from `calibration/`, so this resolves
# to `calibration/results/figures_highres/`.
from pathlib import Path as _Path
PAPER_FIGURES_DIR = _Path('results/figures_highres')


def save_figure(fig, name: str, *, ext: str = 'png', dpi: int = 300,
                bbox_inches: str = 'tight'):
    """Save `fig` to `PAPER_FIGURES_DIR / f'{name}.{ext}'` at high DPI.

    Defaults: PNG at 300 DPI with `bbox_inches='tight'` — the right
    choice for raster map content (network plots, choropleths). Pass `ext='pdf'` for vector-friendly content
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

def edge_highway(d) -> str | None:
    """Flatten OSM `highway` tag (may be list-valued post-merge) to a single str."""
    hwy = d.get('highway')
    if isinstance(hwy, list):
        return hwy[0] if hwy else None
    return hwy


# Fill colour for the optional water underlay in `plot_network_map`.
WATER_COLOR = '#dbe6ee'


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
    water=None,
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
        xlim, ylim: optional bbox crop tuples.
        water: optional water polygons (GeoDataFrame / GeoSeries in the
            graph's CRS, e.g. OSM `natural=water`) drawn in light blue
            underneath the network — geographic orientation without a
            tile service or API key.
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

    if water is not None:
        water.plot(ax=ax, color=WATER_COLOR, edgecolor='none', zorder=0)
    viz.plot_edge_values(
        graph, values, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        edge_widths=edge_widths,
        sort_key=lambda key, d: OSM_HIGHWAY_RANKS.get(edge_highway(d), -1),
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
