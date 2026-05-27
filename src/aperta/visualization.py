"""
Plotting helpers for the patterns that recur across aperta notebooks.

Two patterns dominate accessibility-notebook visualisations:

1. **Cell choropleth** — colour each cell by a per-cell value (typically an
   accessibility metric or some derived per-cell quantity).
   `plot_cell_values` handles the single-panel case;
   `plot_cell_values_comparison` handles side-by-side multi-panel with a
   shared colour scale (before/after, walk vs car, etc.).

2. **Tiered destination viz** — for one origin cell, draw the origin, its
   cell-tier dests, its zone-tier dests, and the underlying network nodes.
   `plot_tiered_destinations` packages this.

These helpers exist to keep notebook cells focused on the substantive
computation rather than matplotlib boilerplate. They're not meant to cover
every plotting need — drop down to matplotlib directly for one-off custom
visuals. Future work can extend this module with helpers for path-feature
maps, density choropleths, route-line overlays, and so on.
"""
from typing import Any, Callable

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODPairs


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A per-cell value series — either a pd.Series indexed by cell_id, a dict
# {cell_id -> value}, or the name of a column already on the `cells`
# GeoDataFrame.
CellValues = pd.Series | dict | str

# Optional overlays: list of (GeoDataFrame, kwargs-for-GeoDataFrame.plot)
# tuples. Drawn on top of the choropleth in the order given.
Overlays = list[tuple[gpd.GeoDataFrame, dict]] | None


# ---------------------------------------------------------------------------
# Cell choropleths
# ---------------------------------------------------------------------------

def plot_cell_values(
    cells: gpd.GeoDataFrame,
    values: CellValues,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str = 'viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    missing_color: str = 'lightgrey',
    overlays: Overlays = None,
    boundary: gpd.GeoDataFrame | None = None,
    legend: bool = True,
    treat_neg_inf_as_missing: bool = True,
) -> plt.Axes:
    """Colour each cell by a per-cell value (single-panel choropleth).

    Args:
        cells: cell-indexed GeoDataFrame (index = cell_id; geometry column
            present).
        values: per-cell values, one of: `pd.Series` keyed by cell_id, a
            `dict` keyed by cell_id, or the name of a column already on
            `cells`. Cells absent from a Series / dict are drawn as
            missing (`missing_color`).
        ax: existing axes; if `None`, a new figure + axes is created.
        title: optional axes title.
        cmap, vmin, vmax: standard matplotlib choropleth args. `vmin` /
            `vmax` `None` → auto-scaled from the data.
        missing_color: fill colour for cells with NaN value.
        overlays: list of `(GeoDataFrame, kwargs)` tuples to overlay on top
            of the choropleth in order (e.g. supermarkets, routes, network).
        boundary: optional area-of-interest polygon to outline on top
            (drawn after overlays).
        legend: show the colour-scale legend.
        treat_neg_inf_as_missing: replace `-inf` with NaN before plotting.
            Useful for logsum outputs where unreachable cells produce
            `-inf` (drawn as missing rather than as a colour-bar extreme).

    Returns:
        The Axes object the plot was drawn on.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 8))

    if isinstance(values, str):
        cells_plot = cells
        column = values
    else:
        s = values if isinstance(values, pd.Series) else pd.Series(values)
        if treat_neg_inf_as_missing:
            s = s.replace(-np.inf, np.nan)
        column = '__plot_value__'
        cells_plot = cells.join(s.rename(column))

    cells_plot.plot(
        column=column, ax=ax, legend=legend,
        cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none',
        missing_kwds={'color': missing_color},
    )
    if overlays:
        for overlay, kw in overlays:
            overlay.plot(ax=ax, **kw)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color='black', linewidth=0.5)
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def plot_cell_values_comparison(
    cells: gpd.GeoDataFrame,
    values_by_label: dict[str, CellValues],
    *,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = 'viridis',
    missing_color: str = 'lightgrey',
    overlays: Overlays = None,
    boundary: gpd.GeoDataFrame | None = None,
    shared_scale: bool = True,
    treat_neg_inf_as_missing: bool = True,
    legend: bool = True,
) -> plt.Figure:
    """Side-by-side cell choropleths with a shared colour scale.

    Args:
        cells: cell-indexed GeoDataFrame.
        values_by_label: ordered dict `{panel_title -> per-cell values}`.
            One panel per entry. Values may be `pd.Series` / `dict` /
            column-name (same as `plot_cell_values`).
        suptitle: optional figure-level super-title.
        figsize: figure size; defaults to `(7 × n_panels, 7)`.
        cmap, missing_color, overlays, boundary, legend,
            treat_neg_inf_as_missing: see `plot_cell_values`.
        shared_scale: if True, all panels share `vmin` / `vmax` derived
            from the global min/max across all value series. If False,
            each panel auto-scales independently.

    Returns:
        The Figure object. (Each subplot Axes is accessible via
        `fig.axes`.)
    """
    n = len(values_by_label)
    if n == 0:
        raise ValueError("`values_by_label` must contain at least one entry.")
    if figsize is None:
        figsize = (7 * n, 7)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    vmin = vmax = None
    if shared_scale:
        all_vals: list[np.ndarray] = []
        for vals in values_by_label.values():
            if isinstance(vals, str):
                s = pd.Series(cells[vals])
            else:
                s = vals if isinstance(vals, pd.Series) else pd.Series(vals)
            if treat_neg_inf_as_missing:
                s = s.replace(-np.inf, np.nan)
            all_vals.append(np.asarray(s.dropna().values, dtype=float))
        if all_vals:
            cat = np.concatenate(all_vals)
            if cat.size > 0:
                vmin = float(np.min(cat))
                vmax = float(np.max(cat))

    for ax, (title, vals) in zip(axes, values_by_label.items()):
        plot_cell_values(
            cells, vals, ax=ax, title=title, cmap=cmap,
            vmin=vmin, vmax=vmax, missing_color=missing_color,
            overlays=overlays, boundary=boundary,
            treat_neg_inf_as_missing=treat_neg_inf_as_missing,
            legend=legend,
        )
    if suptitle:
        plt.suptitle(suptitle, y=1.02, fontsize=12)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tiered destinations viz
# ---------------------------------------------------------------------------

def plot_tiered_destinations(
    cells: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    pairs: TieredODPairs,
    origin_cell_id,
    *,
    cell_node_column: str = 'node_id',
    zone_node_column: str = 'node_id',
    graph=None,
    boundary: gpd.GeoDataFrame | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Visualise the tiered OD structure from one origin cell.

    Draws, in z-order back-to-front:
        - All cells in pale grey (background).
        - All zone boundaries (faint outlines).
        - **Far-tier** destination zones (`zones_to_zones`) for the origin's
          zone (pale blue fill).
        - **Middle-tier** destination zones (`cells_to_zones`) for the
          origin cell (mid blue-green fill — distinct from far-tier so the
          two layers read separately when they overlap).
        - **Cell-tier** destination cells (`cells_to_cells`) for the origin
          (gold fill).
        - The origin cell itself (red).
        - (If `graph` given) Network nodes as markers: origin (red star),
          cell-tier dest nodes (small orange dots), middle-tier dest zone
          nodes (medium teal squares), far-tier dest zone nodes (larger
          blue squares).

    Currently supports `TieredODNodePairs` only (the node-keyed pairs
    returned by `od_pairs.get_pairs`). For geo-keyed pairs, lookup happens
    via cell_id / zone_id directly — straightforward to add when needed.

    Args:
        cells: cell-indexed GeoDataFrame. Must have `cell_node_column` and
            a `'zone_id'` column.
        zones: zone-indexed GeoDataFrame. Must have `zone_node_column`.
        pairs: `TieredODNodePairs` (node-keyed). `cells_to_cells` and
            `cells_to_zones` keys are cell-tier network nodes;
            `zones_to_zones` keys are zone-tier network nodes.
        origin_cell_id: index value of the cell to highlight as the
            origin (looked up in `cells`).
        cell_node_column: column on `cells` carrying the cell-tier network
            node ID. Default `'node_id'`.
        zone_node_column: column on `zones` carrying the zone-tier network
            node ID. Default `'node_id'`.
        graph: optional networkx graph (or any object with a `.nodes`
            mapping providing `'x'` / `'y'` attributes per node). Enables
            the node-marker overlay; omit to draw cells/zones only.
        boundary: optional AOI boundary to outline on top.
        ax: existing axes; if `None`, a new figure + axes is created.
        title: axes title.
        legend: show the marker legend (only meaningful when `graph` is
            given).

    Returns:
        The Axes object the plot was drawn on.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10))

    origin_node = cells.loc[origin_cell_id, cell_node_column]
    origin_zone_id = cells.loc[origin_cell_id, 'zone_id']
    origin_zone_node = zones.loc[origin_zone_id, zone_node_column]

    cell_dest_nodes = pairs.cells_to_cells.get(origin_node, np.array([]))
    cell_dest_cells = cells[cells[cell_node_column].isin(cell_dest_nodes)]
    middle_dest_nodes = (pairs.cells_to_zones.get(origin_node, np.array([]))
                         if pairs.cells_to_zones is not None else np.array([]))
    middle_dest_zones = zones[zones[zone_node_column].isin(middle_dest_nodes)]
    far_dest_nodes = (pairs.zones_to_zones.get(origin_zone_node, np.array([]))
                      if pairs.zones_to_zones is not None else np.array([]))
    far_dest_zones = zones[zones[zone_node_column].isin(far_dest_nodes)]

    cells.plot(ax=ax, color='whitesmoke', edgecolor='lightgray', linewidth=0.1)
    zones.boundary.plot(ax=ax, color='gray', linewidth=0.3, alpha=0.5)
    far_dest_zones.plot(ax=ax, color='lightblue', alpha=0.5,
                        edgecolor='steelblue', linewidth=0.5)
    middle_dest_zones.plot(ax=ax, color='mediumaquamarine', alpha=0.55,
                           edgecolor='seagreen', linewidth=0.5)
    cell_dest_cells.plot(ax=ax, color='gold', alpha=0.6,
                         edgecolor='darkorange', linewidth=0.2)
    cells.loc[[origin_cell_id]].plot(ax=ax, color='red',
                                     edgecolor='darkred', linewidth=1.5)

    if graph is not None:
        cell_xy = [(graph.nodes[n]['x'], graph.nodes[n]['y'])
                   for n in cell_dest_nodes if n in graph.nodes]
        if cell_xy:
            ax.scatter(*zip(*cell_xy), color='darkorange', s=4, marker='o',
                       zorder=5,
                       label=f'Cell-tier dest nodes ({len(cell_xy)})')
        middle_xy = [(graph.nodes[n]['x'], graph.nodes[n]['y'])
                     for n in middle_dest_nodes if n in graph.nodes]
        if middle_xy:
            ax.scatter(*zip(*middle_xy), color='seagreen', s=25, marker='s',
                       edgecolor='black', linewidth=0.3, zorder=6,
                       label=f'Middle-tier dest nodes ({len(middle_xy)})')
        far_xy = [(graph.nodes[n]['x'], graph.nodes[n]['y'])
                  for n in far_dest_nodes if n in graph.nodes]
        if far_xy:
            ax.scatter(*zip(*far_xy), color='steelblue', s=40, marker='s',
                       edgecolor='black', linewidth=0.3, zorder=7,
                       label=f'Far-tier dest nodes ({len(far_xy)})')
        if origin_node in graph.nodes:
            ax.scatter([graph.nodes[origin_node]['x']],
                       [graph.nodes[origin_node]['y']],
                       color='red', s=150, marker='*',
                       edgecolor='black', linewidth=0.5,
                       zorder=10, label='Origin node')

    if boundary is not None:
        boundary.boundary.plot(ax=ax, color='black', linewidth=0.5)
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    if legend and graph is not None:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    return ax


# ---------------------------------------------------------------------------
# Generic per-edge plot + colorbar pattern
# ---------------------------------------------------------------------------


def add_styled_colorbar(ax, *, cmap, vmin: float, vmax: float, label: str,
                        size: str = '3%', pad: float = 0.10, extend: str = 'max'):
    """Append a height-matched colour bar to the right of `ax`.

    Wraps the `mpl_toolkits.axes_grid1.make_axes_locatable` +
    `ScalarMappable` pattern that recurs in every map cell. Factored out
    so notebook cells don't repeat the four-line ritual.

    Args:
        ax: target axes; colour bar appends to its right via
            `make_axes_locatable`.
        cmap: matplotlib colormap (Colormap or name).
        vmin, vmax: data range the colour bar covers.
        label: colour bar label.
        size: width of colour bar relative to `ax` (matplotlib syntax,
            default `'3%'`).
        pad: padding between `ax` and the colour bar in inches.
        extend: arrow indicator for out-of-range values
            (`'neither'`, `'min'`, `'max'`, `'both'`).

    Returns:
        The newly-created colour-bar axes object.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=pad)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, cax=cax, label=label, extend=extend)
    return cax


def plot_edge_values(
    graph: nx.MultiDiGraph,
    values: dict | pd.Series,
    *,
    ax=None,
    cmap='viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    edge_widths: dict[tuple, float] | float = 1.0,
    sort_key: Callable[[tuple, dict], Any] | None = None,
    default_value: float = 0.0,
):
    """Plot per-edge values on a network using a `LineCollection`.

    The `LineCollection` approach (over per-edge `ax.plot`) is ~10× faster
    on large graphs and lets the caller control draw order by sorting
    edges before adding them to the collection. `sort_key=` is the
    z-order knob: for a "high-tier roads on top" effect on an OSM-derived
    graph, sort ascending by `HIGHWAY_RANKS` and the motorways land
    last (= drawn on top of) the residential mesh.

    Uses `d['geometry']` from each edge if present (always true post-
    `network_processing.consolidate_intersections`); falls back to a
    straight line between endpoint node positions otherwise.

    Args:
        graph: a networkx graph with `x`/`y` on every node and
            optionally `geometry` (LineString) on each edge.
        values: per-edge value mapping `(u, v, k) -> float`, either a
            dict or a `pd.Series` with a MultiIndex of edge keys.
            Missing keys get `default_value`.
        ax: matplotlib axes; new figure created if `None`.
        cmap: matplotlib colormap (Colormap or name).
        vmin, vmax: colour-scale range. `None` auto-derives from data.
        edge_widths: per-edge line width — scalar applies uniformly,
            dict maps `(u, v, k) -> width`. Defaults to 1.0.
        sort_key: optional `(edge_key, edge_data) -> sortable` callable
            that controls draw order (ascending → drawn last on top).
            `None` preserves graph iteration order.
        default_value: fallback for edges missing from `values`.

    Returns:
        The matplotlib `Axes`. Caller is responsible for the colour bar
        (use `add_styled_colorbar`), title, axes limits, aspect, etc.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if isinstance(values, pd.Series):
        values_dict = values.to_dict()
    else:
        values_dict = values

    edges = list(graph.edges(keys=True, data=True))
    if sort_key is not None:
        edges = sorted(edges, key=lambda e: sort_key((e[0], e[1], e[2]), e[3]))

    geoms: list = []
    widths: list = []
    vals: list = []
    for u, v, k, d in edges:
        geom = d.get('geometry')
        if geom is not None:
            coords = list(geom.coords)
        else:
            coords = [(graph.nodes[u]['x'], graph.nodes[u]['y']),
                      (graph.nodes[v]['x'], graph.nodes[v]['y'])]
        geoms.append(coords)
        if isinstance(edge_widths, dict):
            widths.append(float(edge_widths.get((u, v, k), 1.0)))
        else:
            widths.append(float(edge_widths))
        vals.append(float(values_dict.get((u, v, k), default_value)))

    vals_arr = np.asarray(vals)
    if vmin is None:
        vmin = float(vals_arr.min()) if vals_arr.size else 0.0
    if vmax is None:
        vmax = float(vals_arr.max()) if vals_arr.size else 1.0
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cmap_obj(norm(vals_arr))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    lc = LineCollection(geoms, colors=colors, linewidths=widths)
    ax.add_collection(lc)
    return ax
