"""
Aperta vs pandana benchmark on the Bern region (walk / car).

**Self-contained** — fetches its OSM network + boundary + building
counts inline via `osmnx`. No pre-computed inputs required. Full run:
~15–30 min (car fetch dominates on first run; OSMnx's built-in file
cache makes subsequent runs faster).

Run from `aperta/examples/benchmarks/`:

    python benchmark.py

Headline numbers from the latest run live in the project README.

## What's measured

Cumulative-opportunity accessibility per mode. Each cell contributes a
**uniform weight of 1** as a destination — this is a routing-engine
benchmark, so realistic weight distributions don't matter and using
uniform weights removes the (expensive, flaky) OSM building fetch that
canton-scale queries frequently drop mid-stream.

The destination structure (3-tier `cells_to_cells + cells_to_zones +
zones_to_zones`) and the routing cutoff (`metric_t_s`) are held constant
across the three aperta variants — only the origin set differs:

  A. **All graph nodes** as origins (pandana-comparable; tiered structure
     can't help when every node is already an origin, so this collapses
     to a single-tier Euclidean-cutoff routing). The Euclidean pair-
     building radius (`variant_a_radius_m` in `MODES`) is tightened
     from `r_zones_m` to `metric_t_s × mode_top_speed × 1.2` — otherwise
     aperta's dense per-origin OD arrays balloon into the tens-of-GB
     range on canton-scale or dense-walk graphs.
  B. **Cell-snap origins** — every unique snap-node referenced by at
     least one cell in the outer polygon. Standard tiered aperta usage;
     3-tier destination structure.
  C. **Inner-polygon cell-snap origins** — variant B further restricted
     to cells whose centroid lies inside the inner polygon (the typical
     production case: a buffer zone provides destinations and
     through-routing but is *not* an origin). Same 3-tier dest structure.
     This is where aperta's tiered-OD advantage shows up.

Pandana (always all-nodes) is the reference baseline for variant A. All
aperta variants route via scipy `csgraph.dijkstra` (aperta's only
routing backend) with `cutoff=metric_t_s` applied uniformly.

## Geometry (per mode)

Walk and car target different regions because the natural analysis
scale differs:

- **Walk**: inner = **city of Bern**, outer = city + 5 km.
- **Car**: inner = **city of Bern**, outer = city + 30 km.

Both modes use the city as the AOI. The buffer differences reflect
each mode's realistic reach in `metric_t_s` (walk: 15 min → ~1 km;
car: 30 min → ~30 km). The car case is where aperta's "compute for a
small AOI inside a much larger network" story is strongest — the
~52 km² city is ~1.4 % of the ~3,600 km² outer polygon.

Cells + zones (Uber H3 res 10 / 8) cover the *outer* polygon; the
inner-origin variant's mask is `centroid.within(inner_polygon)`.
Weights are uniform (1 per cell); cells whose centroid is > 1 km from
the nearest graph node are dropped in `snap_and_filter`.

## Edge durations

- **Walk**: `length / 1.25 m/s` (= 4.5 km/h, roughly matches Bern
  pedestrian survey averages).
- **Car**: `length / max(speed_limit_kph - 15, 10 km/h) → m/s`. The
  15 km/h offset roughly captures (1) baseline peak speed is ~1.25× lower
  than posted limit and (2) more penalty in high-density / signal-heavy
  areas (which coincide with lower speed limits).

Uncalibrated on purpose — the benchmark measures routing-engine cost,
not accessibility-model quality. Matches Pandana's typical usage.
"""
import time

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd
import pandana
from shapely.geometry import Point, Polygon

from aperta import accessibility, geo_processing, network_snap, od_pairs, routing


ox.settings.requests_timeout = 600                    # 10 min per sub-query
# Force Overpass queries to auto-split into smaller sub-queries
ox.settings.max_query_area_size = 1000 * 1000 * 1000
# The main Overpass endpoint (default) closes connections under load for
# canton-scale queries. Mirrors are less loaded. Flip between these if
# one is having a bad day; leave the default commented for reference.
ox.settings.overpass_url = 'https://overpass-api.de/api'         # default (main)
# ox.settings.overpass_url = 'https://overpass.kumi.systems/api'     # Kumi mirror (most popular fallback)
# ox.settings.overpass_url = 'https://overpass.private.coffee/api' # Private.coffee mirror


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CRS_METRIC = 'EPSG:2056'              # LV95 (Swiss metric)

# H3 grids for the cell / zone hierarchy — matches calibration.ipynb.
H3_RES_CELLS = 10                     # ~130 m hex edge
H3_RES_ZONES = 8                      # ~460 m hex edge; parent-child assignment

# Edge-duration priors (see docstring).
WALK_SPEED_MS = 1.25                  # exactly 4.5 km/h
CAR_SPEED_OFFSET_KPH = 15.0           # subtracted from OSM speed limit
CAR_SPEED_MIN_KPH = 10.0              # floor after offset

WEIGHT_COL = 'weight'

# Snap-radius (metres). Cells whose centroid is farther than this from
# every graph node are dropped. Replaces the previous "cells with ≥1 OSM
# building" filter — which is fine for a routing benchmark since it
# measures how much work the engine does, not whether destinations are
# realistic. 1 km drops only very remote / disconnected cells.
MAX_SNAP_RADIUS_M = 1_000.0

# Per-mode settings. Each mode picks its own inner polygon via `place`,
# with a mode-appropriate outer buffer (`buffer_m`) around it — walk
# stays local (city of Bern), car spans the canton.
#   - place:      Nominatim query for the inner polygon.
#   - place_label: printed for the benchmark banner.
#   - buffer_m:   outer-polygon buffer added to inner polygon (in EPSG:2056).
#   - network_type: osmnx `network_type` (`'walk'` or `'drive'`).
#   - r_cells_m:  cells_to_cells outer radius.
#   - r_medium_m: cells_to_zones outer radius. `None` → auto-infer as
#                  `min(r_cells_m * 10, r_zones_m)`.
#   - r_zones_m:  zones_to_zones outer radius (Euclidean dest-distance cap).
#   - metric_t_s: cumulative-opportunity time threshold; also the routing
#                  cutoff passed to Dijkstra (correctness-preserving — no
#                  destinations beyond this time can contribute).
#   - variant_a_radius_m: Euclidean radius used by variant A (all-nodes)
#                  when building candidate OD pairs. Kept tight — matches
#                  what's actually reachable in `metric_t_s` at the mode's
#                  plausible top speed (walk ~1.25 m/s, car ~100 km/h avg)
#                  with a 1.2× safety margin. Passing the wider `r_zones_m`
#                  here would blow up RAM into the hundreds of GB range for
#                  canton-scale car / dense walk graphs (Dijkstra prunes,
#                  but `od_pairs.get_pairs` pre-materialises all pairs
#                  within the Euclidean cap).
MODES = {
    'walk': dict(
        place='Bern, Switzerland',
        place_label='Bern (city)',
        buffer_m=5_000,
        network_type='walk',
        r_cells_m=1_000,
        r_medium_m=2_500,
        r_zones_m=5_000,
        metric_t_s=15 * 60,
        variant_a_radius_m=1_700,     # 15 min × 1.5 m/s × 1.2 ≈ 1.7 km
    ),
    'car': dict(
        place='Bern, Switzerland',
        place_label='Bern (city)',
        buffer_m=30_000,               # Buffer is the point — preserves
                                       # the "small AOI, huge outer" story.
                                       # AOI = city (~52 km²), outer ≈
                                       # 3,600 km² → buffer/AOI ratio ~50×,
                                       # a stronger aperta demo than the
                                       # canton + 30 km original (2.5×).
        network_type='drive',
        r_cells_m=1_000,
        r_medium_m=10_000,
        r_zones_m=50_000,
        metric_t_s=15 * 60,
        variant_a_radius_m=30_000,    # 15 min × 100 km/h × 1.2 → 30 km
    ),
}


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def fetch_inner_polygon(place: str):
    """Nominatim query → metric inner polygon in `CRS_METRIC` (LV95)."""
    boundary_ll = ox.geocode_to_gdf(place).dissolve()
    return boundary_ll.to_crs(CRS_METRIC).geometry.iloc[0]


def outer_polygon_ll(inner_polygon, buffer_m):
    """Inner buffered by `buffer_m` metres, returned as a WGS84 polygon
    for OSMnx queries."""
    outer_m = inner_polygon.buffer(buffer_m)
    return gpd.GeoSeries([outer_m], crs=CRS_METRIC).to_crs('EPSG:4326').iloc[0]


def bake_edge_times(graph, mode):
    """Write per-edge travel time (seconds) into `<mode>_time_s`.

    walk: `length / 1.25` (m / (m/s)).
    car:  `length / (max(speed_kph − 15, 10) × 1000/3600)`.
    """
    attr = f'{mode}_time_s'
    if mode == 'walk':
        for u, v, k, d in graph.edges(keys=True, data=True):
            d[attr] = float(d['length']) / WALK_SPEED_MS
    else:  # car
        for u, v, k, d in graph.edges(keys=True, data=True):
            limit_kph = float(d.get('speed_kph', 50.0))
            eff_kph = max(limit_kph - CAR_SPEED_OFFSET_KPH, CAR_SPEED_MIN_KPH)
            d[attr] = float(d['length']) / (eff_kph * 1000.0 / 3600.0)


def build_cells_zones(outer_polygon_ll_geom):
    """H3-res-10 cells + res-8 zones covering the outer polygon; parent-child
    gives the cell → zone assignment for free (same pattern as
    calibration.ipynb §2)."""
    cells = geo_processing.build_h3_grid(
        outer_polygon_ll_geom, H3_RES_CELLS,
        polygon_crs='EPSG:4326', target_crs=CRS_METRIC,
    )
    cells['zone_id'] = [h3.cell_to_parent(c, H3_RES_ZONES) for c in cells.index]
    zone_ids = sorted(cells['zone_id'].unique())
    zones = gpd.GeoDataFrame(
        {'geometry': [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(z)])
                      for z in zone_ids]},
        index=pd.Index(zone_ids, name='zone_id'),
        crs='EPSG:4326',
    ).to_crs(CRS_METRIC)
    return cells, zones


def snap_and_filter(cells, zones, graph, node_col):
    """Snap cell + zone centroids to nearest graph node within
    `MAX_SNAP_RADIUS_M`; drop unsnappable rows and cells whose zone
    dropped out. Assign a uniform weight of 1 per cell (aggregated
    cell-count per zone) — this is a routing benchmark, so realistic
    weight distributions don't matter."""
    cells[node_col], _ = network_snap.snap_to_network_nodes(
        gpd.GeoDataFrame(geometry=cells.geometry.centroid, index=cells.index, crs=cells.crs),
        graph, max_distance=MAX_SNAP_RADIUS_M)
    zones[node_col], _ = network_snap.snap_to_network_nodes(
        gpd.GeoDataFrame(geometry=zones.geometry.centroid, index=zones.index, crs=zones.crs),
        graph, max_distance=MAX_SNAP_RADIUS_M)
    cells = cells.dropna(subset=[node_col]).copy()
    zones = zones.dropna(subset=[node_col]).copy()
    cells[node_col] = cells[node_col].astype(int)
    zones[node_col] = zones[node_col].astype(int)
    cells = cells[cells['zone_id'].isin(zones.index)].copy()
    cells[WEIGHT_COL] = 1
    zones[WEIGHT_COL] = cells.groupby('zone_id').size().reindex(
        zones.index, fill_value=0).astype(int)
    return cells, zones


def build_mode_data(mode, cfg):
    """Fetch inner polygon + OSM network + H3 cells + zones for one mode.
    Returns (graph, cells, zones, inner_polygon). No building fetch —
    weights are uniform (1 per cell), assigned in `snap_and_filter`."""
    t0 = time.perf_counter()
    inner_polygon = fetch_inner_polygon(cfg['place'])
    outer_ll = outer_polygon_ll(inner_polygon, cfg['buffer_m'])

    graph = ox.graph_from_polygon(outer_ll, network_type=cfg['network_type'], simplify=True,
                                  retain_all=False)
    graph = ox.project_graph(graph, to_crs=CRS_METRIC)
    if mode == 'car':
        graph = ox.add_edge_speeds(graph)
    else:
        # Pedestrian graphs are MultiDiGraph even for walk — respecting one-way
        # would give zero-accessibility outliers at one-way termini. Undirect
        # AFTER project_graph.
        graph = ox.convert.to_undirected(graph)
    bake_edge_times(graph, mode)

    cells, zones = build_cells_zones(outer_ll)
    print(f"  {mode} fetch + build: {time.perf_counter() - t0:.1f} s   "
          f"(graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges, "
          f"cells (unfiltered): {len(cells):,})")
    return graph, cells, zones, inner_polygon


# ----------------------------------------------------------------------------
# Variant runners — each returns total wall-clock seconds for the full
# end-to-end pipeline (setup + routing + accessibility metric).
# ----------------------------------------------------------------------------
def graph_to_pandana_dfs(graph, time_attr):
    """Node / edge tables for pandana. Node IDs forced to int64 (pandana's
    C++ backend rejects strings / floats). Multi-graph parallels collapse
    to one edge per (u, v) via the minimum-weight parallel — matches
    aperta's routing choice."""
    nodes = pd.DataFrame(
        [(int(n), float(d['x']), float(d['y']))
         for n, d in graph.nodes(data=True)],
        columns=['node_id', 'x', 'y'],
    ).set_index('node_id')
    edges = pd.DataFrame(
        [(int(u), int(v), float(d[time_attr]))
         for u, v, _, d in graph.edges(keys=True, data=True)],
        columns=['from', 'to', time_attr],
    )
    edges = edges.groupby(['from', 'to'], as_index=False)[time_attr].min()
    return nodes['x'], nodes['y'], edges


def run_pandana(graph, mode, time_attr, cells, node_col, t_metric):
    """Pandana baseline: contraction-hierarchy precompute + aggregate."""
    node_x, node_y, edges_df = graph_to_pandana_dfs(graph, time_attr)
    t0 = time.perf_counter()
    net = pandana.Network(
        node_x=node_x, node_y=node_y,
        edge_from=edges_df['from'], edge_to=edges_df['to'],
        edge_weights=edges_df[[time_attr]],
        twoway=(mode == 'walk'),    # walk graph is undirected; car is directed
    )
    net.precompute(t_metric)
    per_node = (cells.groupby(node_col)[WEIGHT_COL].sum()
                .reindex(node_x.index, fill_value=0.0))
    net.set(per_node.index.to_series(), variable=per_node.values, name='w')
    net.aggregate(t_metric, type='sum', name='w', decay='flat')
    return time.perf_counter() - t0


def run_aperta_all_nodes(graph, time_attr, cells, node_col, r_outer_m, t_metric):
    """Variant A: every graph node is its own 'cell'. Single-tier — the
    tiered destination structure can't help when every node is already an
    origin. Memory-heavy at country / canton scale; the more natural
    aperta use case is variant B/C below."""
    t0 = time.perf_counter()
    node_ids = list(graph.nodes())
    weights = (cells.groupby(node_col)[WEIGHT_COL].sum()
               .reindex(node_ids, fill_value=0.0))
    nodes_gdf = gpd.GeoDataFrame(
        {'node_id_synth': node_ids, WEIGHT_COL: weights.values},
        index=pd.Index(node_ids, name='synth_cell_id'),
        geometry=[Point(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in node_ids],
        crs=cells.crs,
    )
    pairs = od_pairs.get_pairs(
        nodes_gdf, r_cells=r_outer_m, node_column='node_id_synth',
    )
    costs = routing.tiered_path_costs(graph, pairs, weight=time_attr, cutoff=t_metric)
    w_vals = od_pairs.lookup_dest_column_node(WEIGHT_COL, pairs, nodes_gdf,
                                  node_column='node_id_synth')
    accessibility.cumulative_opportunities(
        costs, {'w': w_vals}, {},
        [accessibility.Bin('in_T', 0, t_metric)],
    )
    return time.perf_counter() - t0


def run_aperta_tiered(graph, time_attr, cells, zones, node_col,
                      r_cells_m, r_medium_m, r_zones_m, t_metric,
                      *, orig_cells=None):
    """Variants B and C: 3-tier destination structure, with optional
    `orig_cells` mask:

    - `orig_cells=None` → variant B (every cell in the outer polygon is
      an origin).
    - `orig_cells=<bool array aligned to cells.index>` → variant C
      (typical production case: cells whose centroid is inside the
      inner polygon).
    """
    t0 = time.perf_counter()
    pairs = od_pairs.get_pairs(
        cells, r_cells=r_cells_m, node_column=node_col,
        zones=zones, r_zones=r_zones_m, r_medium=r_medium_m,
        orig_cells=orig_cells,
    )
    costs = routing.tiered_path_costs(graph, pairs, weight=time_attr, cutoff=t_metric)
    pairs_geo, costs_geo = od_pairs.reindex_by_geo_unit(
        pairs, costs, cells,
        cell_node_column=node_col, zones=zones, zone_node_column=node_col,
        r_cells=r_cells_m, r_medium=r_medium_m, r_zones=r_zones_m,
    )
    w_geo = od_pairs.lookup_dest_column_geo(WEIGHT_COL, pairs_geo, cells, zones=zones)
    accessibility.cumulative_opportunities(
        costs_geo, {'w': w_geo}, cells['zone_id'].to_dict(),
        [accessibility.Bin('in_T', 0, t_metric)],
    )
    return time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Per-mode driver
# ----------------------------------------------------------------------------
def bench_one_mode(mode, cfg):
    """Fetch + prep data for `mode`, then run pandana + outer/inner
    aperta variants. Returns timings dict."""
    r_medium_label = (f"{cfg['r_medium_m']/1000:.0f} km" if cfg['r_medium_m']
                      else f"auto≈{min(cfg['r_cells_m']*10, cfg['r_zones_m'])/1000:.0f} km")
    metric_t_s = int(cfg['metric_t_s'])
    print(f"\n{'='*70}\n{mode.upper()}  "
          f"({cfg['place_label']}; cumulative {WEIGHT_COL} within "
          f"{metric_t_s // 60} min; buffer={cfg['buffer_m']/1000:.0f} km; "
          f"r_cells={cfg['r_cells_m']/1000:.0f} km, r_medium={r_medium_label}, "
          f"r_zones={cfg['r_zones_m']/1000:.0f} km)\n{'='*70}")

    graph, cells, zones, inner_polygon = build_mode_data(mode, cfg)
    node_col = f'node_id_{mode}'
    time_attr = f'{mode}_time_s'
    cells, zones = snap_and_filter(cells, zones, graph, node_col)

    in_inner = cells.geometry.centroid.within(inner_polygon).to_numpy()
    n_inner = int(in_inner.sum())
    print(f"  Origin universe — graph nodes: {graph.number_of_nodes():,} · "
          f"cells: {len(cells):,} ({cells[node_col].nunique():,} unique snap-nodes) · "
          f"inner-polygon cells: {n_inner:,} ({100*in_inner.mean():.1f}%)\n")

    timings = {}
    print(f"  Pandana                                       ...", end=' ', flush=True)
    timings['pandana'] = run_pandana(graph, mode, time_attr, cells, node_col,
                                     cfg['metric_t_s'])
    print(f"→ {timings['pandana']:6.1f} s")

    print(f"  [A] Aperta all nodes                          ...", end=' ', flush=True)
    timings['A'] = run_aperta_all_nodes(graph, time_attr, cells, node_col,
                                        cfg['variant_a_radius_m'], cfg['metric_t_s'])
    print(f"→ {timings['A']:6.1f} s  ({graph.number_of_nodes():,} origins, "
          f"r={int(cfg['variant_a_radius_m']) / 1000:.1f} km)")

    print(f"  [B] Aperta cell-snap (outer origins)          ...", end=' ', flush=True)
    timings['B'] = run_aperta_tiered(
        graph, time_attr, cells, zones, node_col,
        cfg['r_cells_m'], cfg['r_medium_m'], cfg['r_zones_m'], cfg['metric_t_s'])
    print(f"→ {timings['B']:6.1f} s  ({cells[node_col].nunique():,} origins)")

    print(f"  [C] Aperta cell-snap (inner origins, 'AOI')   ...", end=' ', flush=True)
    timings['C'] = run_aperta_tiered(
        graph, time_attr, cells, zones, node_col,
        cfg['r_cells_m'], cfg['r_medium_m'], cfg['r_zones_m'], cfg['metric_t_s'],
        orig_cells=in_inner)
    print(f"→ {timings['C']:6.1f} s  ({n_inner:,} origins)")

    return timings


# ----------------------------------------------------------------------------
# Main + summary
# ----------------------------------------------------------------------------
SUMMARY_ROWS = [
    ('Pandana — all graph nodes',                                        'pandana'),
    ('Aperta A — all graph nodes (single-tier, Euclidean cutoff)',       'A'),
    ('Aperta B — cell-snap origins, tiered destinations',                'B'),
    ('Aperta C — AOI-restricted cell origins, tiered destinations',      'C'),
]


def print_summary(results):
    """Final pivot table — direct copy-paste source for the README."""
    print(f"\n{'='*70}\nSUMMARY (wall-clock seconds, lower = better)\n{'='*70}")
    headers = [f"{m} ({int(MODES[m]['metric_t_s']) // 60} min)" for m in results]
    label_w = max(len(r[0]) for r in SUMMARY_ROWS) + 2
    col_w = max(max(len(h) for h in headers), 10) + 2
    print(' ' * label_w + ''.join(f"{h:>{col_w}}" for h in headers))
    for label, key in SUMMARY_ROWS:
        cells_str = ''.join(f"{results[m][key]:>{col_w-2}.1f}s "
                            for m in results)
        print(f"{label:{label_w}}{cells_str}")


def main():
    print(f"Modes: {list(MODES)}")
    for mode, cfg in MODES.items():
        print(f"  {mode:5}: inner={cfg['place_label']!r}  "
              f"outer=+{int(cfg['buffer_m']) / 1000:.0f} km")
    results = {mode: bench_one_mode(mode, cfg) for mode, cfg in MODES.items()}
    print_summary(results)


if __name__ == '__main__':
    main()
