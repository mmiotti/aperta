"""
Aperta vs pandana benchmark on the Bern + 40 km consolidated graphs.

**Requires `examples/extended/prepare/` to have been run first** — this
script consumes `data/prepared/{walk,car}_graph.graphml`, `cells.gpkg`,
`zones.gpkg`, and `aoi_polygon.gpkg`. With prep done the full benchmark
runs in ~5–10 min; the prep itself takes the better part of an hour
(network download + consolidation).

Run from `aperta/examples/extended/`:

    python benchmark.py

Headline numbers from the latest run live in the project README.

## What's measured

Cumulative-opportunity accessibility to total employment, per mode. The
story this version tells: how does aperta's runtime scale with the
**origin set size**? The destination structure (3-tier `cells_to_cells +
cells_to_zones + zones_to_zones`) and the routing cutoff (`metric_t_s`)
are held constant across the three aperta variants — only the origin
set differs:

  A. **All graph nodes** as origins (pandana-comparable; tiered structure
     can't help when every node is already an origin, so this collapses
     to a single-tier Euclidean-cutoff routing).
  B. **Cell-snap origins** — every unique snap-node referenced by at
     least one cell. Standard tiered aperta usage; 3-tier destination
     structure.
  C. **AOI cell-snap origins** — variant B further restricted to cells
     whose centroid lies inside the AOI polygon (the typical production
     case: a buffer zone around the AOI provides destinations and
     through-routing but is *not* an origin). Same 3-tier dest structure.

Pandana (always all-nodes) is the reference baseline for variant A. All
aperta variants route via scipy `csgraph.dijkstra` (aperta's only
routing backend) with `cutoff=metric_t_s` applied uniformly.

Toggle `TEST_MODE = True` at the top of this file for a fast end-to-end
smoke test on a small bbox subset (~30 s); leave `TEST_MODE = False` for
the full Bern + 40 km run.
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pandana
from shapely.geometry import Point, box

from aperta import accessibility, network_processing, od_pairs, routing


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PREPARED_DIR = Path('data/prepared')
WEIGHT_COL = 'employment_total'

TEST_MODE = False              # True → small bbox subset; False → full Bern
TEST_BBOX_HALF_KM = 3.0        # half-size of the test bbox around AOI centroid

# Per-mode settings.
#   - speed_kph:  fallback per-edge speed; `None` = use per-edge `speed_kph` attr.
#   - r_cells_m:  cells_to_cells outer radius.
#   - r_medium_m: cells_to_zones outer radius. `None` → auto-infer as
#                  `min(r_cells_m * 10, r_zones_m)`.
#   - r_zones_m:  zones_to_zones outer radius (Euclidean dest-distance cap).
#   - metric_t_s: cumulative-opportunity time threshold; also the routing
#                  cutoff passed to Dijkstra (correctness-preserving — no
#                  destinations beyond this time can contribute to the metric).
MODES = {
    'walk': dict(
        graph_file='walk_graph.graphml',
        speed_kph=5.0,
        r_cells_m=1_000,
        r_medium_m=None,
        r_zones_m=2_000,
        metric_t_s=15 * 60,
    ),
    'car': dict(
        graph_file='car_graph.graphml',
        speed_kph=None,
        r_cells_m=1_000,
        r_medium_m=10_000,
        r_zones_m=50_000,
        metric_t_s=30 * 60,
    ),
}


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def bake_edge_times(graph, mode, fallback_kph):
    """Write per-edge travel time (seconds) into `<mode>_time_s`."""
    attr = f'{mode}_time_s'
    for u, v, k, data in graph.edges(keys=True, data=True):
        length = float(data['length'])
        speed_kph = (float(data.get('speed_kph', 30.0)) if fallback_kph is None
                     else fallback_kph)
        data[attr] = length / (speed_kph * 1000.0 / 3600.0)


def snap_cells(cells, graph, node_col):
    """Add `node_col` to cells via centroid → nearest network node.
    Overwrites any existing column (used to re-snap after TEST_MODE clipping)."""
    centroids = gpd.GeoDataFrame(
        geometry=cells.geometry.centroid, index=cells.index, crs=cells.crs,
    )
    cells[node_col], _ = network_processing.snap_to_network_nodes(centroids, graph)


def clip_to_test_bbox(cells, zones, graph, aoi_polygon, half_km):
    """Clip cells, zones, graph, and AOI to a square box around the cells'
    centroid. Used by `TEST_MODE` for fast smoke tests."""
    centre = cells.union_all().centroid
    half_m = half_km * 1000
    minx, miny = centre.x - half_m, centre.y - half_m
    maxx, maxy = centre.x + half_m, centre.y + half_m
    cells_sub = cells.cx[minx:maxx, miny:maxy].copy()
    zones_sub = zones.cx[minx:maxx, miny:maxy].copy()
    cells_sub = cells_sub[cells_sub['zone_id'].isin(zones_sub.index)]
    keep_nodes = [n for n, d in graph.nodes(data=True)
                  if minx <= d['x'] <= maxx and miny <= d['y'] <= maxy]
    graph_sub = graph.subgraph(keep_nodes).copy()
    aoi_sub = aoi_polygon.intersection(box(minx, miny, maxx, maxy))
    return cells_sub, zones_sub, graph_sub, aoi_sub


# ----------------------------------------------------------------------------
# Variant runners — each returns total wall-clock seconds for the full
# end-to-end pipeline (setup + routing + accessibility metric).
# ----------------------------------------------------------------------------
def graph_to_pandana_dfs(graph, time_attr):
    """Node / edge tables for pandana. Node IDs forced to int64 (pandana's
    C++ backend rejects strings / floats). MultiDiGraph parallels collapse
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


def run_pandana(graph, time_attr, cells, node_col, t_metric):
    """Pandana baseline: contraction-hierarchy precompute + aggregate."""
    node_x, node_y, edges_df = graph_to_pandana_dfs(graph, time_attr)
    t0 = time.perf_counter()
    net = pandana.Network(
        node_x=node_x, node_y=node_y,
        edge_from=edges_df['from'], edge_to=edges_df['to'],
        edge_weights=edges_df[[time_attr]],
        twoway=False,    # graphs are already directed
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
    origin."""
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
    costs = routing.tiered_path_costs(pairs, graph, weight=time_attr, cutoff=t_metric)
    w_vals = od_pairs.dest_values(WEIGHT_COL, pairs, nodes_gdf,
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

    - `orig_cells=None` → variant B (every cell is an origin).
    - `orig_cells=<bool array aligned to cells.index>` → variant C
      (e.g. AOI-restricted).
    """
    t0 = time.perf_counter()
    pairs = od_pairs.get_pairs(
        cells, r_cells=r_cells_m, node_column=node_col,
        zones=zones, r_zones=r_zones_m, r_medium=r_medium_m,
        orig_cells=orig_cells,
    )
    costs = routing.tiered_path_costs(pairs, graph, weight=time_attr, cutoff=t_metric)
    pairs_geo, costs_geo = od_pairs.reindex_by_geo_unit(
        pairs, costs, cells,
        cell_node_column=node_col, zones=zones, zone_node_column=node_col,
    )
    w_geo = od_pairs.dest_values_geo(WEIGHT_COL, pairs_geo, cells, zones=zones)
    accessibility.cumulative_opportunities(
        costs_geo, {'w': w_geo}, cells['zone_id'].to_dict(),
        [accessibility.Bin('in_T', 0, t_metric)],
    )
    return time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Per-mode driver
# ----------------------------------------------------------------------------
def bench_one_mode(mode, cfg, cells_full, zones_full, aoi_polygon):
    """Run pandana + variants A/B/C for one mode. Returns timings dict."""
    r_medium_label = (f"{cfg['r_medium_m']/1000:.0f} km" if cfg['r_medium_m']
                      else f"auto≈{min(cfg['r_cells_m']*10, cfg['r_zones_m'])/1000:.0f} km")
    print(f"\n{'='*70}\n{mode.upper()}  "
          f"(cumulative employment within {cfg['metric_t_s']//60} min; "
          f"r_cells={cfg['r_cells_m']/1000:.0f} km, r_medium={r_medium_label}, "
          f"r_zones={cfg['r_zones_m']/1000:.0f} km)\n{'='*70}")

    graph = network_processing.load_consolidated_graphml(PREPARED_DIR / cfg['graph_file'])
    bake_edge_times(graph, mode, cfg['speed_kph'])
    time_attr = f'{mode}_time_s'
    node_col = f'node_id_{mode}'

    if TEST_MODE:
        cells, zones, graph, aoi = clip_to_test_bbox(
            cells_full, zones_full, graph, aoi_polygon, TEST_BBOX_HALF_KM)
        print(f"  TEST subset: {len(cells):,} cells, {len(zones):,} zones, "
              f"{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    else:
        cells, zones, aoi = cells_full.copy(), zones_full.copy(), aoi_polygon
        print(f"  Graph: {graph.number_of_nodes():,} nodes, "
              f"{graph.number_of_edges():,} edges")
    snap_cells(cells, graph, node_col)
    snap_cells(zones, graph, node_col)

    in_aoi = cells.geometry.centroid.within(aoi).to_numpy()
    n_aoi = int(in_aoi.sum())
    print(f"  Origin universe — graph nodes: {graph.number_of_nodes():,} · "
          f"cells: {len(cells):,} ({cells[node_col].nunique():,} unique snap-nodes) · "
          f"AOI cells: {n_aoi:,} ({100*in_aoi.mean():.1f}%)\n")

    timings = {}
    print(f"  Pandana                  ...", end=' ', flush=True)
    timings['pandana'] = run_pandana(graph, time_attr, cells, node_col, cfg['metric_t_s'])
    print(f"→ {timings['pandana']:6.1f} s")

    print(f"  [A] Aperta all-nodes      ...", end=' ', flush=True)
    timings['A'] = run_aperta_all_nodes(graph, time_attr, cells, node_col,
                                        cfg['r_zones_m'], cfg['metric_t_s'])
    print(f"→ {timings['A']:6.1f} s  ({graph.number_of_nodes():,} origins)")

    print(f"  [B] Aperta cell-snap      ...", end=' ', flush=True)
    timings['B'] = run_aperta_tiered(graph, time_attr, cells, zones, node_col,
                                     cfg['r_cells_m'], cfg['r_medium_m'],
                                     cfg['r_zones_m'], cfg['metric_t_s'])
    print(f"→ {timings['B']:6.1f} s  ({cells[node_col].nunique():,} origins)")

    print(f"  [C] Aperta AOI cell-snap  ...", end=' ', flush=True)
    timings['C'] = run_aperta_tiered(graph, time_attr, cells, zones, node_col,
                                     cfg['r_cells_m'], cfg['r_medium_m'],
                                     cfg['r_zones_m'], cfg['metric_t_s'],
                                     orig_cells=in_aoi)
    print(f"→ {timings['C']:6.1f} s  ({n_aoi:,} origins)")

    return timings


# ----------------------------------------------------------------------------
# Main + summary
# ----------------------------------------------------------------------------
SUMMARY_ROWS = [
    ('Pandana — all graph nodes',                                   'pandana'),
    ('A. Aperta all nodes (single-tier, Euclidean cutoff)',         'A'),
    ('B. Aperta cell-snap origins, tiered destinations',            'B'),
    ('C. Aperta AOI cell-snap origins, tiered destinations',        'C'),
]


def print_summary(results):
    """Final pivot table — direct copy-paste source for the README."""
    print(f"\n{'='*70}\nSUMMARY (wall-clock seconds, lower = better)\n{'='*70}")
    headers = [f"{m} ({MODES[m]['metric_t_s']//60} min)" for m in results]
    label_w = max(len(r[0]) for r in SUMMARY_ROWS) + 2
    col_w = max(max(len(h) for h in headers), 10) + 2
    print(' ' * label_w + ''.join(f"{h:>{col_w}}" for h in headers))
    for label, key in SUMMARY_ROWS:
        cells_str = ''.join(f"{results[m][key]:>{col_w-2}.1f}s "
                            for m in results)
        print(f"{label:{label_w}}{cells_str}")


def main():
    cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
    zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
    aoi_polygon = gpd.read_file(PREPARED_DIR / 'aoi_polygon.gpkg').geometry.iloc[0]
    print(f"Cells (full): {len(cells):,}   Zones (full): {len(zones):,}")
    if TEST_MODE:
        print(f"TEST_MODE — subsetting to ±{TEST_BBOX_HALF_KM} km around AOI centroid.")

    results = {mode: bench_one_mode(mode, cfg, cells, zones, aoi_polygon)
               for mode, cfg in MODES.items()}
    print_summary(results)


if __name__ == '__main__':
    main()
