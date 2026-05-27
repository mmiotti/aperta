"""
Quick aperta vs pandana benchmark on the Bern + 25 km consolidated graphs.

Cumulative-opportunity to total employment, per mode, three routing variants:

  1. Pandana — native, routes from every graph node. Reported in three
     phases: Network construction, precompute (the bulk routing), and
     set+aggregate (destination weights + metric).
  2. Aperta tiered (cells as origins) — standard aperta usage; cells_to_cells
     plus zones_to_zones for the far tier. `od_pairs.get_pairs(r_cells=...,
     r_zones=...)` does the Euclidean mask before routing — each origin
     only sees destinations within that radius, which is what makes the
     "sparse origins" case fast.
  3. Aperta all-nodes — every graph node is its own "cell"; no zones,
     same Euclidean radius mask. Apples-to-apples vs pandana on raw
     per-origin throughput.

Toggle `TEST_MODE = True` for a fast end-to-end smoke test on a small
bbox subset (~30 s); set `TEST_MODE = False` for the full Bern + 25 km
run (probably 30 min+).

Run from `aperta/examples/extended/`. Prep outputs assumed available
under `data/prepared/`.
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pandana
from shapely.geometry import Point

from aperta import accessibility, network_processing, od_pairs, routing


PREPARED_DIR = Path('data/prepared')

TEST_MODE = False              # True → small bbox subset; False → full Bern
TEST_BBOX_HALF_KM = 3.0        # half-size of the test bbox around AOI centroid

# Per-mode settings (Euclidean cutoff for OD construction; metric threshold).
MODES = {
    'walk': dict(
        graph_file='walk_graph.graphml',
        speed_kph=5.0,            # length / (5 km/h) → seconds; matches prep
        r_cells_m=1_000,          # aperta tiered cell tier
        r_zones_m=2_000,          # aperta tiered zone tier (outer radius)
        radius_m=2_000,           # aperta-all-nodes + pandana cutoff (outer)
        metric_t_s=15 * 60,       # cumulative within 15 minutes
    ),
    'car': dict(
        graph_file='car_graph.graphml',
        # car uses per-edge OSM speed_kph; speed_kph below is the fallback
        speed_kph=None,           # use per-edge speed_kph instead
        r_cells_m=1_000,
        r_zones_m=10_000,
        r_regions_m=50_000,
        radius_m=50_000,
        metric_t_s=30 * 60,
    ),
}


def time_attr_for(mode):
    return f'{mode}_time_s'


def bake_edge_times(graph, mode, fallback_kph):
    """Write per-edge travel time (seconds) into `<mode>_time_s` attribute."""
    attr = time_attr_for(mode)
    for u, v, k, data in graph.edges(keys=True, data=True):
        length = float(data['length'])
        if fallback_kph is None:
            speed_kph = float(data.get('speed_kph', 30.0))
        else:
            speed_kph = fallback_kph
        data[attr] = length / (speed_kph * 1000.0 / 3600.0)


def snap_cells(cells, graph, node_col):
    """Add `node_col` column to cells via centroid → nearest network node."""
    centroids = gpd.GeoDataFrame(
        geometry=cells.geometry.centroid, index=cells.index, crs=cells.crs,
    )
    cells[node_col], _ = network_processing.snap_to_network_nodes(centroids, graph)


def graph_to_pandana_dfs(graph, time_attr):
    """Node and edge structures for pandana. Node IDs are forced to int64
    (pandana's C++ backend rejects strings / floats). MultiDiGraph parallels
    collapse to one edge per (u, v) via the minimum-weight parallel —
    matches aperta's routing choice."""
    node_ids = [int(n) for n in graph.nodes()]
    node_index = pd.Index(node_ids, dtype='int64', name='node_id')
    node_x = pd.Series([graph.nodes[n]['x'] for n in graph.nodes()],
                       index=node_index, dtype=float)
    node_y = pd.Series([graph.nodes[n]['y'] for n in graph.nodes()],
                       index=node_index, dtype=float)
    min_weight = {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        t = float(data[time_attr])
        key = (int(u), int(v))
        if key not in min_weight or min_weight[key] > t:
            min_weight[key] = t
    edges_df = pd.DataFrame({
        'from': np.array([u for u, _v in min_weight.keys()], dtype='int64'),
        'to':   np.array([v for _u, v in min_weight.keys()], dtype='int64'),
        time_attr: np.array(list(min_weight.values()), dtype=float),
    })
    return node_x, node_y, edges_df


def per_node_weight(cells, node_col, weight_col, all_node_index):
    """Sum cell weights per snapped node; reindex to the full node list."""
    s = cells.groupby(node_col)[weight_col].sum()
    return s.reindex(all_node_index, fill_value=0.0)


# ----------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------

def run_pandana(graph, time_attr, cells, node_col, weight_col, t_metric):
    """All nodes routed; precompute + aggregate. Returns (acc, construct_s,
    precompute_s, set_aggregate_s)."""
    node_x, node_y, edges_df = graph_to_pandana_dfs(graph, time_attr)
    t0 = time.time()
    net = pandana.Network(
        node_x=node_x, node_y=node_y,
        edge_from=edges_df['from'], edge_to=edges_df['to'],
        edge_weights=edges_df[[time_attr]],
        twoway=False,    # graphs are already directed
    )
    t_construct = time.time() - t0
    t0 = time.time()
    net.precompute(t_metric)
    t_precompute = time.time() - t0
    t0 = time.time()
    # Snap-node IDs from `cells` are strings (graphml-loaded). Convert to
    # int64 to match `node_x.index` so the per-node weight reindex hits.
    snap_ids = cells[node_col].astype('int64')
    per_node = cells.assign(_snap=snap_ids).groupby('_snap')[weight_col].sum()
    per_node = per_node.reindex(node_x.index, fill_value=0.0)
    net.set(per_node.index.to_series(), variable=per_node.values, name='w')
    acc = net.aggregate(t_metric, type='sum', name='w', decay='flat')
    t_set_aggregate = time.time() - t0
    return acc, t_construct, t_precompute, t_set_aggregate


def run_aperta_tiered(graph, time_attr, cells, zones, regions, node_col,
                      weight_col, r_cells_m, r_zones_m, r_regions_m, t_metric,
                      cutoff_s=None):
    """Cells (+ optional zones + optional regions) as tiered origins / dests.

    Pass `regions=None, r_regions_m=None` for the 2-tier walk case; pass all
    three for the 3-tier car case. With `cutoff_s` set, routing uses scipy
    (cutoff in weight units = seconds for time-weighted edges); otherwise
    igraph (no cutoff). Returns `(acc, total_s, n_origins)`.
    """
    t0 = time.time()
    pairs = od_pairs.get_pairs(
        cells, r_cells=r_cells_m, node_column=node_col,
        zones=zones, r_zones=r_zones_m,
        regions=regions, r_regions=r_regions_m,
    )
    times = routing.tiered_path_costs(pairs, graph, weight=time_attr,
                                       cutoff=cutoff_s)
    pairs_geo, times_geo = od_pairs.reindex_by_geo_unit(
        pairs, times, cells,
        cell_node_column=node_col,
        zones=zones, zone_node_column=node_col,
        regions=regions, region_node_column=node_col,
    )
    w_geo = od_pairs.dest_values_geo(
        weight_col, pairs_geo, cells, zones=zones, regions=regions)
    cell_to_zone = cells['zone_id'].to_dict()
    acc = accessibility.count_in_bins(
        times_geo, {'w': w_geo}, cell_to_zone,
        [accessibility.Bin('in_T', 0, t_metric)],
    )
    n_origins = len(pairs.cells_to_cells)
    return acc, time.time() - t0, n_origins


def run_aperta_all_nodes(graph, time_attr, cells, node_col, weight_col,
                         r_cells_m, t_metric):
    """Every graph node = origin = its own 'cell'; single tier.

    Returns `(acc, total_s, n_origins)` where n_origins = graph node count.
    """
    t0 = time.time()
    node_ids = list(graph.nodes())
    weights = per_node_weight(cells, node_col, weight_col, node_ids)
    nodes_gdf = gpd.GeoDataFrame(
        {'node_id_synth': node_ids,
         weight_col: weights.values},
        index=pd.Index(node_ids, name='synth_cell_id'),
        geometry=[Point(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in node_ids],
        crs=cells.crs,
    )
    pairs = od_pairs.get_pairs(
        nodes_gdf, r_cells=r_cells_m, node_column='node_id_synth',
    )
    times = routing.tiered_path_costs(pairs, graph, weight=time_attr)
    w_vals = od_pairs.dest_values(
        weight_col, pairs, nodes_gdf, node_column='node_id_synth',
    )
    acc = accessibility.count_in_bins(
        times, {'w': w_vals}, {},
        [accessibility.Bin('in_T', 0, t_metric)],
    )
    return acc, time.time() - t0, len(node_ids)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def clip_to_test_bbox(cells, zones, regions, graph, half_km):
    """Clip cells, zones, regions, and graph to a square box around the
    cells' centroid. Returns (cells_sub, zones_sub, regions_sub, graph_sub).
    `regions` may be None (returns None back)."""
    centre = cells.union_all().centroid
    half_m = half_km * 1000
    minx, miny = centre.x - half_m, centre.y - half_m
    maxx, maxy = centre.x + half_m, centre.y + half_m
    cells_sub = cells.cx[minx:maxx, miny:maxy].copy()
    zones_sub = zones.cx[minx:maxx, miny:maxy].copy()
    regions_sub = (regions.cx[minx:maxx, miny:maxy].copy()
                   if regions is not None else None)
    # Drop cells/zones referencing a parent not in the subset (keeps tier
    # stitching consistent).
    cells_sub = cells_sub[cells_sub['zone_id'].isin(zones_sub.index)]
    if regions_sub is not None and 'region_id' in zones_sub.columns:
        zones_sub = zones_sub[zones_sub['region_id'].isin(regions_sub.index)]
        cells_sub = cells_sub[cells_sub['zone_id'].isin(zones_sub.index)]
    # Subset graph to nodes inside the bbox.
    keep_nodes = [n for n, d in graph.nodes(data=True)
                  if minx <= d['x'] <= maxx and miny <= d['y'] <= maxy]
    graph_sub = graph.subgraph(keep_nodes).copy()
    return cells_sub, zones_sub, regions_sub, graph_sub


def main():
    cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
    zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
    regions = gpd.read_file(PREPARED_DIR / 'regions.gpkg').set_index('region_id')
    # Derive zone→region from cells (H3 hierarchy is deterministic).
    if 'region_id' not in zones.columns:
        zones['region_id'] = cells.groupby('zone_id')['region_id'].first()
    print(f"Cells (full): {len(cells):,}   Zones (full): {len(zones):,}   "
          f"Regions (full): {len(regions):,}")
    if TEST_MODE:
        print(f"TEST_MODE — subsetting to ±{TEST_BBOX_HALF_KM} km around AOI centroid.")

    for mode, cfg in MODES.items():
        r_regions_m = cfg.get('r_regions_m')
        n_tiers = 3 if r_regions_m else 2
        print(f"\n{'='*70}\n{mode.upper()}  "
              f"(cumulative employment within {cfg['metric_t_s']//60} min; "
              f"{n_tiers}-tier: r_cells={cfg['r_cells_m']/1000:.0f} km"
              + (f", r_zones={cfg['r_zones_m']/1000:.0f} km" if cfg.get('r_zones_m') else "")
              + (f", r_regions={r_regions_m/1000:.0f} km" if r_regions_m else "")
              + f"; all-nodes/pandana r={cfg['radius_m']/1000:.0f} km)\n{'='*70}")
        graph = network_processing.load_consolidated_graphml(
            PREPARED_DIR / cfg['graph_file'])
        bake_edge_times(graph, mode, cfg['speed_kph'])
        attr = time_attr_for(mode)
        node_col = f'node_id_{mode}'

        mode_regions = regions if r_regions_m else None
        if TEST_MODE:
            mode_cells, mode_zones, mode_regions, graph = clip_to_test_bbox(
                cells, zones, mode_regions, graph, TEST_BBOX_HALF_KM)
            r_extra = f", {len(mode_regions):,} regions" if mode_regions is not None else ""
            print(f"  Subset: {len(mode_cells):,} cells, {len(mode_zones):,} zones"
                  f"{r_extra}, {graph.number_of_nodes():,} graph nodes, "
                  f"{graph.number_of_edges():,} edges")
        else:
            mode_cells, mode_zones = cells, zones
            print(f"  Graph: {graph.number_of_nodes():,} nodes, "
                  f"{graph.number_of_edges():,} edges")
        snap_cells(mode_cells, graph, node_col)
        snap_cells(mode_zones, graph, node_col)
        if mode_regions is not None:
            snap_cells(mode_regions, graph, node_col)

        n_all = graph.number_of_nodes()

        print(f"  Pandana ({n_all:,} nodes routed) ...    ", end='', flush=True)
        _, t_c, t_p, t_a = run_pandana(graph, attr, mode_cells, node_col,
                                       'employment_total', cfg['metric_t_s'])
        print(f"construct {t_c:5.1f}s  precompute {t_p:6.1f}s  "
              f"set+aggregate {t_a:5.1f}s  → total {t_c+t_p+t_a:6.1f}s")

        print(f"  Aperta tiered (igraph) ...", end=' ', flush=True)
        _, t, n_origins = run_aperta_tiered(
            graph, attr, mode_cells, mode_zones, mode_regions, node_col,
            'employment_total',
            cfg['r_cells_m'], cfg['r_zones_m'], r_regions_m,
            cfg['metric_t_s'])
        print(f"({n_origins:,} unique snap-node origins from {len(mode_cells):,} cells)  "
              f"→ {t:6.1f} s")

        print(f"  Aperta tiered (scipy, cutoff={cfg['metric_t_s']}s) ...",
              end=' ', flush=True)
        _, t, _ = run_aperta_tiered(
            graph, attr, mode_cells, mode_zones, mode_regions, node_col,
            'employment_total',
            cfg['r_cells_m'], cfg['r_zones_m'], r_regions_m,
            cfg['metric_t_s'], cutoff_s=cfg['metric_t_s'])
        print(f"→ {t:6.1f} s")

        print(f"  Aperta all-nodes ...", end=' ', flush=True)
        _, t, n_origins = run_aperta_all_nodes(
            graph, attr, mode_cells, node_col, 'employment_total',
            cfg['radius_m'], cfg['metric_t_s'])
        print(f"({n_origins:,} origins)  → {t:6.1f} s")


if __name__ == '__main__':
    main()
