"""
Quick aperta vs pandana benchmark on the Bern + 25 km consolidated graphs.

Cumulative-opportunity to total employment, per mode. The story this version
tells: how does aperta's runtime scale with the **origin set size**? The
destination structure (3-tier `cells_to_cells + cells_to_zones +
zones_to_zones`) and the routing cutoff (`r_zones_m`) are held constant
across the three aperta variants — only the origin set differs:

  A. **All graph nodes** as origins (pandana-comparable; tiered structure
     can't help when every node is already an origin, so this collapses to
     a single-tier Euclidean-cutoff routing).
  B. **Cell-snap origins** — every unique snap-node referenced by at least
     one cell. Standard tiered aperta usage; 3-tier destination structure.
  C. **AOI cell-snap origins** — variant B further restricted to cells
     whose centroid lies inside the AOI polygon (the typical production
     case: a buffer zone around the AOI provides destinations and through-
     routing but is *not* an origin). Same 3-tier dest structure.

Pandana (always all-nodes) is reported as a reference baseline for variant A.

Each tiered variant runs twice: once via igraph (no cutoff) and once via
scipy with `cutoff=metric_t_s` to show the cutoff speedup.

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
# Tier semantics:
#   - r_cells_m:  cells_to_cells outer radius (close, cell-cell)
#   - r_medium_m: cells_to_zones outer radius (medium, cell-zone) — `None`
#                 auto-infers as min(r_cells * 10, r_zones)
#   - r_zones_m:  zones_to_zones outer radius (far, zone-zone). This is the
#                 effective routing cutoff for variants A/B/C and pandana.
MODES = {
    'walk': dict(
        graph_file='walk_graph.graphml',
        speed_kph=5.0,            # length / (5 km/h) → seconds; matches prep
        r_cells_m=1_000,
        r_medium_m=None,          # auto: min(1000*10, 2000) = 2000
        r_zones_m=2_000,
        metric_t_s=15 * 60,       # cumulative within 15 minutes
    ),
    'car': dict(
        graph_file='car_graph.graphml',
        # car uses per-edge OSM speed_kph; speed_kph below is the fallback
        speed_kph=None,           # use per-edge speed_kph instead
        r_cells_m=1_000,
        r_medium_m=10_000,        # preserve cell-origin precision to 10 km
        r_zones_m=50_000,
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
    """All graph nodes routed via pandana CH. Returns (acc, construct_s,
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
    snap_ids = cells[node_col].astype('int64')
    per_node = cells.assign(_snap=snap_ids).groupby('_snap')[weight_col].sum()
    per_node = per_node.reindex(node_x.index, fill_value=0.0)
    net.set(per_node.index.to_series(), variable=per_node.values, name='w')
    acc = net.aggregate(t_metric, type='sum', name='w', decay='flat')
    t_set_aggregate = time.time() - t0
    return acc, t_construct, t_precompute, t_set_aggregate


def run_variant_a_all_nodes(graph, time_attr, cells, node_col, weight_col,
                            r_outer_m, t_metric, cutoff_s=None):
    """Variant A: every graph node is its own 'cell', single-tier — the
    tiered destination structure can't help when every node is already an
    origin. With `cutoff_s` set, routing uses scipy; otherwise igraph.

    Returns `(acc, total_s, n_origins)`.
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
        nodes_gdf, r_cells=r_outer_m, node_column='node_id_synth',
    )
    times = routing.tiered_path_costs(pairs, graph, weight=time_attr,
                                      cutoff=cutoff_s)
    w_vals = od_pairs.dest_values(
        weight_col, pairs, nodes_gdf, node_column='node_id_synth',
    )
    acc = accessibility.count_in_bins(
        times, {'w': w_vals}, {},
        [accessibility.Bin('in_T', 0, t_metric)],
    )
    return acc, time.time() - t0, len(node_ids)


def run_variant_tiered(graph, time_attr, cells, zones, node_col,
                       weight_col, r_cells_m, r_medium_m, r_zones_m,
                       t_metric, *, orig_cells=None, cutoff_s=None):
    """Variants B and C: 3-tier destination structure (cells_to_cells +
    cells_to_zones + zones_to_zones), with optional `orig_cells` mask.

    - `orig_cells=None` → variant B (every cell is an origin).
    - `orig_cells=<bool array aligned to cells.index>` → variant C (e.g.
      AOI-restricted origins).

    With `cutoff_s` set, routing uses scipy; otherwise igraph (no cutoff).
    Returns `(acc, total_s, n_origins)`.
    """
    t0 = time.time()
    pairs = od_pairs.get_pairs(
        cells, r_cells=r_cells_m, node_column=node_col,
        zones=zones, r_zones=r_zones_m, r_medium=r_medium_m,
        orig_cells=orig_cells,
    )
    times = routing.tiered_path_costs(pairs, graph, weight=time_attr,
                                      cutoff=cutoff_s)
    pairs_geo, times_geo = od_pairs.reindex_by_geo_unit(
        pairs, times, cells,
        cell_node_column=node_col,
        zones=zones, zone_node_column=node_col,
    )
    w_geo = od_pairs.dest_values_geo(
        weight_col, pairs_geo, cells, zones=zones)
    cell_to_zone = cells['zone_id'].to_dict()
    acc = accessibility.count_in_bins(
        times_geo, {'w': w_geo}, cell_to_zone,
        [accessibility.Bin('in_T', 0, t_metric)],
    )
    n_origins = len(pairs.cells_to_cells)
    return acc, time.time() - t0, n_origins


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def clip_to_test_bbox(cells, zones, graph, aoi_polygon, half_km):
    """Clip cells, zones, graph, and AOI to a square box around the cells'
    centroid. Returns (cells_sub, zones_sub, graph_sub, aoi_sub)."""
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
    # Intersect AOI with the bbox so the "in-AOI" fraction is meaningful.
    from shapely.geometry import box
    aoi_sub = aoi_polygon.intersection(box(minx, miny, maxx, maxy))
    return cells_sub, zones_sub, graph_sub, aoi_sub


def main():
    cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
    zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
    aoi_polygon = gpd.read_file(PREPARED_DIR / 'aoi_polygon.gpkg').geometry.iloc[0]
    print(f"Cells (full): {len(cells):,}   Zones (full): {len(zones):,}")
    if TEST_MODE:
        print(f"TEST_MODE — subsetting to ±{TEST_BBOX_HALF_KM} km around AOI centroid.")

    for mode, cfg in MODES.items():
        r_medium_m = cfg.get('r_medium_m')
        medium_label = (f"{r_medium_m/1000:.0f} km" if r_medium_m
                        else f"auto≈{min(cfg['r_cells_m']*10, cfg['r_zones_m'])/1000:.0f} km")
        print(f"\n{'='*70}\n{mode.upper()}  "
              f"(cumulative employment within {cfg['metric_t_s']//60} min; "
              f"r_cells={cfg['r_cells_m']/1000:.0f} km, "
              f"r_medium={medium_label}, "
              f"r_zones={cfg['r_zones_m']/1000:.0f} km)\n{'='*70}")
        graph = network_processing.load_consolidated_graphml(
            PREPARED_DIR / cfg['graph_file'])
        bake_edge_times(graph, mode, cfg['speed_kph'])
        attr = time_attr_for(mode)
        node_col = f'node_id_{mode}'

        if TEST_MODE:
            mode_cells, mode_zones, graph, mode_aoi = clip_to_test_bbox(
                cells, zones, graph, aoi_polygon, TEST_BBOX_HALF_KM)
            print(f"  Subset: {len(mode_cells):,} cells, {len(mode_zones):,} zones, "
                  f"{graph.number_of_nodes():,} graph nodes, "
                  f"{graph.number_of_edges():,} edges")
        else:
            mode_cells, mode_zones, mode_aoi = cells, zones, aoi_polygon
            print(f"  Graph: {graph.number_of_nodes():,} nodes, "
                  f"{graph.number_of_edges():,} edges")
        snap_cells(mode_cells, graph, node_col)
        snap_cells(mode_zones, graph, node_col)

        # AOI mask: cell centroids inside the AOI polygon.
        in_aoi = mode_cells.geometry.centroid.within(mode_aoi).to_numpy()
        n_aoi = int(in_aoi.sum())
        print(f"  Origin universe — graph nodes: {graph.number_of_nodes():,} · "
              f"cells: {len(mode_cells):,} ({mode_cells[node_col].nunique():,} unique snap-nodes) · "
              f"AOI cells: {n_aoi:,} ({100*in_aoi.mean():.1f}%)")

        # --- Pandana (reference, all graph nodes) -------------------------
        print(f"\n  Pandana (all {graph.number_of_nodes():,} graph nodes) ...    ",
              end='', flush=True)
        _, t_c, t_p, t_a = run_pandana(graph, attr, mode_cells, node_col,
                                       'employment_total', cfg['metric_t_s'])
        print(f"construct {t_c:5.1f}s  precompute {t_p:6.1f}s  "
              f"set+aggregate {t_a:5.1f}s  → total {t_c+t_p+t_a:6.1f}s")

        # --- Variant A: all graph nodes (single tier) ---------------------
        for backend, kwargs in [('igraph', {}),
                                ('scipy', {'cutoff_s': cfg['metric_t_s']})]:
            print(f"  [A] Aperta all-nodes ({backend}) ...", end=' ', flush=True)
            _, t, n_origins = run_variant_a_all_nodes(
                graph, attr, mode_cells, node_col, 'employment_total',
                cfg['r_zones_m'], cfg['metric_t_s'], **kwargs)
            print(f"({n_origins:,} origins)  → {t:6.1f} s")

        # --- Variant B: all cell-snap origins (3-tier) --------------------
        for backend, kwargs in [('igraph', {}),
                                ('scipy', {'cutoff_s': cfg['metric_t_s']})]:
            print(f"  [B] Aperta cell-snap ({backend}) ...", end=' ', flush=True)
            _, t, n_origins = run_variant_tiered(
                graph, attr, mode_cells, mode_zones, node_col,
                'employment_total',
                cfg['r_cells_m'], r_medium_m, cfg['r_zones_m'],
                cfg['metric_t_s'], **kwargs)
            print(f"({n_origins:,} origins from {len(mode_cells):,} cells)  → {t:6.1f} s")

        # --- Variant C: AOI cell-snap origins (3-tier) --------------------
        for backend, kwargs in [('igraph', {}),
                                ('scipy', {'cutoff_s': cfg['metric_t_s']})]:
            print(f"  [C] Aperta AOI cell-snap ({backend}) ...", end=' ', flush=True)
            _, t, n_origins = run_variant_tiered(
                graph, attr, mode_cells, mode_zones, node_col,
                'employment_total',
                cfg['r_cells_m'], r_medium_m, cfg['r_zones_m'],
                cfg['metric_t_s'], orig_cells=in_aoi, **kwargs)
            print(f"({n_origins:,} origins from {n_aoi:,} AOI cells)  → {t:6.1f} s")


if __name__ == '__main__':
    main()
