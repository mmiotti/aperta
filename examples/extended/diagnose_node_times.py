"""
How much faster is the 'bright' snap-node than its neighbors?

Routes from each of the 5 snap-nodes around the island location to the
island's destination set (c2c + c2z medium-tier destinations on the
bike graph), and reports the per-destination time delta.

Run from `aperta/examples/extended/`:

    python diagnose_node_times.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from aperta import network_processing, od_pairs, routing


PREPARED_DIR = Path('data/prepared')
TIME_ATTR = 'bike_time_s'
CUTOFF_S = 60 * 60
TIER_CUTOFFS_BIKE = dict(r_cells=1_000.0, r_medium=5_000.0, r_zones=25_000.0)

ISLAND_XY = (2_601_690, 1_199_940)

# Source snap-nodes to compare, from the previous diagnostic.
# Label, node_id, which cells live on it.
SOURCES = [
    ('BRIGHT (island + twin)', 144987, 2),
    ('dim (2 cells)',          143618, 2),
    ('dim',                    148761, 1),
    ('dim',                     82859, 1),
    ('dim',                    163748, 1),
]


def bake_bike_time(g):
    BASE_KPH = 18.0
    ALPHA_UP = 3.1
    ALPHA_DOWN = -0.3
    BETA_INT4 = 7.0
    BETA_SIG = 1.0
    FLOOR_KPH = 1.0
    MAX_DOWNHILL_KPH = 50.0
    min_dur_per_m = 1.0 / (MAX_DOWNHILL_KPH / 3.6)
    for _, _, _, d in g.edges(keys=True, data=True):
        eff_kph = max(BASE_KPH, FLOOR_KPH)
        base_dur = float(d['length']) / (eff_kph / 3.6)
        slope = ALPHA_UP * float(d['elev_gain']) + ALPHA_DOWN * float(d['elev_loss'])
        inter = (BETA_INT4 * float(d['is_degree_4'])
                 + BETA_SIG * float(d['is_traffic_signal']))
        total = base_dur + slope + inter
        total = max(total, float(d['length']) * min_dur_per_m)
        d[TIME_ATTR] = total


def main():
    cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
    zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
    bike_graph = network_processing.load_consolidated_graphml(
        PREPARED_DIR / 'bike_graph.graphml')
    cells = cells.rename(columns={'node_id_bike': 'node_id'})
    zones = zones.rename(columns={'node_id_bike': 'node_id'})
    bake_bike_time(bike_graph)

    island_pt = Point(*ISLAND_XY)
    island_id = cells.geometry.centroid.distance(island_pt).idxmin()
    island_node = int(cells.loc[island_id, 'node_id'])
    print(f"Island cell {island_id} → snap node {island_node}")

    # Build OD pairs once, to get the island's destination set.
    pairs = od_pairs.get_pairs(
        cells, node_column='node_id', zones=zones, **TIER_CUTOFFS_BIKE,
    )
    c2c_dests = pairs.cells_to_cells.get(island_node, np.array([]))
    c2z_dests = pairs.cells_to_zones.get(island_node, np.array([]))
    dests = np.unique(np.concatenate([c2c_dests, c2z_dests]))
    print(f"Destination set: {len(c2c_dests):,} c2c + {len(c2z_dests):,} c2z "
          f"= {len(dests):,} unique nodes\n")

    # Route from each source to the same destination set.
    src_nodes = [s[1] for s in SOURCES]
    src_in_graph = [n for n in src_nodes if n in bike_graph.nodes]
    if len(src_in_graph) != len(src_nodes):
        missing = set(src_nodes) - set(src_in_graph)
        print(f"WARNING: sources missing from graph: {missing}")
    dist_matrix = routing.shortest_distances_pairwise(
        bike_graph, origins=src_in_graph, destinations=dests.tolist(),
        weight=TIME_ATTR, cutoff=CUTOFF_S,
    )
    # rows = sources, cols = dests, values = seconds (inf if unreachable)

    bright_idx = src_in_graph.index(144987)
    bright_times = dist_matrix[bright_idx]
    reach_bright = np.isfinite(bright_times)

    print(f"{'source node':<28s} {'reach':>7s}  "
          f"{'median Δs':>10s} {'mean Δs':>10s} "
          f"{'P10 Δs':>8s} {'P90 Δs':>8s}  "
          f"{'> bright':>8s} {'tied':>5s} {'< bright':>8s}")
    print("-" * 110)
    for (label, node, n_cells), times in zip(SOURCES, dist_matrix):
        reach = np.isfinite(times) & reach_bright
        delta = times[reach] - bright_times[reach]    # positive → this source slower than bright
        is_bright = node == 144987
        marker = ' ← BRIGHT' if is_bright else ''
        if delta.size == 0:
            print(f"{label:<20s} {node:>7d}  no overlap reachable")
            continue
        print(f"{label:<20s} {node:>7d}  {reach.sum():>7d}  "
              f"{np.median(delta):>10.1f} {delta.mean():>10.1f} "
              f"{np.percentile(delta, 10):>8.1f} {np.percentile(delta, 90):>8.1f}  "
              f"{(delta > 1).sum():>8d} {(np.abs(delta) <= 1).sum():>5d} "
              f"{(delta < -1).sum():>8d}{marker}")

    # Walk distance from each source to bright source (gives intuition for the
    # "spatial offset penalty" — these source nodes are <200 m apart).
    bright_node = 144987
    pos = {n: (bike_graph.nodes[n]['x'], bike_graph.nodes[n]['y']) for n in src_in_graph}
    print(f"\nStraight-line distance from BRIGHT node {bright_node}:")
    for n in src_in_graph:
        d = ((pos[n][0] - pos[bright_node][0])**2
             + (pos[n][1] - pos[bright_node][1])**2) ** 0.5
        print(f"  {n}: {d:.0f} m")

    # Travel time between sources via the bike graph (so we know how
    # 'expensive' the alternative entry node would be to reach by bike).
    src_to_src = routing.shortest_distances_pairwise(
        bike_graph, origins=src_in_graph, destinations=src_in_graph,
        weight=TIME_ATTR, cutoff=600,
    )
    print(f"\nBike-time between source nodes (s):")
    print(f"  {'':<10s} " + " ".join(f"{n:>8d}" for n in src_in_graph))
    for i, n in enumerate(src_in_graph):
        row = " ".join(
            f"{src_to_src[i, j]:>8.0f}" if np.isfinite(src_to_src[i, j]) else "     inf"
            for j in range(len(src_in_graph))
        )
        print(f"  {n:<10d} " + row)


if __name__ == '__main__':
    main()
