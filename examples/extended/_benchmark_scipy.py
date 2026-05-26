"""
Quick scipy.sparse.csgraph.dijkstra benchmark for the same Bern + 25 km
walking / car graphs as `_benchmark.py`. Probes whether scipy's
cutoff-aware Dijkstra would be a worthwhile backend swap if/when aperta
adds a `cutoff` parameter to `tiered_path_costs`.

Setup:
  - Same prepared graphs (walk + car).
  - Origins: the unique cell-snap nodes per mode (same set aperta-tiered
    routes from in the main benchmark — "origins inside the AOI").
  - Edge weight: `length` (metres). Cutoff matches the main benchmark's
    radius mask — 2 000 m walk, 50 000 m car — interpreted here as a
    NETWORK-distance cutoff (slightly more generous than Euclidean since
    paths bend, but the right order of magnitude for the comparison).

Two scipy variants timed:
  1. Per-origin loop — one `dijkstra(csr, indices=[orig], limit=T)` call
     per origin. Comparable to how aperta's `tiered_path_costs` iterates.
  2. Batch — one `dijkstra(csr, indices=all_origins, limit=T)` call.
     Tests whether scipy's batched-indices interface gives a meaningful
     speedup over the loop (best knowledge: ~1–10 %, single-threaded).

No accessibility / per-edge aggregation here — just routing time, to
isolate scipy's per-origin Dijkstra cost.

Run from `aperta/examples/extended/`. Don't run while `_benchmark.py` is
still running — both compete for the same cores.
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra

from aperta import network_processing


PREPARED_DIR = Path('data/prepared')

MODES = {
    'walk': dict(
        graph_file='walk_graph.graphml',
        cutoff_m=2_000,
    ),
    'car': dict(
        graph_file='car_graph.graphml',
        cutoff_m=50_000,
    ),
}


def graph_to_csr(graph):
    """Build a scipy CSR matrix from a (Multi)DiGraph using 'length' (m) as
    weight. Collapses MultiDiGraph parallels to one edge per (u, v) via the
    minimum-length parallel (matches the routing primitive's choice).
    Returns (csr_matrix, nx_to_seq) — seq IDs are 0..N-1 row indices.
    """
    nx_to_seq = {n: i for i, n in enumerate(graph.nodes())}
    n = len(nx_to_seq)
    min_len = {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        length = float(data['length'])
        key = (nx_to_seq[u], nx_to_seq[v])
        if key not in min_len or min_len[key] > length:
            min_len[key] = length
    rows = np.fromiter((u for u, _v in min_len.keys()), dtype=np.int64,
                       count=len(min_len))
    cols = np.fromiter((v for _u, v in min_len.keys()), dtype=np.int64,
                       count=len(min_len))
    data = np.fromiter(min_len.values(), dtype=float, count=len(min_len))
    csr = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n),
                                   dtype=float)
    return csr, nx_to_seq


def snap_cells(cells, graph, node_col):
    """Add `node_col` column to cells via centroid → nearest network node."""
    centroids = gpd.GeoDataFrame(
        geometry=cells.geometry.centroid, index=cells.index, crs=cells.crs,
    )
    cells[node_col], _ = network_processing.snap_to_network_nodes(centroids, graph)


def main():
    cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
    print(f"Cells: {len(cells):,}\n")

    for mode, cfg in MODES.items():
        cutoff_m = cfg['cutoff_m']
        print(f"{'='*70}\n{mode.upper()}  (cutoff = {cutoff_m/1000:.0f} km "
              f"network distance, edge weight = length in m)\n{'='*70}")
        graph = network_processing.load_consolidated_graphml(
            PREPARED_DIR / cfg['graph_file'])
        print(f"  Graph: {graph.number_of_nodes():,} nodes, "
              f"{graph.number_of_edges():,} edges")

        node_col = f'node_id_{mode}'
        snap_cells(cells, graph, node_col)
        unique_origins_nx = cells[node_col].dropna().unique()
        print(f"  Unique snap-node origins (= aperta-tiered origin set): "
              f"{len(unique_origins_nx):,}")

        t0 = time.time()
        csr, nx_to_seq = graph_to_csr(graph)
        t_csr = time.time() - t0
        print(f"  CSR build:                  {t_csr:6.1f} s")

        # Map origins to seq IDs, dropping any that aren't in the graph
        # (rare but possible after subset / edge-cleanup steps).
        seq_origins = np.array(
            [nx_to_seq[n] for n in unique_origins_nx if n in nx_to_seq],
            dtype=np.int64,
        )

        # Variant A: per-origin loop.
        t0 = time.time()
        for orig in seq_origins:
            _ = dijkstra(csr, indices=[orig], limit=cutoff_m,
                         return_predecessors=False)
        t_loop = time.time() - t0
        ms_per = 1000 * t_loop / max(1, len(seq_origins))
        print(f"  scipy per-origin loop:      {t_loop:6.1f} s  "
              f"({ms_per:.2f} ms/origin)")

        # Variant B: batch call (all origins at once).
        t0 = time.time()
        _ = dijkstra(csr, indices=seq_origins, limit=cutoff_m,
                     return_predecessors=False)
        t_batch = time.time() - t0
        ms_per_b = 1000 * t_batch / max(1, len(seq_origins))
        print(f"  scipy batched indices:      {t_batch:6.1f} s  "
              f"({ms_per_b:.2f} ms/origin)")

        print(f"  → batch speedup vs loop:    "
              f"{t_loop / t_batch:.2f}×\n")


if __name__ == '__main__':
    main()
