"""
Verify the per-cell tier filter at `reindex_by_geo_unit` eliminates the
multi-zone-snap-node double-counting we measured on the Bern bike data.

Before fix (no radii passed): island cell's c2c parent zones AND c2z zones
overlap on 10 zones, totalling ~26k double-counted raw employment.

After fix (radii passed): overlap should be 0 for every cell.

Run from `aperta/examples/extended/`:

    python verify_filter_fix.py
"""
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np

from aperta import od_pairs


PREPARED_DIR = Path('data/prepared')
TIER_CUTOFFS_BIKE = dict(r_cells=1_000.0, r_medium=5_000.0, r_zones=25_000.0)


def summarize(out_pairs, cells, label):
    cell_zone = cells['zone_id'].to_dict()
    target_cells = ['8a1f8342c16ffff', '8a1f8342cb97fff', '8a1f8342cb9ffff']
    print(f"\n--- {label} ---")
    print(f"{'cell_id':<18s} {'c2c zones':>10s} {'c2z zones':>10s} {'overlap':>8s}")
    print("-" * 60)
    overlap_count_all = 0
    overlap_cells = 0
    for cid in cells.index[:5000]:   # sample first 5000 cells to get global stat
        if cid not in out_pairs.cells_to_cells:
            continue
        c2c_zones = {cell_zone[c] for c in out_pairs.cells_to_cells.get(cid, [])
                     if c in cell_zone}
        c2z_zones = set(out_pairs.cells_to_zones.get(cid, []))
        overlap = c2c_zones & c2z_zones
        if overlap:
            overlap_cells += 1
            overlap_count_all += len(overlap)
    print(f"  (sample 5000 cells: {overlap_cells} with non-empty overlap, "
          f"{overlap_count_all} total overlapping (cell, zone) pairs)\n")
    for cid in target_cells:
        if cid not in out_pairs.cells_to_cells:
            continue
        c2c_zones = {cell_zone[c] for c in out_pairs.cells_to_cells.get(cid, [])
                     if c in cell_zone}
        c2z_zones = set(out_pairs.cells_to_zones.get(cid, []))
        overlap = c2c_zones & c2z_zones
        print(f"{cid:<18s} {len(c2c_zones):>10d} {len(c2z_zones):>10d} {len(overlap):>8d}")


def main():
    cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
    zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
    cells = cells.rename(columns={'node_id_bike': 'node_id'})
    zones = zones.rename(columns={'node_id_bike': 'node_id'})

    print("Building tiered OD pairs (bike)...")
    pairs = od_pairs.get_pairs(
        cells, node_column='node_id', zones=zones, **TIER_CUTOFFS_BIKE,
    )

    print("\nReindexing UNFILTERED (legacy / buggy)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unfiltered, _ = od_pairs.reindex_by_geo_unit(
            pairs, None, cells,
            cell_node_column='node_id', zones=zones, zone_node_column='node_id',
        )
    summarize(unfiltered, cells, "UNFILTERED (legacy)")

    print("\nReindexing FILTERED (with per-cell tier filter)...")
    filtered, _ = od_pairs.reindex_by_geo_unit(
        pairs, None, cells,
        cell_node_column='node_id', zones=zones, zone_node_column='node_id',
        **TIER_CUTOFFS_BIKE,
    )
    summarize(filtered, cells, "FILTERED (per-cell tier filter)")


if __name__ == '__main__':
    main()
