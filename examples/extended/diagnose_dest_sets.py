"""
Compare the actual destination sets (cells_to_cells + cells_to_zones)
between the bright island cell and its 'technically faster' neighbor
cell on snap-node 82859. Where do the destination weights really
diverge — c2c, c2z, or both? Are high-emp destinations migrating
between tiers when the origin shifts by ~100 m?
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from aperta import od_pairs


PREPARED_DIR = Path('data/prepared')
TIER_CUTOFFS_BIKE = dict(r_cells=1_000.0, r_medium=5_000.0, r_zones=25_000.0)

ISLAND_XY = (2_601_690, 1_199_940)
DEST_COL = 'employment_total'


def main():
    cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
    zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
    bike_graph = network_processing.load_consolidated_graphml(
        PREPARED_DIR / 'bike_graph.graphml')
    cells = cells.rename(columns={'node_id_bike': 'node_id'})
    zones = zones.rename(columns={'node_id_bike': 'node_id'})

    # Identify island + the 82859 cell ('technically faster' neighbor).
    island_pt = Point(*ISLAND_XY)
    island_id = cells.geometry.centroid.distance(island_pt).idxmin()
    island_row = cells.loc[island_id]
    island_node = int(island_row['node_id'])

    # The cell snapping to 82859 (one of the 6 nearest neighbors).
    fast_id = '8a1f8342cb9ffff'
    fast_row = cells.loc[fast_id]
    fast_node = int(fast_row['node_id'])

    print(f"Island cell {island_id}: centroid "
          f"({island_row.geometry.centroid.x:.0f}, {island_row.geometry.centroid.y:.0f}), "
          f"snap node {island_node}")
    print(f"'Fast' cell {fast_id}: centroid "
          f"({fast_row.geometry.centroid.x:.0f}, {fast_row.geometry.centroid.y:.0f}), "
          f"snap node {fast_node}")
    centroid_offset = ((island_row.geometry.centroid.x - fast_row.geometry.centroid.x)**2
                       + (island_row.geometry.centroid.y - fast_row.geometry.centroid.y)**2)**0.5
    print(f"Centroid offset: {centroid_offset:.0f} m\n")

    # Build pairs once (NODE-keyed).
    pairs = od_pairs.get_pairs(
        cells, node_column='node_id', zones=zones, **TIER_CUTOFFS_BIKE,
    )

    # Per-cell raw employment weight (for c2c destinations).
    cell_emp = cells[DEST_COL].to_dict()
    # Per-zone raw employment weight (for c2z destinations).
    zone_emp = cells.groupby('zone_id')[DEST_COL].sum().to_dict()
    # Zone node → zone_id lookup (c2z dest nodes are zone snap-nodes).
    zone_id_for_node = zones['node_id'].to_dict()
    zone_id_for_node = {v: k for k, v in zone_id_for_node.items()}   # invert

    print("=" * 70)
    print("c2c destinations (within r_cells = 1000 m)")
    print("=" * 70)

    # c2c is keyed by ORIGIN node. Both cells go through DIFFERENT origin
    # nodes (144987 vs 82859), so the dest sets differ structurally.
    # Map dest-node → list of cells snapped to that node.
    cells_by_node = cells.reset_index().groupby('node_id')['cell_id'].apply(list).to_dict()

    def c2c_dest_cells_and_weight(origin_node):
        """Returns (list of cell_ids, total raw emp weight) for c2c of origin_node."""
        dest_nodes = pairs.cells_to_cells.get(origin_node, np.array([]))
        all_cell_ids = []
        for n in dest_nodes:
            for cid in cells_by_node.get(n, []):
                all_cell_ids.append(cid)
        total_w = sum(cell_emp.get(cid, 0) for cid in all_cell_ids)
        return set(all_cell_ids), total_w

    island_c2c_cells, island_c2c_w = c2c_dest_cells_and_weight(island_node)
    fast_c2c_cells, fast_c2c_w = c2c_dest_cells_and_weight(fast_node)

    print(f"  Island ({island_node}): {len(island_c2c_cells):>4d} dest cells, "
          f"raw emp weight = {island_c2c_w:>7,.0f}")
    print(f"  Fast   ({fast_node}): {len(fast_c2c_cells):>4d} dest cells, "
          f"raw emp weight = {fast_c2c_w:>7,.0f}")

    only_island = island_c2c_cells - fast_c2c_cells
    only_fast = fast_c2c_cells - island_c2c_cells
    common = island_c2c_cells & fast_c2c_cells
    only_island_w = sum(cell_emp.get(c, 0) for c in only_island)
    only_fast_w   = sum(cell_emp.get(c, 0) for c in only_fast)
    common_w      = sum(cell_emp.get(c, 0) for c in common)

    print(f"\n  Common c2c cells:     {len(common):>4d}, raw emp weight = {common_w:>7,.0f}")
    print(f"  ONLY in island c2c:   {len(only_island):>4d}, raw emp weight = {only_island_w:>7,.0f}")
    print(f"  ONLY in fast c2c:     {len(only_fast):>4d}, raw emp weight = {only_fast_w:>7,.0f}")

    # Where do the 'only_island' high-emp cells end up for the fast cell?
    # → in its c2z tier? (i.e., their parent ZONE should appear in fast's c2z)
    print(f"\n  Top-10 cells ONLY in island c2c (by emp):")
    sorted_only_island = sorted(only_island, key=lambda c: -cell_emp.get(c, 0))[:10]
    for c in sorted_only_island:
        z = cells.loc[c, 'zone_id']
        zone_emp_total = zone_emp.get(z, 0)
        print(f"    {c}: emp={cell_emp.get(c, 0):>5.0f}  zone={z} (zone total emp={zone_emp_total:.0f})")

    print()
    print("=" * 70)
    print("c2z destinations (zones whose snap-node lies between r_cells and r_medium)")
    print("=" * 70)

    def c2z_dest_zones_and_weight(origin_node):
        dest_zone_nodes = pairs.cells_to_zones.get(origin_node, np.array([]))
        zones_set = set()
        for n in dest_zone_nodes:
            z = zone_id_for_node.get(n)
            if z is not None:
                zones_set.add(z)
        total_w = sum(zone_emp.get(z, 0) for z in zones_set)
        return zones_set, total_w

    island_c2z_zones, island_c2z_w = c2z_dest_zones_and_weight(island_node)
    fast_c2z_zones, fast_c2z_w = c2z_dest_zones_and_weight(fast_node)

    print(f"  Island ({island_node}): {len(island_c2z_zones):>4d} dest zones, "
          f"raw emp weight = {island_c2z_w:>7,.0f}")
    print(f"  Fast   ({fast_node}): {len(fast_c2z_zones):>4d} dest zones, "
          f"raw emp weight = {fast_c2z_w:>7,.0f}")

    common_z = island_c2z_zones & fast_c2z_zones
    only_island_z = island_c2z_zones - fast_c2z_zones
    only_fast_z = fast_c2z_zones - island_c2z_zones
    print(f"\n  Common c2z zones:     {len(common_z):>4d}, weight = "
          f"{sum(zone_emp.get(z, 0) for z in common_z):>7,.0f}")
    print(f"  ONLY in island c2z:   {len(only_island_z):>4d}, weight = "
          f"{sum(zone_emp.get(z, 0) for z in only_island_z):>7,.0f}")
    print(f"  ONLY in fast c2z:     {len(only_fast_z):>4d}, weight = "
          f"{sum(zone_emp.get(z, 0) for z in only_fast_z):>7,.0f}")

    # Check the migration hypothesis: do the "only_island" c2c cells'
    # parent zones appear in fast's c2z?
    island_only_c2c_zones = set(cells.loc[c, 'zone_id'] for c in only_island)
    migrated_to_fast_c2z = island_only_c2c_zones & fast_c2z_zones
    print(f"\n  Migration check: of the {len(island_only_c2c_zones)} parent zones "
          f"of cells uniquely in island c2c,")
    print(f"    {len(migrated_to_fast_c2z)} appear in fast's c2z (i.e. handled at zone level).")
    print(f"    Missing from fast altogether: "
          f"{len(island_only_c2c_zones - fast_c2z_zones - set(cells.loc[c, 'zone_id'] for c in fast_c2c_cells))}")

    print()
    print("=" * 70)
    print("TOTAL raw emp weight captured by each cell (c2c cells + c2z zones)")
    print("=" * 70)
    print(f"  Island: c2c={island_c2c_w:>7,.0f} + c2z={island_c2z_w:>7,.0f} = "
          f"{island_c2c_w + island_c2z_w:>8,.0f}")
    print(f"  Fast:   c2c={fast_c2c_w:>7,.0f} + c2z={fast_c2z_w:>7,.0f} = "
          f"{fast_c2c_w + fast_c2z_w:>8,.0f}")
    diff = (island_c2c_w + island_c2z_w) - (fast_c2c_w + fast_c2z_w)
    print(f"  Δ (island − fast) = {diff:+,.0f} raw emp")

    print()
    print("=" * 70)
    print("Per-tier decay-weighted gravity sum (from earlier diagnostic, floor=60)")
    print("=" * 70)
    print(f"  Island: c2c=12275  c2z=37213  total=49488  → logsum=54.05")
    print(f"  Fast:   c2c= 8702  c2z=32440  total=41143  → logsum=53.12")
    print(f"  Gap:    c2c= 3573  c2z= 4773  total= 8345  → 0.93 utils")


if __name__ == '__main__':
    main()
