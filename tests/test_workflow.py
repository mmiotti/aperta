"""End-to-end integration test for the canonical aperta workflow.

This test serves two purposes:

  1. **Regression guard at the seams** between modules. Unit tests cover each
     module in isolation; this test verifies they compose — a signature change
     or shape mismatch between `od_pairs → routing → accessibility` would fire
     here.
  2. **Runnable minimal example.** Read this file top-to-bottom to see what a
     typical aperta workflow looks like in ~150 lines: build the cell/zone
     hierarchy + the network graph, compute OD pairs and distances at three
     tiers, route, derive accessibilities.

Toy world: 6 cells in 3 zones (2 cells per zone).

    zone ZA (x=0):  cell A0 (10) , A1 (20)        zone node a0
    zone ZB (x=3):  cell B0 (30) , B1 (40)        zone node b0
    zone ZC (x=10): cell C0 (50) , C1 (60)        zone node c0

    network: undirected, edges with `length`:
        a0-a1 (1), b0-b1 (1), c0-c1 (1), a0-b0 (3), b0-c0 (7)

With `r_cells=2`, `r_medium=5`, `r_zones=15`:
  - cells_to_cells: within-zone pairs only (cell 0 ↔ cell 1 inside each zone,
    plus the trivial self-pair: same-zone-as-itself is always cell-tier).
  - cells_to_zones (middle tier): ZA-ZB (d=3 < 5) → per-cell origin, zone dest.
  - zones_to_zones (far tier): ZA-ZC (d=10) and ZB-ZC (d=7), both in [5, 15).

Conservation invariant: for every origin, the total destination weight visible
across all tiers and bins equals `total_pop` exactly.
"""
import unittest

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from aperta import accessibility, od_pairs, routing


class WorkflowTestCase(unittest.TestCase):
    """One end-to-end happy-path run through the canonical workflow."""

    @classmethod
    def setUpClass(cls):
        # -- Cells: the finest-resolution geographic unit, with population. ---
        # Each cell knows its zone and its network node.
        cell_rows = [
            # (cell_id, x, y, node_id, zone_id, population)
            ('A0',  0, 0, 'a0', 'ZA', 10),
            ('A1',  0, 1, 'a1', 'ZA', 20),
            ('B0',  3, 0, 'b0', 'ZB', 30),
            ('B1',  3, 1, 'b1', 'ZB', 40),
            ('C0', 10, 0, 'c0', 'ZC', 50),
            ('C1', 10, 1, 'c1', 'ZC', 60),
        ]
        cls.cells = gpd.GeoDataFrame(
            {
                'node_id_nw': [r[3] for r in cell_rows],
                'zone_id':    [r[4] for r in cell_rows],
                'population': [r[5] for r in cell_rows],
            },
            geometry=[Point(r[1], r[2]) for r in cell_rows],
            index=pd.Index([r[0] for r in cell_rows], name='cell_id'),
            crs='EPSG:2056',
        )

        # -- Zones: the coarser unit. Population is the sum of its cells.
        # Each zone's representative network node is its `node_id_nw`.
        zone_rows = [
            # (zone_id, node_id, centroid_x, centroid_y, population)
            ('ZA', 'a0',  0, 0.5,  30),
            ('ZB', 'b0',  3, 0.5,  70),
            ('ZC', 'c0', 10, 0.5, 110),
        ]
        cls.zones = gpd.GeoDataFrame(
            {
                'node_id_nw': [r[1] for r in zone_rows],
                'population': [r[4] for r in zone_rows],
            },
            geometry=[Point(r[2], r[3]) for r in zone_rows],
            index=pd.Index([r[0] for r in zone_rows], name='zone_id'),
            crs='EPSG:2056',
        )

        # -- Nodes: one Point per network node, used by `get_euclidian_dists`.
        cls.nodes = gpd.GeoDataFrame(
            geometry=[Point(r[1], r[2]) for r in cell_rows],
            index=pd.Index([r[3] for r in cell_rows], name='node_id'),
            crs='EPSG:2056',
        )

        # -- Network: a plain undirected `nx.Graph` is enough. Aperta accepts
        # any nx graph type (`Graph` / `DiGraph` / `MultiGraph` / `MultiDiGraph`);
        # use whichever shape your data naturally has.
        g = nx.Graph()
        for r in cell_rows:
            g.add_node(r[3], x=r[1], y=r[2])
        for u, v, length in [
            ('a0', 'a1', 1.0), ('b0', 'b1', 1.0), ('c0', 'c1', 1.0),
            ('a0', 'b0', 3.0), ('b0', 'c0', 7.0),
        ]:
            g.add_edge(u, v, length=length)
        cls.graph = g

    def test_workflow_end_to_end(self):
        # ===== Phase 3: tiered OD pairs ==================================
        pairs = od_pairs.get_pairs(
            self.cells, r_cells=2.0, node_column='node_id_nw',
            zones=self.zones, r_zones=15.0, r_medium=5.0,
        )
        # All three tiers populated for this toy world. Plain `assert` (not
        # `assertIsNotNone`) so the type checker narrows `dict | None → dict`
        # for the rest of the test.
        assert pairs.cells_to_cells is not None
        assert pairs.cells_to_zones is not None
        assert pairs.zones_to_zones is not None
        # Cell-tier: a0's same-zone cells = {a0, a1} (self-pairs are included).
        self.assertEqual(set(pairs.cells_to_cells['a0']), {'a0', 'a1'})
        # Middle (cells_to_zones): ZA-ZB qualifies (d=3 < r_medium=5).
        # Cell a0 (in ZA) sees ZB's node b0 as a middle-tier dest.
        self.assertEqual(set(pairs.cells_to_zones['a0']), {'b0'})
        self.assertEqual(set(pairs.cells_to_zones['a1']), {'b0'})
        # Cells in ZC have no middle-tier dests (no other zone within d<5).
        self.assertNotIn('c0', pairs.cells_to_zones)
        # Far (zones_to_zones): ZA-ZC (d=10) and ZB-ZC (d=7), both in [5, 15).
        self.assertEqual(set(pairs.zones_to_zones['a0']), {'c0'})
        self.assertEqual(set(pairs.zones_to_zones['b0']), {'c0'})
        self.assertEqual(set(pairs.zones_to_zones['c0']), {'a0', 'b0'})

        # ===== Phase 3 (cont.): euclidean OD distances ===================
        dists = od_pairs.get_euclidian_dists(self.nodes, pairs)
        # cell-tier dists for a0 are [a0→a1, a0→a0] = {0, 1} regardless of order.
        self.assertEqual(sorted(dists.cells_to_cells['a0'].tolist()), [0.0, 1.0])

        # ===== Phase 3 (cont.): per-tier destination weights =============
        pop = od_pairs.dest_values(
            'population', pairs, self.cells, 'node_id_nw', zones=self.zones)
        assert pop.cells_to_zones is not None and pop.zones_to_zones is not None
        # Cell-tier dests for a0 are {a0 (pop 10), a1 (pop 20)} → sum 30.
        self.assertEqual(sorted(pop.cells_to_cells['a0'].tolist()), [10.0, 20.0])
        # Middle tier (cells_to_zones): ZB's aggregate (30 + 40).
        np.testing.assert_array_equal(pop.cells_to_zones['a0'], np.array([70.0]))
        # Far tier (zones_to_zones): from a0 (zone ZA's node) → ZC, pop 110.
        np.testing.assert_array_equal(pop.zones_to_zones['a0'], np.array([110.0]))

        # ===== Phase 5: routed travel costs ==============================
        times = routing.tiered_path_costs(pairs, self.graph, 'length')
        assert times.cells_to_zones is not None and times.zones_to_zones is not None
        # cells_to_cells for a0: {a0→a1=1, a0→a0=0}.
        self.assertEqual(sorted(times.cells_to_cells['a0'].tolist()), [0.0, 1.0])
        # a0 → b0 along the direct cross-zone edge (middle tier).
        np.testing.assert_array_almost_equal(
            times.cells_to_zones['a0'], np.array([3.0]))
        # a0 → c0 = a0 → b0 → c0 = 3 + 7 (far tier).
        np.testing.assert_array_almost_equal(
            times.zones_to_zones['a0'], np.array([10.0]))

        # ===== Phase 6: accessibility in cost bins =======================
        cell_to_zone_node = od_pairs.build_cell_to_zone_node_map(
            self.cells, self.zones, 'node_id_nw')
        bins = [
            accessibility.Bin('short',   0,   5),
            accessibility.Bin('medium',  5,  15),
            accessibility.Bin('long',   15, 100),
        ]
        df = accessibility.count_in_bins(
            times, {'population': pop}, cell_to_zone_node, bins)

        # Output shape: 6 origins × (3 bins × 1 property).
        self.assertEqual(df.shape, (6, 3))
        self.assertEqual(df.columns.names, ['bin', 'property'])
        self.assertFalse(df.isna().any().any())

        # Hand-computed values for a0:
        #   short  [0, 5):  cell-tier a0 (self, cost 0, pop 10) + cell-tier a1
        #                   (cost 1, pop 20) + middle-tier b0 (cost 3, pop 70) = 100
        #   medium [5, 15): far-tier c0 (cost 10, pop 110) = 110
        #   long   [15, ∞): 0
        self.assertEqual(df.loc['a0', ('short',  'population')], 100.0)
        self.assertEqual(df.loc['a0', ('medium', 'population')], 110.0)
        self.assertEqual(df.loc['a0', ('long',   'population')], 0.0)

        # Hand-computed values for c0 (no middle-tier dests; only cell + far):
        #   short:  c0 (self, cost 0, pop 50) + c1 (cost 1, pop 60) = 110
        #   medium: ZB via far (cost 7, pop 70) + ZA via far (cost 10, pop 30) = 100
        self.assertEqual(df.loc['c0', ('short',  'population')], 110.0)
        self.assertEqual(df.loc['c0', ('medium', 'population')], 100.0)

        # Conservation: sum of all bins per origin == total population.
        # Cell-tier includes the origin's own cell as a self-pair (cost 0), so
        # every cell/zone in this connected toy world contributes its full
        # weight to *some* bin from every origin.
        total_pop = float(self.cells['population'].sum())
        for origin in df.index:
            seen = float(df.loc[origin].sum())
            self.assertAlmostEqual(seen, total_pop,
                                   msg=f"conservation for origin {origin}")


if __name__ == '__main__':
    unittest.main()
