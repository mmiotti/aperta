"""Smoke tests for `aperta.visualization`.

Visualisation correctness is hard to assert numerically. These tests verify
the API surface — that each helper accepts its documented input shapes,
returns the right object type, and doesn't error on edge cases (missing
cells, all-NaN values, single-panel comparisons, etc.). Visual correctness
is assessed by eyeballing notebooks.

Uses the Agg backend so the tests run headlessly.
"""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import networkx as nx  # noqa: E402
from shapely.geometry import LineString, Point, box  # noqa: E402

from aperta.od_pairs import TieredODNodePairs  # noqa: E402
from aperta.visualization import (  # noqa: E402
    add_styled_colorbar,
    plot_cell_values,
    plot_cell_values_comparison,
    plot_edge_values,
    plot_tiered_destinations,
)


def _toy_cells_zones():
    """4 cells in a 2x2 grid grouped into 2 zones (top row Z1, bottom Z2)."""
    cells_geom = [
        box(0, 1, 1, 2), box(1, 1, 2, 2),  # top row → Z1
        box(0, 0, 1, 1), box(1, 0, 2, 1),  # bottom row → Z2
    ]
    cells = gpd.GeoDataFrame(
        {'zone_id': ['Z1', 'Z1', 'Z2', 'Z2'],
         'node_id': ['N1', 'N2', 'N3', 'N4']},
        geometry=cells_geom,
        index=pd.Index(['C1', 'C2', 'C3', 'C4'], name='cell_id'),
        crs='EPSG:3857',
    )
    zones_geom = [box(0, 1, 2, 2), box(0, 0, 2, 1)]
    zones = gpd.GeoDataFrame(
        {'node_id': ['ZN1', 'ZN2']},
        geometry=zones_geom,
        index=pd.Index(['Z1', 'Z2'], name='zone_id'),
        crs='EPSG:3857',
    )
    return cells, zones


class PlotCellValuesTestCase(unittest.TestCase):
    """`plot_cell_values` — single-panel choropleth."""

    def setUp(self):
        self.cells, _ = _toy_cells_zones()
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close('all')

    def test_with_series(self):
        values = pd.Series({'C1': 1.0, 'C2': 2.0, 'C3': 3.0, 'C4': 4.0})
        ax = plot_cell_values(self.cells, values, ax=self.ax)
        self.assertIs(ax, self.ax)

    def test_with_dict(self):
        values = {'C1': 1.0, 'C2': 2.0, 'C3': 3.0, 'C4': 4.0}
        plot_cell_values(self.cells, values, ax=self.ax)

    def test_with_column_name(self):
        cells = self.cells.assign(score=[1.0, 2.0, 3.0, 4.0])
        plot_cell_values(cells, 'score', ax=self.ax)

    def test_missing_cells_render(self):
        """Cells absent from the value series get the missing colour."""
        values = pd.Series({'C1': 1.0, 'C2': 2.0})  # C3, C4 missing
        plot_cell_values(self.cells, values, ax=self.ax)

    def test_neg_inf_treated_as_missing(self):
        values = pd.Series({'C1': -np.inf, 'C2': 1.0, 'C3': 2.0, 'C4': 3.0})
        plot_cell_values(self.cells, values, ax=self.ax)

    def test_creates_new_axes_if_none(self):
        ax = plot_cell_values(
            self.cells,
            pd.Series({'C1': 1.0, 'C2': 2.0, 'C3': 3.0, 'C4': 4.0}),
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_with_overlays_and_boundary(self):
        overlay = gpd.GeoDataFrame(
            geometry=[Point(0.5, 0.5), Point(1.5, 1.5)], crs='EPSG:3857')
        boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs='EPSG:3857')
        plot_cell_values(
            self.cells,
            pd.Series({'C1': 1.0, 'C2': 2.0, 'C3': 3.0, 'C4': 4.0}),
            ax=self.ax,
            overlays=[(overlay, {'color': 'red', 'markersize': 10})],
            boundary=boundary,
            title='Test plot',
        )

    def test_explicit_vmin_vmax(self):
        plot_cell_values(
            self.cells,
            pd.Series({'C1': 1.0, 'C2': 2.0, 'C3': 3.0, 'C4': 4.0}),
            ax=self.ax, vmin=0, vmax=10,
        )


class PlotCellValuesComparisonTestCase(unittest.TestCase):
    """`plot_cell_values_comparison` — side-by-side multi-panel."""

    def setUp(self):
        self.cells, _ = _toy_cells_zones()

    def tearDown(self):
        plt.close('all')

    def test_two_panels(self):
        vals = {
            'walk': pd.Series({'C1': 1.0, 'C2': 2.0, 'C3': 3.0, 'C4': 4.0}),
            'car':  pd.Series({'C1': 0.5, 'C2': 1.0, 'C3': 1.5, 'C4': 2.0}),
        }
        fig = plot_cell_values_comparison(self.cells, vals)
        self.assertEqual(len(fig.axes), 2 + 2)  # 2 plot axes + 2 legends

    def test_single_panel(self):
        vals = {'only': pd.Series({'C1': 1., 'C2': 2., 'C3': 3., 'C4': 4.})}
        plot_cell_values_comparison(self.cells, vals)

    def test_shared_scale_uses_global_min_max(self):
        """With shared_scale=True, both panels render the same colour range."""
        vals = {
            'low':  pd.Series({'C1': 1., 'C2': 2., 'C3': 3., 'C4': 4.}),
            'high': pd.Series({'C1': 10., 'C2': 20., 'C3': 30., 'C4': 40.}),
        }
        fig = plot_cell_values_comparison(self.cells, vals, shared_scale=True)
        # Both choropleths share colour limits; sanity-check by reading
        # the image extents.
        images = [a.collections[0] for a in fig.axes if a.collections]
        if len(images) >= 2:
            clim_a = images[0].get_clim()
            clim_b = images[1].get_clim()
            self.assertEqual(clim_a, clim_b)

    def test_independent_scale(self):
        vals = {
            'low':  pd.Series({'C1': 1., 'C2': 2., 'C3': 3., 'C4': 4.}),
            'high': pd.Series({'C1': 100., 'C2': 200., 'C3': 300., 'C4': 400.}),
        }
        plot_cell_values_comparison(self.cells, vals, shared_scale=False)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            plot_cell_values_comparison(self.cells, {})


class PlotTieredDestinationsTestCase(unittest.TestCase):
    """`plot_tiered_destinations` — single-origin tiered visualisation."""

    def setUp(self):
        self.cells, self.zones = _toy_cells_zones()
        # Pairs: from C1's node (N1), reach C2 (N2) at cell tier; from C1's
        # node also reach Z2 (ZN2) at the middle tier; from C1's zone
        # (Z1 → ZN1), reach Z2 (ZN2) at the far tier. (Z2 appears at both
        # middle + far tiers in this toy — fine, the function just draws
        # each layer separately.)
        self.pairs = TieredODNodePairs(
            cells_to_cells={'N1': np.array(['N2'])},
            cells_to_zones={'N1': np.array(['ZN2'])},
            zones_to_zones={'ZN1': np.array(['ZN2'])},
        )
        # Toy graph with the four cell nodes + two zone nodes positioned.
        self.graph = nx.Graph()
        for nid, (x, y) in [('N1', (0.5, 1.5)), ('N2', (1.5, 1.5)),
                            ('N3', (0.5, 0.5)), ('N4', (1.5, 0.5)),
                            ('ZN1', (1.0, 1.5)), ('ZN2', (1.0, 0.5))]:
            self.graph.add_node(nid, x=x, y=y)

    def tearDown(self):
        plt.close('all')

    def test_basic_without_graph(self):
        ax = plot_tiered_destinations(
            self.cells, self.zones, self.pairs, origin_cell_id='C1')
        self.assertIsInstance(ax, plt.Axes)

    def test_with_graph_node_markers(self):
        ax = plot_tiered_destinations(
            self.cells, self.zones, self.pairs,
            origin_cell_id='C1', graph=self.graph,
        )
        # Legend should be present (because graph was given and legend=True default)
        self.assertIsNotNone(ax.get_legend())

    def test_unknown_origin_cell_raises(self):
        with self.assertRaises(KeyError):
            plot_tiered_destinations(
                self.cells, self.zones, self.pairs, origin_cell_id='C_BOGUS')


class PlotEdgeValuesTestCase(unittest.TestCase):
    """`plot_edge_values` is a LineCollection-based per-edge plot with
    optional draw-order control."""

    def _graph(self) -> nx.MultiDiGraph:
        """3 directed edges in a row: (1,2), (2,3), (3,4)."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=10.0, y=0.0)
        g.add_node(3, x=20.0, y=0.0)
        g.add_node(4, x=30.0, y=0.0)
        for u, v in [(1, 2), (2, 3), (3, 4)]:
            g.add_edge(u, v, key=0,
                       geometry=LineString([(g.nodes[u]['x'], 0),
                                            (g.nodes[v]['x'], 0)]),
                       length=10.0, highway='primary')
        return g

    def test_returns_ax(self):
        g = self._graph()
        values = {(1, 2, 0): 1.0, (2, 3, 0): 2.0, (3, 4, 0): 3.0}
        ax = plot_edge_values(g, values, cmap='viridis')
        self.assertIsNotNone(ax)
        plt.close('all')

    def test_accepts_series_input(self):
        g = self._graph()
        values = pd.Series({(1, 2, 0): 1.0, (2, 3, 0): 2.0, (3, 4, 0): 3.0})
        ax = plot_edge_values(g, values)
        self.assertIsNotNone(ax)
        plt.close('all')

    def test_missing_edges_use_default(self):
        g = self._graph()
        values = {(1, 2, 0): 1.0}  # only one edge populated
        ax = plot_edge_values(g, values, default_value=0.0,
                              vmin=0.0, vmax=1.0)
        self.assertIsNotNone(ax)
        plt.close('all')

    def test_edges_without_geometry_fall_back_to_straight(self):
        """Raw OSMnx graphs may have edges without `geometry`. Should
        still plot using endpoint positions."""
        g = self._graph()
        # Strip geometry from one edge.
        del g[1][2][0]['geometry']
        values = {(1, 2, 0): 1.0, (2, 3, 0): 2.0, (3, 4, 0): 3.0}
        ax = plot_edge_values(g, values)
        self.assertIsNotNone(ax)
        plt.close('all')

    def test_sort_key_changes_collection_order(self):
        """With sort_key set, the LineCollection segments come in sorted
        order — verifiable by inspecting the collection's segments."""
        g = self._graph()
        values = {(1, 2, 0): 1.0, (2, 3, 0): 2.0, (3, 4, 0): 3.0}
        # Sort descending by value (high value drawn last on top).
        ax = plot_edge_values(g, values, sort_key=lambda k, d: -values[k])
        coll = ax.collections[-1]
        segs = coll.get_segments()
        # First segment should be the lowest-value edge (1,2) — sorted last.
        # Actually with -values, sort ascending of -value = descending of value,
        # so highest value first. So segs[0] should be edge (3,4).
        self.assertEqual(tuple(segs[0][0]), (20.0, 0.0))  # edge (3,4) starts at x=20
        plt.close('all')

    def test_dict_edge_widths(self):
        g = self._graph()
        values = {(1, 2, 0): 1.0, (2, 3, 0): 2.0, (3, 4, 0): 3.0}
        widths = {(1, 2, 0): 0.5, (2, 3, 0): 2.0, (3, 4, 0): 5.0}
        ax = plot_edge_values(g, values, edge_widths=widths)
        coll = ax.collections[-1]
        self.assertEqual(list(coll.get_linewidths()), [0.5, 2.0, 5.0])
        plt.close('all')


class AddStyledColorbarTestCase(unittest.TestCase):
    """`add_styled_colorbar` appends a height-matched colour bar."""

    def test_returns_cax(self):
        fig, ax = plt.subplots()
        cax = add_styled_colorbar(ax, cmap='viridis', vmin=0, vmax=10,
                                  label='test')
        self.assertIsNotNone(cax)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
