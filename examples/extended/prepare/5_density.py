# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prep notebook 5: Per-node density + propagate node features to edges
#
# Two related steps:
#
# 1. **Compute per-node density.** Pop + employment in a 1 km buffer around
#    each node, normalised — a single dimensionless number per node that
#    summarises how dense the local context is. This is the same density
#    measure used in the published Miotti et al. coefficients
#    (`beta_density` term).
# 2. **Propagate per-node features to edges.** Density, intersection flags
#    (`is_degree_3`, `is_degree_4`), and traffic-signal flags are all
#    per-node by nature, but the edge-weight formula in downstream
#    notebooks needs them at the edge level. We propagate as the mean of
#    the two endpoint values per edge — so values land in `{0, 0.5, 1}`
#    for binary flags and a continuous mean for density.
#
# `is_degree_3` / `is_degree_4` / `is_traffic_signal` are already on the
# nodes from consolidation (`network_processing.consolidate_intersections`
# in notebook 1). We just need to push them to edges here.
#
# Outputs: the three `.graphml` files in `data/prepared/` get updated in
# place with new attributes:
#
# - **Per node**: `density_norm`
# - **Per edge**: `density_norm`, `is_degree_3`, `is_degree_4`,
#   `is_traffic_signal` (all endpoint-mean of the per-node values)

# %%
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox

from aperta import geo_processing, network_processing

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')

PREPARED_DIR = Path('../data/prepared')


# %% [markdown]
# ## 1. Load inputs

# %%
walk_graph = network_processing.load_consolidated_graphml(
    PREPARED_DIR / 'walk_graph.graphml')
bike_graph = network_processing.load_consolidated_graphml(
    PREPARED_DIR / 'bike_graph.graphml')
car_graph = network_processing.load_consolidated_graphml(
    PREPARED_DIR / 'car_graph.graphml')

# Cells supply the per-cell `population` + `employment_total` that the
# density aggregation sums into the per-node buffer.
cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg')
cells['pop_plus_emp'] = cells['population'] + cells['employment_total']
print(f"Cells: {len(cells):,} (Σ pop+emp = {cells['pop_plus_emp'].sum():,.0f})")


# %% [markdown]
# ## 2. Per-node density + propagate node features to edges
#
# Density formula (matches the Miotti et al. coefficients):
#
# ```
# density_norm = sqrt(pop_plus_emp_per_km² / 10_000)
# ```
#
# where the numerator aggregates pop+emp from cell centroids within
# 1 km of the node. `cross_sum_within_radius` with `return_density=True`
# returns the density per m²; multiplying by 100 (× 1e6 to convert to
# per-km², then ÷ 10_000) gives the quantity under the sqrt.

# %%
def add_density_and_propagate(graph: nx.MultiDiGraph, label: str) -> None:
    """Per-node density + endpoint-mean propagation to edges. Mutates `graph`."""
    print(f"\n--- {label} graph ---")

    node_ids = list(graph.nodes)
    from shapely.geometry import Point
    nodes_gdf = gpd.GeoDataFrame(
        {'node_id': node_ids},
        geometry=[Point(graph.nodes[n]['x'], graph.nodes[n]['y'])
                  for n in node_ids],
        crs=cells.crs,
    ).set_index('node_id')

    # Per-node density.
    raw_per_m2 = geo_processing.cross_sum_within_radius(
        targets=nodes_gdf, sources=cells, radius=1000.0,
        weight_column='pop_plus_emp', return_density=True,
    )
    density_norm = np.sqrt(raw_per_m2 * 100.0)
    nx.set_node_attributes(graph, density_norm.to_dict(), name='density_norm')
    print(f"  density_norm: median {density_norm.median():.3f}, "
          f"P95 {density_norm.quantile(0.95):.3f}, max {density_norm.max():.3f}.")

    # Propagate four per-node features to edges (endpoint mean). Density
    # is a continuous value; the others are binary {0, 1} on nodes and
    # become {0, 0.5, 1} on edges.
    NODE_FEATURES = (
        'density_norm', 'is_degree_3', 'is_degree_4', 'is_traffic_signal',
    )
    for u, v, k, data in graph.edges(keys=True, data=True):
        u_attr, v_attr = graph.nodes[u], graph.nodes[v]
        for f in NODE_FEATURES:
            data[f] = 0.5 * (float(u_attr.get(f, 0.0))
                             + float(v_attr.get(f, 0.0)))


add_density_and_propagate(walk_graph, 'walk')
add_density_and_propagate(bike_graph, 'bike')
add_density_and_propagate(car_graph, 'car')


# %% [markdown]
# ## 3. Save updated networks

# %%
ox.save_graphml(walk_graph, PREPARED_DIR / 'walk_graph.graphml')
ox.save_graphml(bike_graph, PREPARED_DIR / 'bike_graph.graphml')
ox.save_graphml(car_graph, PREPARED_DIR / 'car_graph.graphml')
print("\nGraphs updated with per-edge `density_norm`, `is_degree_3`, "
      "`is_degree_4`, `is_traffic_signal`.")
