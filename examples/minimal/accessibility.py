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
# # Aperta minimal example: walking accessibility in Cambridge, MA
#
# This notebook runs the full aperta workflow end-to-end on a small real-world
# example: walking accessibility to supermarkets across Cambridge,
# Massachusetts. It takes ~5–10 minutes on a laptop. All data is pulled from
# public sources (OpenStreetMap via OSMnx; Uber's H3 grid system); no
# authentication or external downloads beyond `pip install` are needed.
#
# The notebook follows aperta's six-phase workflow:
#
# 1. **Load and prepare data** — OSM walking network, building footprints (as a
#    synthetic population proxy), and supermarket POIs.
# 2. **Map data to units** — Uber H3 hex cells (resolution 10, ~66 m edge) and
#    parent zones (resolution 8, ~460 m edge). H3's native parent-child
#    relationship makes the cell→zone mapping deterministic and zero-cost.
# 3. **Build sparse OD pairs** — Two tiers: cells-to-cells for near pairs,
#    zones-to-zones for far pairs.
# 4. *(skipped here)* Estimate traffic flows — not relevant for walking.
# 5. **Estimate travel costs** — Dijkstra shortest paths over the walking graph.
# 6. **Calculate accessibilities** — Cumulative-opportunity, gravity, and
#    nearest-*k* metrics.
#
# The same code structure works for any city anywhere in the world — only the
# area-of-interest string needs to change.

# %%
# Auto-reload aperta (and any other imported module) when its source files
# change on disk — saves a kernel restart on each library edit. The two lines
# below are IPython magics; running this notebook as a plain `.py` script is a
# no-op for them (a Python comment).
# %load_ext autoreload
# %autoreload 2

# %%
import warnings

import h3
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

import aperta.accessibility as accessibility
import aperta.geo_mapping as geo_mapping
import aperta.geo_processing as geo_processing
import aperta.network_processing as network_processing
import aperta.od_pairs as od_pairs
import aperta.overhead as overhead
import aperta.routing as routing
import aperta.utility as utility
import aperta.visualization as viz

warnings.filterwarnings('ignore', category=FutureWarning)


# %% [markdown]
# ## 1. Area of interest

# %%
PLACE = 'Cambridge, Massachusetts, USA'
H3_RES_CELLS = 10  # ~66 m hex edge — building-block scale
H3_RES_ZONES = 8   # ~460 m hex edge — small-neighbourhood scale

# OSMnx returns the place polygon in WGS84 (EPSG:4326).
boundary = ox.geocode_to_gdf(PLACE)
boundary_proj_crs = boundary.estimate_utm_crs()  # local metric CRS

print(f"Place:          {PLACE}")
print(f"Projected CRS:  {boundary_proj_crs}")

# %% [markdown]
# ## 2. Build H3 hex grids: cells and zones
#
# Uber's H3 is a hierarchical hex-grid system. Each H3 cell at resolution *N*
# has a deterministic parent at resolution *N–1*, *N–2*, ... — no spatial join
# needed for the cell→zone mapping.

# %%
boundary_geom = boundary.geometry.iloc[0]
cells = geo_processing.build_h3_grid(
    boundary_geom, H3_RES_CELLS,
    polygon_crs='EPSG:4326',
    target_crs=boundary_proj_crs,
)
# Each cell's parent at the zone resolution gives its zone assignment.
cells['zone_id'] = [h3.cell_to_parent(c, H3_RES_ZONES) for c in cells.index]

# Zones materialised from the unique zone parents.
zone_ids = sorted(cells['zone_id'].unique())
zones = gpd.GeoDataFrame(
    {'geometry': [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(z)])
                  for z in zone_ids]},
    index=pd.Index(zone_ids, name='zone_id'),
    crs='EPSG:4326',
).to_crs(boundary_proj_crs)

print(f"{len(cells):,} cells in {len(zones):,} zones")


# %% [markdown]
# ## 3. Fetch OSM data
#
# Three pulls: the walking network (for routing), building footprints (used as
# a synthetic population proxy), and supermarket POIs (destinations).

# %% [markdown]
# A note on the network filter. The intuitive choice is `network_type='walk'`,
# which excludes motorways and trunk roads. In practice this often produces
# small disconnected pedestrian islands at city boundaries: pedestrian-only
# paths (`highway=path`, `foot=designated`) that connect to the main grid
# only via a trunk-tagged road get orphaned when that road is filtered out,
# and OSMnx's default `retain_all=False` then drops them as non-largest
# components. The cleaner choice for accessibility analyses in well-mapped
# urban areas is `network_type='all'` — it keeps trunk roads (so the
# pedestrian paths stay connected) at the cost of letting the router walk
# along trunk roads in the (rare) cases where that's the shortest path.
# For a careful production analysis you would either patch the graph
# manually (re-add specific way IDs that bridge the gap) or use a custom
# filter that respects `sidewalk=*` tags. We use `'all'` here as the
# pragmatic compromise.

# %%
# Network. `graph_from_place` returns a MultiDiGraph clipped to the place
# polygon. See the note above on `network_type='all'` vs `'walk'`.
graph = ox.graph_from_place(PLACE, network_type='all', simplify=True)
graph = ox.project_graph(graph, to_crs=boundary_proj_crs)
print(f"Network: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# %%
# Building footprints — synthetic population proxy.
# In a real analysis, replace this with gridded population (GHSL, WorldPop).
# Project to a metric CRS before computing centroids — centroid in geographic
# CRS is geometrically meaningless.
_b = ox.features_from_place(PLACE, tags={'building': True})
_b = _b[_b.geometry.type.isin(['Polygon', 'MultiPolygon'])][['geometry']].to_crs(boundary_proj_crs)
buildings = gpd.GeoDataFrame(
    geometry=_b.geometry.centroid.values,
    index=pd.Index(range(len(_b)), name='building_id'),
    crs=boundary_proj_crs,
)
print(f"{len(buildings):,} building footprints (centroids used as population pseudo-points)")

# %%
# Supermarket POIs — destinations of interest for the accessibility analysis.
_s = ox.features_from_place(PLACE, tags={'shop': 'supermarket'}).to_crs(boundary_proj_crs)
supermarkets = gpd.GeoDataFrame(
    geometry=_s.geometry.centroid.values,
    index=pd.Index(range(len(_s)), name='supermarket_id'),
    crs=boundary_proj_crs,
)
print(f"{len(supermarkets):,} supermarkets")

# %% [markdown]
# A quick look at the inputs: the walking network (grey edges) and the
# supermarket destinations (red), inside the AOI boundary.

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ox.plot_graph(
    graph, ax=ax, node_size=0,
    edge_color='gray', edge_linewidth=0.3,
    bgcolor='white', show=False, close=False,
)
boundary.to_crs(boundary_proj_crs).boundary.plot(
    ax=ax, color='black', linewidth=0.6,
)
supermarkets.plot(
    ax=ax, color='red', markersize=30,
    edgecolor='black', linewidth=0.5, zorder=10,
)
ax.set_title(f'Inputs: walking network + {len(supermarkets)} supermarkets, {PLACE}')
ax.set_axis_off()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Map data to canonical units
#
# Three mappings: cells (and zones) get nearest network nodes; buildings and
# supermarkets get assigned to the cell whose polygon contains them.

# %%
# Walking speed in m/s (OSMnx default; used both for edge routing and for the
# cell→node walking-overhead estimate below).
WALK_SPEED_MS = 1.4

# Snap each cell's centroid to its nearest network node. `snap_to_network_nodes`
# returns both the node ID and the Euclidean distance from the cell centroid to
# that node — exactly what we need for the per-cell trip-overhead estimate.
cell_centroids_gdf = gpd.GeoDataFrame(
    geometry=cells.geometry.centroid, index=cells.index, crs=cells.crs,
)
cells['node_id'], dist_to_node = network_processing.snap_to_network_nodes(
    cell_centroids_gdf, graph,
)
# Walking overhead is the centroid→node distance divided by walking speed.
# Aperta's accessibility cell-mode adds this to every destination cost so that
# two cells sharing the same network node still report different accessibilities.
cells['walk_overhead_s'] = dist_to_node / WALK_SPEED_MS

# %% [markdown]
# **Zone snapping via tiered transport centroid.** Naively snapping each zone's
# *geometric* centroid to the nearest node often lands on something arbitrary —
# a service road behind a building, a cul-de-sac, or (with `network_type='all'`)
# even a motorway on-ramp. For zones at ~460 m H3 hex resolution covering
# heterogeneous urban geometry, this matters: the zone-representative node
# anchors all zone-tier OD-routing AND the routing-based destination-overhead
# calculations.
#
# The cleaner approach: classify nodes by the highest tier of road that
# touches them, exclude the top tier (motorways / trunk roads) and the bottom
# tier (pedestrian-only paths), then snap each zone to the eligible node
# nearest to the *median* coordinates of the zone's eligible interior nodes —
# the zone's transport-weighted centroid. This is what
# `assign_to_eligible_centroid` does.

# %%
# Map OSM highway tags to an ordinal road class (1 = pedestrian-only / off-road
# trails, 5 = primary / trunk). Used both for the tier-aware zone snap below
# and the path-first per-edge feature aggregation in §10.
HIGHWAY_CLASS_MAP = {
    'footway': 1, 'pedestrian': 1, 'path': 1, 'steps': 1, 'track': 1,
    'living_street': 2, 'residential': 2,
    'service': 3, 'unclassified': 3, 'tertiary': 3, 'tertiary_link': 3,
    'secondary': 4, 'secondary_link': 4,
    'primary': 5, 'primary_link': 5, 'trunk': 5, 'trunk_link': 5,
    'motorway': 5, 'motorway_link': 5,
}


def edge_road_class(u, v, data) -> float:
    """Per-edge road class. OSMnx stores `highway` as a string or list (for
    edges tagged with multiple types); pick the first and look it up,
    defaulting to 3 (mid-range) for tags we don't know."""
    h = data.get('highway')
    if isinstance(h, list):
        h = h[0] if h else None
    return float(HIGHWAY_CLASS_MAP.get(h, 3))


# Per-node tier = max road class among the node's connected edges.
node_road_class = network_processing.aggregate_edges_to_nodes(
    graph, edge_attribute=edge_road_class, aggregator='max',
)
# Eligible zone-snap targets: nodes whose highest-class touching road is
# residential through secondary (tiers 2–4). Excludes pure-pedestrian-path-only
# nodes (tier 1) — too minor to anchor a zone — and motorway / trunk nodes
# (tier 5) — pedestrians don't realistically arrive at those.
eligible_zone_nodes = node_road_class[
    (node_road_class >= 2) & (node_road_class <= 4)
].index
print(f"{len(eligible_zone_nodes):,} / {graph.number_of_nodes():,} graph nodes "
      f"are eligible as zone-snap targets (tier 2–4).")

# Snap each zone via the median of its eligible interior nodes (transport
# centroid), falling back to the geometric centroid for any zone with no
# eligible node inside.
zones['node_id'], _ = network_processing.assign_to_eligible_centroid(
    zones, graph, eligible_node_ids=eligible_zone_nodes,
)

# %%
# Assign each building to the cell whose polygon contains its centroid.
# `allow_nearest=True` catches the rare case of a centroid landing just
# outside the H3 hex coverage; the distance return is unused here.
buildings['cell_id'], _ = geo_mapping.map_points_to_polygons(
    buildings, cells, allow_nearest=True,
)
pop_by_cell = buildings.groupby('cell_id').size()
cells['population'] = pop_by_cell.reindex(cells.index, fill_value=0).astype(float)

# %%
# Same for supermarkets.
supermarkets['cell_id'], _ = geo_mapping.map_points_to_polygons(
    supermarkets, cells, allow_nearest=True,
)
sm_by_cell = supermarkets.groupby('cell_id').size()
cells['supermarkets'] = sm_by_cell.reindex(cells.index, fill_value=0).astype(float)

# %%
# Aggregate destination weights up to the zone level so the zone tier carries
# the right per-zone totals (one row per zone, summing cells below it).
zones['population'] = cells.groupby('zone_id')['population'].sum().reindex(
    zones.index, fill_value=0).astype(float)
zones['supermarkets'] = cells.groupby('zone_id')['supermarkets'].sum().reindex(
    zones.index, fill_value=0).astype(float)

print(f"Total pseudo-population: {cells['population'].sum():,.0f} buildings")
print(f"Cells with at least one supermarket: {(cells['supermarkets'] > 0).sum()}")


# %% [markdown]
# ## 5. Build tiered origin-destination pairs

# %%
R_CELLS = 1500.0   # 1.5 km: cell-tier near pairs (covers ~20 min of walking)
R_ZONES = 5000.0   # 5 km:   zone-tier far pairs

pairs = od_pairs.get_pairs(
    cells, r_cells=R_CELLS, node_column='node_id',
    zones=zones, r_zones=R_ZONES,
)
print(pairs)

# %% [markdown]
# Visualising the tiered structure from one origin. For any origin cell,
# aperta routes to *cell-tier destinations* (other cells within
# `r_cells`, at full cell resolution) and to *zone-tier destinations*
# (zones within `r_zones` but beyond cell-tier coverage, at zone
# resolution). This is the trade-off the multi-scale tiered architecture
# makes: high resolution where it matters (near the origin, where short
# trips dominate) and coarser resolution where it doesn't (farther away,
# where individual cell identity becomes statistically interchangeable).

# %%
# Pick an origin cell near the centre of the AOI for the illustration.
_center = boundary.to_crs(boundary_proj_crs).geometry.iloc[0].centroid
demo_cell_id = cells.geometry.centroid.distance(_center).idxmin()

# %%
viz.plot_tiered_destinations(
    cells, zones, pairs,
    origin_cell_id=demo_cell_id,
    graph=graph,
    boundary=boundary.to_crs(boundary_proj_crs),
    title=('Tiered destinations from one origin cell\n'
           'red = origin · gold = cell-tier dests (within r_cells) · '
           'blue = zone-tier dests (within r_zones, beyond r_cells)'),
)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Compute travel times (Dijkstra over the walking graph)

# %%
# Convert each edge's length (m) into a walking time (s).
for u, v, k, data in graph.edges(keys=True, data=True):
    data['walk_time_s'] = data['length'] / WALK_SPEED_MS

times = routing.tiered_path_costs(pairs, graph, weight='walk_time_s')
print(times)


# %% [markdown]
# ## 7. Lift to geo-keyed form, bake per-cell overhead, build weights
#
# Routing produces a *node-keyed* `TieredODNodePairs` — keys are network node
# IDs. To get **per-cell** accessibility output, baked **per-cell origin
# overhead**, or cross-modal aggregation across different graphs, lift to
# `TieredODGeoPairs` (keys = cell IDs / zone IDs).
#
# Three steps:
#
# 1. `reindex_by_geo_unit` — fan out node-keyed entries to cell/zone entries.
# 2. `add_origin_cell_overhead` — bake per-cell first-mile into the cost ODM
#    (per-cell at cell tier, per-zone-mean at zone tier).
# 3. `dest_values_geo` — build per-cell destination weight ODMs directly.

# %%
pairs_geo, times_geo = od_pairs.reindex_by_geo_unit(
    pairs, times, cells,
    cell_node_column='node_id',
    zones=zones, zone_node_column='node_id',
)
times_geo = overhead.add_origin_cell_overhead(
    times_geo, pairs_geo, cells, 'walk_overhead_s',
)
sm_weights = od_pairs.dest_values_geo(
    'supermarkets', pairs_geo, cells, zones=zones,
)
# Cell → zone lookup for tier stitching in the accessibility metrics.
cell_to_zone = cells['zone_id'].to_dict()

print(times_geo)


# %% [markdown]
# ## 8. Accessibility metrics
#
# Three flavours, computed against the geo-keyed cost ODM with overhead baked
# in. Output is indexed by cell ID — ready to join back to the cells
# GeoDataFrame for mapping. No per-call `cells=` / `node_column=` /
# `cell_overhead_column=` kwargs needed — the geo-keyed input does the work.

# %% [markdown]
# ### Cumulative-opportunity: supermarkets reachable per time band

# %%
bins = [
    accessibility.Bin('0_to_5min',   0,           5 * 60),
    accessibility.Bin('5_to_15min',  5 * 60,     15 * 60),
    accessibility.Bin('15_to_30min', 15 * 60,    30 * 60),
]
acc_cum = accessibility.count_in_bins(
    times_geo, {'supermarkets': sm_weights}, cell_to_zone, bins,
)

# Cumulative variant — supermarkets reachable WITHIN X minutes.
acc_within = pd.DataFrame({
    'within_5min':  acc_cum[('0_to_5min', 'supermarkets')],
    'within_15min': (acc_cum[('0_to_5min', 'supermarkets')]
                     + acc_cum[('5_to_15min', 'supermarkets')]),
    'within_30min': (acc_cum[('0_to_5min', 'supermarkets')]
                     + acc_cum[('5_to_15min', 'supermarkets')]
                     + acc_cum[('15_to_30min', 'supermarkets')]),
})
acc_within.head()

# %% [markdown]
# ### Gravity: exponential decay, three β values in one call
#
# The cost ODM is floored at 30 s via `set_min_intrazonal_cost` before passing
# to gravity. Without that floor, the intrazonal self-pair would route at
# cost 0, sending `exp(-β · 0) = 1` to maximum decay weight (giving a cell's
# own supermarkets infinite advantage over neighbours).

# %%
times_geo_floored = routing.set_min_intrazonal_cost(times_geo, min_cost=30.0)

decays = [
    accessibility.exp_decay('beta_005', 0.005),  # half-decay ~140 s ≈ 2 min
    accessibility.exp_decay('beta_002', 0.002),  # half-decay ~350 s ≈ 6 min
    accessibility.exp_decay('beta_001', 0.001),  # half-decay ~700 s ≈ 12 min
]
acc_gravity = accessibility.gravity(
    times_geo_floored, {'supermarkets': sm_weights}, cell_to_zone, decays,
)
acc_gravity.head()

# %% [markdown]
# ### Nearest-*k*: mean walking time to the *k* nearest supermarkets
#
# Default aggregator is `'cost_mean'` — the mean cost over the *k* nearest
# weight-units (each supermarket counts as one opportunity at its
# destination's cost). **Lower values = better accessibility**, and `k = 3`
# and `k = 5` are directly comparable on the same (seconds) scale. Cells with
# fewer than *k* reachable supermarkets show `NaN`.

# %%
acc_nk = accessibility.nearest_k(
    times_geo, {'supermarkets': sm_weights}, cell_to_zone, ks=[1, 3, 5],
)
acc_nk.head()


# %% [markdown]
# ## 9. Visualise
#
# Three choropleths, one per metric family. Cells with their accessibility
# value as fill colour; supermarket locations overlaid.

# %%
# Three panels, three metrics. The first two are "bigger = better"; the
# third (nearest-3 cost in minutes) is "smaller = better", so we use a
# reversed colormap to keep the visual convention "bright = good" consistent.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sm_overlay = [(supermarkets, {'color': 'red', 'markersize': 8})]
viz.plot_cell_values(
    cells, acc_within['within_15min'], ax=axes[0],
    title='Cumulative: supermarkets reachable within 15 min',
    overlays=sm_overlay,
)
viz.plot_cell_values(
    cells, acc_gravity[('beta_002', 'supermarkets')], ax=axes[1],
    title='Gravity (exp decay, β = 0.002)',
    overlays=sm_overlay,
)
viz.plot_cell_values(
    cells, acc_nk[(3, 'supermarkets')] / 60, ax=axes[2],
    title='Nearest-3: mean walk time to 3 closest supermarkets (min)',
    cmap='viridis_r',
    overlays=sm_overlay,
)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 10. Path-first: per-edge feature aggregation along realised routes
#
# Everything above used aperta's *cost* primitive — routing returns one scalar
# (travel time) per OD pair, and the accessibility metrics aggregate those
# scalars. But aperta is also **path-first**: `routing.tiered_path_aggregate`
# routes shortest paths and aggregates any per-edge attribute along each
# realised path, returning per-OD-pair scalars. Memory cost is the same as
# `tiered_path_costs` (paths are processed per-origin and discarded after
# aggregation).
#
# Concrete demo: for each OD pair, compute the **mean road class** of the
# edges traversed. OSM tags each road with a `highway` value (`residential`,
# `secondary`, `primary`, …); we map those to a 1–5 ordinal "road class"
# where higher = busier / less-pedestrian-friendly. Aggregated by mean
# along the route, this gives "how pedestrian-friendly are the streets
# I walk through on average?" — a route-aware quality metric that
# cumulative-opportunity / gravity / nearest-*k* cannot express, because
# they only see the OD cost, not the geometry of the route itself.

# %%
# `edge_road_class` was defined back in §4 for the tiered zone-snapping —
# we reuse it here as the per-edge attribute to aggregate along realised
# routes. `PathAggregation` matches the `Bin` / `Decay` namedtuple style
# used elsewhere in aperta: a named spec consisting of an attribute
# extractor and an aggregator.
costs, path_aggs = routing.tiered_path_aggregate(
    pairs, graph, weight='walk_time_s',
    edge_aggregations=[
        routing.PathAggregation('mean_road_class', edge_road_class, 'mean'),
    ],
)
mean_road_class_odm = path_aggs['mean_road_class']
print(mean_road_class_odm)

# %% [markdown]
# `mean_road_class_odm` is a `TieredODPairs` of the same shape as `times` —
# one scalar per OD pair, but now representing a route-geometry feature instead
# of a travel cost. To visualise per origin, take the unweighted mean across
# cell-tier destinations. (For a more focused metric, you could weight by
# `supermarkets` and only consider routes to supermarket cells; left as an
# exercise to keep this section short.)

# %%
per_origin_road_class = pd.Series(
    {origin: np.nanmean(values)
     for origin, values in mean_road_class_odm.cells_to_cells.items()},
    name='mean_road_class_on_walks',
)
# Map node-indexed values back to cell-indexed (many cells can share a node).
per_cell_road_class = cells['node_id'].map(per_origin_road_class)

viz.plot_cell_values(
    cells, per_cell_road_class,
    cmap='magma',
    title='Mean OSM road class along\nshortest-path walking routes '
          '(1 = pedestrian, 5 = primary)',
    overlays=[(supermarkets,
               {'color': 'cyan', 'markersize': 10, 'edgecolor': 'black'})],
)
plt.tight_layout()
plt.show()


# %% [markdown]
# This demonstrates the path-first design concretely: any per-edge feature
# you can attach to graph edges — surface type, gradient, perceived safety,
# pollution exposure, lit-vs-unlit, cycling-infrastructure presence — can
# be aggregated along realised routes with the same primitive, producing
# a per-OD or per-origin metric that captures *what the route is like*,
# not just how long it takes.


# %% [markdown]
# ## 11. Utility-based accessibility + logsum
#
# Building on the path-first primitive: define a per-mode *utility* function
# combining travel cost and per-edge route features, then derive
# discrete-choice-style accessibility metrics from it.
#
# A linear utility per OD pair:
#
#     U_ij = constant
#          + β_cost · cost_ij                       (routing weight)
#          + Σ_f β_f · aggregated_f(i, j)            (route features)
#          + Σ_o β_o · feature_o(i)                  (origin features)
#          + Σ_d β_d · feature_d(j)                  (destination features)
#
# For walking, we'll use a small example spec: time has a cost coefficient
# (less time = better), and the road class along the route has a moderately
# negative coefficient (pedestrian-friendly routes preferred over busy roads).

# %%
walking_utility = utility.Utility(
    constant=0.0,
    cost_coefficient=-0.01,  # utils per second — ≈ -0.6 per minute of walking
    route_features=[
        utility.RouteFeature(
            name='road_class',
            attribute=edge_road_class,   # reuses the callable defined above
            coefficient=-0.2,            # less utility on busier streets
            aggregator='mean',
        ),
    ],
)

# %% [markdown]
# **Step 1**: compute the route-dependent utility components
# (β_cost · cost + Σ β_route · feature). Wraps `tiered_path_aggregate`
# internally so the routing pass is shared across the cost and all features.

# %%
route_u = utility.route_utility(
    pairs, graph, cost_weight='walk_time_s', utility=walking_utility,
)
route_u

# %% [markdown]
# **Step 2**: add the constant + origin + destination components. This
# example has neither origin nor destination features in the utility (the
# destination "attractiveness" enters via the supermarket weight in the
# gravity step below), and constant = 0, so this is essentially a no-op —
# included for completeness of the workflow.

# %%
full_u = utility.add_endpoint_utility(route_u, pairs, walking_utility, cells=cells)

# %% [markdown]
# **Step 3**: gravity-on-utility with the `exp` decay = expected sum of
# attractiveness across destinations, weighted by `exp(utility)`. Taking the
# log of that sum gives the canonical *logsum accessibility* — the
# discrete-choice expected utility from the choice set.
#
# Lift to geo-keyed form, then bake the per-cell utility overhead
# (`β_cost · walk_overhead_s`) into the utility ODM via
# `add_origin_cell_overhead`. Units match (utils + utils), no per-call
# overhead kwarg needed.

# %%
# Lift full_u (node-keyed) to geo-keyed using the SAME `pairs_geo` we
# already built in §7 (origins / dests are the same cells).
_, full_u_geo = od_pairs.reindex_by_geo_unit(
    pairs, full_u, cells,
    cell_node_column='node_id',
    zones=zones, zone_node_column='node_id',
)
cells['util_overhead'] = walking_utility.cost_coefficient * cells['walk_overhead_s']
full_u_geo = overhead.add_origin_cell_overhead(
    full_u_geo, pairs_geo, cells, 'util_overhead',
)

# Custom Decay: gravity expects `exp(-β · cost)`, but utility is "more = better"
# (not a cost). Plain `np.exp` applied to utility gives the correct exp(U).
exp_utility_decay = accessibility.Decay('exp_u', np.exp)

gravity_u = accessibility.gravity(
    full_u_geo, {'supermarkets': sm_weights}, cell_to_zone, exp_utility_decay,
)
# Σ_j supermarkets_j · exp(U_ij), per origin cell.

logsum_accessibility = np.log(gravity_u[('exp_u', 'supermarkets')]).rename('logsum_acc')
logsum_accessibility.head()

# %% [markdown]
# Logsum is in the same units as utility (utils). It can be negative when the
# accessible attractiveness is low — that's expected for cells far from any
# supermarket or that face long walking-times on busy streets. *Less negative
# = better access.*
#
# Note: when there are no reachable supermarkets at all from an origin, the
# gravity sum is 0 and `log(0) = -inf`. Visualise these as missing cells.

# %%
# `plot_cell_values` replaces `-inf` (no-reachable cells) with NaN by default.
viz.plot_cell_values(
    cells, logsum_accessibility,
    title=('Logsum accessibility to supermarkets (walking)\n'
           'utility = −0.01·time − 0.2·mean_road_class; less negative = better access'),
    overlays=[(supermarkets,
               {'color': 'red', 'markersize': 10, 'edgecolor': 'black'})],
)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Cross-modal logsum
#
# The walk-only logsum above is the single-mode special case of the canonical
# discrete-choice accessibility:
#
#     A_i = ln Σ_j W_j Σ_m exp(U_ijm)
#
# §14 below demonstrates the *cross-modal* case: walking + driving combined
# via `od_pairs.aggregate_across_modes` at the geo-unit level. Aperta
# computes this natively across modes that live on different network graphs
# — one of the architectural distinctions called out in the paper.


# %% [markdown]
# ## 12. Adding a car mode (on a separate network graph)
#
# Cross-modal accessibility — combining walking and driving into a single
# "best-mode" measure — is one of aperta's distinguishing capabilities. The
# `TieredODGeoPairs` data structure and `aggregate_across_modes` helper let
# us route each mode on its own graph (different node IDs, different edge
# attributes) and then combine at the geo-unit level where IDs align.
#
# This section adds a driving network, computes density-adjusted edge travel
# times and a density-based parking penalty, then routes + lifts + bakes
# car-side overheads in the same pattern as walking.

# %% [markdown]
# ### 12.1 Fetch and snap the driving network

# %%
car_graph = ox.graph_from_place(PLACE, network_type='drive', simplify=True)
car_graph = ox.project_graph(car_graph, to_crs=boundary_proj_crs)
print(f"Car network: {car_graph.number_of_nodes():,} nodes, "
      f"{car_graph.number_of_edges():,} edges")

# Per-cell snapping (centroid → nearest car-network node), capturing the
# centroid→node distance for the per-cell first-mile overhead below.
cells['car_node_id'], car_dist_to_node = network_processing.snap_to_network_nodes(
    cell_centroids_gdf, car_graph,
)

# Zone snapping via transport centroid (same tier-aware logic as for walk).
# The 'drive' filter excludes pedestrian-only paths upfront, so eligible
# zone-snap targets are any tier-2-through-5 nodes.
car_node_road_class = network_processing.aggregate_edges_to_nodes(
    car_graph, edge_attribute=edge_road_class, aggregator='max',
)
car_eligible_nodes = car_node_road_class[
    (car_node_road_class >= 2) & (car_node_road_class <= 5)
].index
zones['car_node_id'], _ = network_processing.assign_to_eligible_centroid(
    zones, car_graph, eligible_node_ids=car_eligible_nodes,
)

# %% [markdown]
# ### 12.2 Building density around each car-network node
#
# Two car-mode features depend on local density: travel speed (denser =
# slower) and origin overhead (parking time rises with density). Both come
# from a single "buildings within R metres of each node" measure, computed
# in one KDTree pass over the building centroids we already downloaded.
#
# Density is a deliberately simple proxy here; the *calibration* notebook
# (a separate, more advanced example) covers fitting realistic edge weights
# from empirical travel-time data.

# %%
BUFFER_M = 200.0  # neighbourhood scale: count buildings within 200 m

# Convert graph nodes to a point GeoDataFrame so we can pass them as targets
# to `aggregate_within_radius`. OSMnx's `graph_to_gdfs(edges=False)` does
# exactly this: index = node IDs, geometry = node positions.
car_nodes_gdf = ox.graph_to_gdfs(car_graph, edges=False)
car_node_density = geo_processing.aggregate_within_radius(
    targets=car_nodes_gdf, sources=buildings,
    radius=BUFFER_M, return_density=True, name='density',
)
# Per-edge density: average of endpoint node densities.
for u, v, k, data in car_graph.edges(keys=True, data=True):
    data['density'] = 0.5 * (car_node_density[u] + car_node_density[v])

# Normalise density to [0, 1] for use as a penalty factor. Using the 95th
# percentile (not the max) so a handful of outlier nodes don't compress the
# rest of the distribution into the bottom of the scale.
_d_high = float(car_node_density.quantile(0.95)) or 1e-9
car_node_density_norm = (car_node_density / _d_high).clip(upper=1.0)
print(f"Car-node density (buildings/m²): "
      f"5-95 pct [{car_node_density.quantile(0.05):.5f}, "
      f"{car_node_density.quantile(0.95):.5f}]")

# %% [markdown]
# ### 12.3 Density-adjusted car edge travel times
#
# OSMnx fills in baseline speed limits per edge from OSM `maxspeed` tags
# (with sensible fallbacks per highway class) via `add_edge_speeds`. We then
# apply a density penalty:
#
#     effective_kph = speed_limit_kph × max(DENSITY_FLOOR, 1 − α · density_norm)
#
# floored at `DENSITY_FLOOR` of the speed limit so peak-density edges don't
# collapse to zero speed.

# %%
ox.add_edge_speeds(car_graph)
ALPHA_DENSITY = 0.6   # speed reduction strength
DENSITY_FLOOR = 0.3   # never go below 30 % of speed limit

for u, v, k, data in car_graph.edges(keys=True, data=True):
    speed_kph = float(data['speed_kph'])
    edge_density_norm = float(data['density']) / _d_high
    factor = max(DENSITY_FLOOR, 1.0 - ALPHA_DENSITY * min(edge_density_norm, 1.0))
    effective_kph = speed_kph * factor
    data['car_time_s'] = float(data['length']) / (effective_kph * 1000 / 3600)

# %% [markdown]
# ### 12.4 Car overheads — first-mile + density-based parking
#
# Per-cell origin overhead has two components:
#
# - **Centroid → assigned car node**, divided by a slow neighbourhood
#   driving speed (≈ 5 m/s — capturing the "back out of the driveway and
#   crawl to the main road" portion of any trip).
# - **Parking penalty at the assigned node**: a baseline plus a density-
#   scaled term (dense cells → harder to park).

# %%
CAR_INIT_SPEED_MS = 5.0     # slow neighbourhood driving speed
PARKING_BASE_S = 30.0       # baseline parking-search time
PARKING_DENSITY_S = 90.0    # additional parking time at peak density

cells['car_centroid_to_node_s'] = car_dist_to_node / CAR_INIT_SPEED_MS
car_parking_penalty_s = (
    PARKING_BASE_S + PARKING_DENSITY_S * car_node_density_norm
)
cells['car_overhead_s'] = (
    cells['car_centroid_to_node_s']
    + cells['car_node_id'].map(car_parking_penalty_s).fillna(PARKING_BASE_S)
)
print(f"Car overhead per cell: mean {cells['car_overhead_s'].mean():.1f}s, "
      f"5-95 pct [{cells['car_overhead_s'].quantile(0.05):.1f}, "
      f"{cells['car_overhead_s'].quantile(0.95):.1f}]")

# %% [markdown]
# ### 12.5 Car routing pipeline (mirrors the walk pipeline of §5–§11)

# %%
# Build car OD pairs and route.
car_pairs = od_pairs.get_pairs(
    cells, r_cells=R_CELLS, node_column='car_node_id',
    zones=zones, r_zones=R_ZONES,
)
car_times = routing.tiered_path_costs(car_pairs, car_graph, weight='car_time_s')
print(car_times)

# Build the car utility spec. Same structure as the walk one but with a
# smaller (in magnitude) road-class coefficient — cars typically prefer
# arterial roads, opposite preference to pedestrians.
car_utility = utility.Utility(
    constant=0.0,
    cost_coefficient=-0.01,
    route_features=[
        utility.RouteFeature(
            name='road_class',
            attribute=edge_road_class,
            coefficient=-0.05,  # weaker penalty than walking (-0.2)
            aggregator='mean',
        ),
    ],
)
car_route_u = utility.route_utility(
    car_pairs, car_graph, cost_weight='car_time_s', utility=car_utility,
)
car_full_u = utility.add_endpoint_utility(
    car_route_u, car_pairs, car_utility, cells=cells,
)

# Lift the car utility ODM to geo-keyed form, then bake the per-cell car
# origin overhead (in utility units: β_cost · car_overhead_s).
car_pairs_geo, car_full_u_geo = od_pairs.reindex_by_geo_unit(
    car_pairs, car_full_u, cells,
    cell_node_column='car_node_id',
    zones=zones, zone_node_column='car_node_id',
)
cells['car_util_overhead'] = car_utility.cost_coefficient * cells['car_overhead_s']
car_full_u_geo = overhead.add_origin_cell_overhead(
    car_full_u_geo, car_pairs_geo, cells, 'car_util_overhead',
)


# %% [markdown]
# ## 13. Destination overheads (last-mile)
#
# §7's `add_origin_cell_overhead` baked the **origin-side** overhead (each
# cell's first-mile from centroid to its network node) into the walking
# cost ODM. The symmetric **destination-side** overhead — the last-mile
# from the destination network node to the actual destination cell — is
# also material, especially for **short trips** where it consumes a real
# fraction of the total cost.
#
# `add_geo_overheads` applies destination overheads at each tier directly
# by unit ID (no node-ID re-keying):
#
# - **cell-tier dest** (`dest_cell=`): per-cell last-mile = each cell's own
#   first-mile (here `car_overhead_s`, which itself bundles the centroid →
#   node driving time plus a parking penalty). Passed directly from `cells`.
# - **zone-tier dest** (`dest_zone=`): per-zone last-mile = average over
#   cells in the zone of (Euclidean centroid distance / speed + each cell's
#   own first-mile), computed via
#   `aggregate_dest_overhead_per_group_euclidean`.
#
# We illustrate with the **car** mode and a **gravity** accessibility
# metric (continuous, on a shared scale across the two panels). Gravity is
# better than a cumulative threshold for this comparison: every OD pair
# contributes via `exp(-β · cost)`, so the overhead's impact varies
# *multiplicatively* across the map. Cells with longer routed times to
# begin with see a larger relative penalty from the same added overhead,
# making the spatial pattern of "where overhead bites hardest" easy to
# read.

# %%
# Build a car cost ODM WITHOUT any overheads — pure routed driving time,
# lifted to geo-keyed form. (`car_full_u_geo` from §12 already has origin
# overhead baked in, but that's a *utility* ODM; for the comparison we
# need a clean *cost* ODM to start from.)
_, car_times_geo_raw = od_pairs.reindex_by_geo_unit(
    car_pairs, car_times, cells,
    cell_node_column='car_node_id',
    zones=zones, zone_node_column='car_node_id',
)

# Per-zone destination overhead for the car mode.
zones['car_dest_overhead_s'] = overhead.aggregate_dest_overhead_per_group_euclidean(
    cells, zones, speed=CAR_INIT_SPEED_MS,
    group_id_column='zone_id', cell_overhead_column='car_overhead_s',
)
print(f"Per-zone car destination overhead: "
      f"mean {zones['car_dest_overhead_s'].mean():.1f}s, "
      f"5–95 pct [{zones['car_dest_overhead_s'].quantile(0.05):.1f}, "
      f"{zones['car_dest_overhead_s'].quantile(0.95):.1f}]")

# %%
# Apply BOTH origin and destination overheads on top of the raw car costs.
car_times_geo_full = overhead.add_origin_cell_overhead(
    car_times_geo_raw, car_pairs_geo, cells, 'car_overhead_s',
)
car_times_geo_full = overhead.add_geo_overheads(
    car_times_geo_full, car_pairs_geo,
    dest_cell=cells['car_overhead_s'],
    dest_zone=zones['car_dest_overhead_s'],
)

# %% [markdown]
# Compute gravity (β = 0.005, ≈ 2 min half-life) on both cost ODMs and
# render side-by-side.

# %%
# Destination weights aligned to the car pairs.
sm_weights_car = od_pairs.dest_values_geo(
    'supermarkets', car_pairs_geo, cells, zones=zones,
)
# Floor the raw ODM's intrazonal cost: cell-self-pair at cost 0 would
# otherwise contribute exp(-β·0) = 1 to the gravity sum, drowning out
# the rest. The "with overhead" ODM has finite self-pair cost already
# (origin + dest overheads on both sides), so no flooring there.
car_times_geo_raw_floored = routing.set_min_intrazonal_cost(
    car_times_geo_raw, min_cost=30.0)

car_decay = accessibility.exp_decay('beta_005', 0.005)
gravity_car_raw = accessibility.gravity(
    car_times_geo_raw_floored, {'supermarkets': sm_weights_car},
    cell_to_zone, car_decay,
)
gravity_car_full = accessibility.gravity(
    car_times_geo_full, {'supermarkets': sm_weights_car},
    cell_to_zone, car_decay,
)

# %%
viz.plot_cell_values_comparison(
    cells,
    {'Without overheads\n(pure routed driving time)':
         gravity_car_raw[('beta_005', 'supermarkets')],
     'With origin + destination overheads\n(first-mile + parking)':
         gravity_car_full[('beta_005', 'supermarkets')]},
    suptitle=('Gravity accessibility to supermarkets by car '
              '(β = 0.005, ≈ 2 min half-life)'),
    overlays=[(supermarkets, {'color': 'red', 'markersize': 8})],
)
plt.show()


# %% [markdown]
# The "with overhead" map is uniformly dimmer — every OD pair gets the
# overhead added, and `exp(-β · overhead)` multiplies the contribution by
# roughly `exp(-0.005 · 180 s) ≈ 0.41` for a typical ~3-minute combined
# origin + destination overhead. Spatial variation in the penalty is
# visible too: central cells in dense areas eat the highest parking
# penalty on BOTH the origin and the destination side, so they lose
# relatively more accessibility than suburban cells.
#
# Beyond the cumulative metric, the same `car_times_geo_full` ODM feeds into
# any other accessibility primitive — gravity, nearest-*k*, utility, logsum —
# unchanged. The destination overhead is a one-time augmentation of the cost
# ODM that propagates through the whole pipeline.


# %% [markdown]
# ## 14. Cross-modal logsum accessibility (walk + car)
#
# Combine the per-mode utility ODMs into a single combined ODM via
# `aggregate_across_modes` with a custom utility-domain logsum aggregator.
# The result expresses, per OD pair, the "expected max utility" across
# modes: `ln Σ_m exp(U_ijm)`. Downstream gravity-on-utility + log produces
# the canonical cross-modal logsum accessibility per cell.
#
# `aggregate_across_modes` unions the per-mode origin sets and the per-origin
# dest sets — for OD pairs reachable by one mode but not the other, the
# missing mode is treated as `inf` cost (NaN-and-inf-tolerant inside the
# aggregator).

# %%
def logsum_utility(stacked: np.ndarray) -> np.ndarray:
    """Cross-modal logsum in *utility* space (more = better).

    Stacked shape: `(n_modes, n_dests)`. Per OD pair, returns
    `ln Σ_m exp(U_ijm)` — the expected-max-utility across modes.
    NaN entries (mode unreachable for that OD pair) contribute 0 to the sum.
    """
    exp_terms = np.exp(stacked)
    exp_terms = np.where(np.isnan(exp_terms), 0.0, exp_terms)
    sum_exp = exp_terms.sum(axis=0)
    with np.errstate(divide='ignore'):
        return np.log(sum_exp)


# Combine the walk + car utility ODMs at the geo-unit level.
combined_u_pairs, combined_u = od_pairs.aggregate_across_modes(
    {'walk': (pairs_geo, full_u_geo),
     'car':  (car_pairs_geo, car_full_u_geo)},
    aggregator=logsum_utility,
)
print(combined_u_pairs)

# Destination weights aligned to the UNION dest set (combined_u_pairs may
# include cells reachable from some origins by car but not by walk, or
# vice versa — sm_weights from §7 was built against `pairs_geo` only).
sm_weights_combined = od_pairs.dest_values_geo(
    'supermarkets', combined_u_pairs, cells, zones=zones,
)

# Gravity-on-utility: Σ_j W_j · exp(combined_u_ij) = Σ_j W_j · Σ_m exp(U_ijm).
# Then log → canonical cross-modal logsum accessibility per cell.
exp_utility_decay = accessibility.Decay('exp_u', np.exp)
gravity_combined = accessibility.gravity(
    combined_u, {'supermarkets': sm_weights_combined}, cell_to_zone,
    exp_utility_decay,
)
cross_modal_logsum = np.log(
    gravity_combined[('exp_u', 'supermarkets')]
).rename('cross_modal_logsum')
cross_modal_logsum.head()

# %% [markdown]
# Visualisation: side-by-side walking-only logsum (from §11) and the
# cross-modal walking-or-driving logsum, on a shared colour scale. Cells
# where neither mode reaches any supermarket appear as missing (light grey).

# %%
viz.plot_cell_values_comparison(
    cells,
    {'Walking-only logsum': logsum_accessibility,
     'Cross-modal logsum (walking or driving)': cross_modal_logsum},
    suptitle='Logsum accessibility to supermarkets — less negative = better',
    overlays=[(supermarkets,
               {'color': 'red', 'markersize': 10, 'edgecolor': 'black'})],
)
plt.show()


# %% [markdown]
# The cross-modal map is uniformly *less negative* (better) than walking-only
# — adding a second mode can only expand reachability. The gap is largest in
# cells far from any supermarket: walking-only sees long travel times (very
# negative utility, so very small contribution to the gravity sum), but
# driving brings them within practical reach. In dense central cells with
# many walkable supermarkets, the walking utility already dominates the
# modal sum and the cross-modal addition is small.
#
# Architectural notes:
#
# - Each mode has its own graph, its own node IDs, its own edge attributes,
#   and its own snapping. Alignment happens only at the geo-unit (cell / zone)
#   level via `TieredODGeoPairs`.
# - Per-mode origin overhead is baked into the per-mode utility ODM
#   *before* aggregation, so the cross-modal logsum sees the right
#   mode-specific first-mile cost (walking time at 1.4 m/s vs slow driving
#   at 5 m/s + parking penalty).
# - `aggregate_across_modes` unions the per-origin dest sets across modes
#   and fills missing entries with `inf` (mode-unreachable). For our toy
#   example walk and car cover roughly the same Cambridge cells, but at
#   country scale this matters — different modes naturally reach different
#   sub-graphs.
# - Destination-side overheads (last-mile, also mode-specific) are NOT
#   baked into the utility ODMs in this minimal example. A production
#   analysis would add them via `add_geo_overheads(dest_cell=...,
#   dest_zone=...)` on each per-mode utility ODM before aggregation, using
#   mode-specific utils-per-second conversions.
