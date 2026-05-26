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
# # Swiss accessibility — multi-modal × multi-scale
#
# Production-scale accessibility analysis for Bern + 25 km on public data.
# Combines all four `prepare/` notebooks into a full accessibility
# computation across **5 mode-variants** × **3 destination types** plus
# **cross-modal logsum** aggregation — the multi-scale × multi-modal
# combination that motivates the aperta library.
#
# Inputs (all from `prepare/`):
#
# - 3 networks (`walk_graph`, `bike_graph`, `car_graph`) each carrying
#   per-edge travel-time attributes for 5 (mode, variant) combinations:
#   `walk_time_s`, `bike_time_s`, `ebike_time_s`,
#   `car_time_s_offpeak`, `car_time_s_peak`.
# - H3 cells (res 10), zones (res 8), regions (res 6) with
#   `population`, `employment_*`, `poi_*` columns.
# - Coefficient table at `coefficients/edge_weights_and_overhead.csv`
#   (the `overhead_*` rows for first/last-mile cost would go here too —
#   deferred to a later iteration to keep this notebook focused on the
#   routing → accessibility pipeline).
#
# **Three destinations**, picked to span the temporal-decay spectrum:
#
# | Destination          | Column                  | Decay profile       |
# |----------------------|-------------------------|---------------------|
# | Jobs (FTE)           | `employment_total`      | medium (commute)    |
# | Grocery shopping     | `poi_errands_groceries` | sharp (frequent)    |
# | Hiking POIs          | `poi_leisure_hiking`    | slow (destination)  |
#
# **Five mode-variants:**
#
# | Mode           | Variant   | Edge attribute        | Distance mask |
# |----------------|-----------|-----------------------|---------------|
# | walk           | —         | `walk_time_s`         | < 5 km        |
# | bike           | regular   | `bike_time_s`         | < 25 km       |
# | bike           | e-bike 25 | `ebike_time_s`        | < 25 km       |
# | car            | off-peak  | `car_time_s_offpeak`  | none          |
# | car            | peak      | `car_time_s_peak`     | none          |

# %%
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd

from aperta import (
    accessibility,
    geo_processing,
    network_processing,
    od_pairs,
    overhead,
    routing,
    visualization as viz,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')

PREPARED_DIR = Path('data/prepared')
CRS_METRIC = 'EPSG:2056'


# %% [markdown]
# ## 1. Load networks + geo units

# %%
walk_graph = network_processing.load_consolidated_graphml(PREPARED_DIR / 'walk_graph.graphml')
bike_graph = network_processing.load_consolidated_graphml(PREPARED_DIR / 'bike_graph.graphml')
car_graph = network_processing.load_consolidated_graphml(PREPARED_DIR / 'car_graph.graphml')

# `ox.load_graphml` returns edge attributes as strings — cast the time
# columns we need back to float so routing / arithmetic works.
TIME_COLS = {
    walk_graph: ['walk_time_s', 'length'],
    bike_graph: ['bike_time_s', 'ebike_time_s', 'length'],
    car_graph:  ['car_time_s_offpeak', 'car_time_s_peak', 'length'],
}
for g, cols in TIME_COLS.items():
    for _, _, d in g.edges(data=True):
        for c in cols:
            d[c] = float(d[c])

print(f"Walk: {walk_graph.number_of_nodes():>7,} nodes / "
      f"{walk_graph.number_of_edges():>7,} edges")
print(f"Bike: {bike_graph.number_of_nodes():>7,} nodes / "
      f"{bike_graph.number_of_edges():>7,} edges")
print(f"Car:  {car_graph.number_of_nodes():>7,} nodes / "
      f"{car_graph.number_of_edges():>7,} edges")

# %%
cells   = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
zones   = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
regions = gpd.read_file(PREPARED_DIR / 'regions.gpkg').set_index('region_id')

# H3 cells nest, so `region_id` for each zone is its parent at res 6
# (zones are res 8, regions are res 6). `get_pairs` requires this
# column to build the zones-to-regions tier.
import h3
zones['region_id'] = zones.index.map(lambda zid: h3.cell_to_parent(zid, 6))
print(f"Cells:   {len(cells):>6,}  (Σ pop {cells['population'].sum():>10,.0f}, "
      f"Σ jobs {cells['employment_total'].sum():>10,.0f})")
print(f"Zones:   {len(zones):>6,}")
print(f"Regions: {len(regions):>6,}")

DESTINATIONS = ['employment_total', 'poi_errands_groceries', 'poi_leisure_hiking']
for d in DESTINATIONS:
    print(f"  Σ {d:25s}: {cells[d].sum():>10,.0f}")

# %%
# Origin / destination split — origins are only cells inside the AOI
# (Bern + 5 km from `prepare/1_download`), but every cell in the dest
# polygon (Bern + 25 km) remains a valid destination. This dramatically
# cuts routing cost: each Dijkstra is one-to-many, so the cost scales
# with origin count, not destination count.
aoi_polygon = gpd.read_file(PREPARED_DIR / 'aoi_polygon.gpkg').geometry.iloc[0]
ORIG_MASK = cells.geometry.centroid.within(aoi_polygon)
print(f"\nOrigin cells: {ORIG_MASK.sum():,} of {len(cells):,} "
      f"({100 * ORIG_MASK.mean():.1f}%) inside AOI; "
      f"all {len(cells):,} remain valid destinations.")


# %% [markdown]
# ## 2. Snap geo units to nodes — per network
#
# Each of the 5 geo-units × 3 networks pairing needs a snap. We store
# the result as a per-mode column (`node_id_walk`, `node_id_bike`,
# `node_id_car`) on `cells`, `zones`, `regions` — keeping all three in
# the same DataFrame means downstream calls just pass the right
# `node_column=` and there's no data duplication.

# %%
def snap_layer_to_all_networks(layer: gpd.GeoDataFrame) -> None:
    """Mutate `layer` to add node-id + snap-distance columns for each network.

    Snap distances are reused later as the cell-to-network-node first-mile
    component of trip overheads (section 8).
    """
    centroids = layer.copy()
    centroids['geometry'] = centroids.geometry.centroid
    for graph, label in [(walk_graph, 'walk'),
                         (bike_graph, 'bike'),
                         (car_graph,  'car')]:
        nid, dist = network_processing.snap_to_network_nodes(centroids, graph)
        layer[f'node_id_{label}'] = nid
        layer[f'snap_dist_{label}'] = dist

for layer, name in [(cells, 'cells'), (zones, 'zones'), (regions, 'regions')]:
    snap_layer_to_all_networks(layer)
    print(f"  Snapped {name}: "
          f"walk {layer['node_id_walk'].notna().sum():>6,} "
          f"(median dist {layer['snap_dist_walk'].median():.0f} m), "
          f"bike {layer['node_id_bike'].notna().sum():>6,}, "
          f"car  {layer['node_id_car'].notna().sum():>6,}")


# %% [markdown]
# ## 3. Build tiered OD pairs — per network
#
# Same tier cutoffs for all 3 networks so the per-mode ODMs share a
# geo-unit grid and can be lifted to `TieredODGeoPairs` for cross-modal
# aggregation.
#
# Choice of cutoffs:
# - `r_cells = 1500 m` — dense cell-to-cell within ~20 min walk
# - `r_zones = 8000 m` — zone-tier covers ~30 min walk, ~15 min bike,
#   ~10 min car. Anything beyond is zone-to-region.

# %%
R_CELLS = 1000.0
R_ZONES = 7500.0
R_REGIONS = 100_000.0  # >> dest polygon extent → keep every region pair

PAIRS = {}
for label, graph in [('walk', walk_graph),
                     ('bike', bike_graph),
                     ('car',  car_graph)]:
    pairs = od_pairs.get_pairs(
        cells, r_cells=R_CELLS, node_column=f'node_id_{label}',
        zones=zones, r_zones=R_ZONES,
        regions=regions, r_regions=R_REGIONS,
        orig_cells=ORIG_MASK,
    )
    PAIRS[label] = pairs
    print(f"  {label:5s} {pairs}")


# %% [markdown]
# ## 4. Distance-based masks — drop unrealistic walk / bike trips
#
# Walking 27 km to work doesn't happen — masking out long pairs before
# routing skips Dijkstra for them entirely. The walk graph is huge
# (290k nodes); the mask cuts routing time roughly in proportion to the
# pairs dropped.

# %%
def make_distance_mask(pairs, graph, cutoff_m: float):
    """Build a TieredODNodePairs of bools from euclidean node distances."""
    nodes_xy = pd.DataFrame.from_dict(
        {n: (float(graph.nodes[n]['x']), float(graph.nodes[n]['y']))
         for n in graph.nodes}, orient='index', columns=['x', 'y'])
    dists = od_pairs.get_euclidian_dists(nodes_xy, pairs)
    return od_pairs.make_mask(dists, lambda d: d < cutoff_m)

MASKS = {
    'walk': make_distance_mask(PAIRS['walk'], walk_graph, cutoff_m=5_000),
    'bike': make_distance_mask(PAIRS['bike'], bike_graph, cutoff_m=25_000),
    # car: no mask (any pair is potentially reachable in reasonable time).
}

def _mask_kept_pct(mask):
    tot = kept = 0
    for tier_name in ('cells_to_cells', 'zones_to_zones', 'zones_to_regions'):
        tier = getattr(mask, tier_name)
        if tier is None:
            continue
        for arr in tier.values():
            tot += arr.size
            kept += int(arr.sum())
    return 100 * kept / tot if tot else 0.0

print(f"  walk mask: keeps {_mask_kept_pct(MASKS['walk']):.1f}% of pairs")
print(f"  bike mask: keeps {_mask_kept_pct(MASKS['bike']):.1f}% of pairs")


# %% [markdown]
# ## 5. Travel-cost ODMs — one per (mode, variant)
#
# `routing.tiered_path_costs` runs Dijkstra per origin, summing the
# specified edge attribute along each shortest path. Pairs with a
# `False` mask entry are skipped (the cost ODM stores `inf` for them,
# which gravity / cumulative metrics treat as unreachable).

# %%
ROUTING_PLAN = [
    # (label,             graph,      pairs,           mask,           edge attr)
    ('walk',              walk_graph, PAIRS['walk'],   MASKS['walk'],  'walk_time_s'),
    ('bike_regular',      bike_graph, PAIRS['bike'],   MASKS['bike'],  'bike_time_s'),
    ('car_offpeak',       car_graph,  PAIRS['car'],    None,           'car_time_s_offpeak'),
]
# This showcase uses one representative variant per mode. The published
# paper has more (e-bike 25/45, car peak/night) — see projects/lumos/ for
# the production pipeline that exercises all of them with per-scenario
# coefficient tables.
import time
COSTS = {}
ROUTING_TIMES = {}
for label, graph, pairs, mask, weight in ROUTING_PLAN:
    t0 = time.perf_counter()
    COSTS[label] = routing.tiered_path_costs(
        pairs, graph, weight=weight, mask=mask,
    )
    ROUTING_TIMES[label] = time.perf_counter() - t0
    print(f"  Routed {label:14s} ({weight:22s}) in {ROUTING_TIMES[label]:>6.1f} s",
          flush=True)
print(f"\nTotal routing time: {sum(ROUTING_TIMES.values()):.1f} s")


# %% [markdown]
# ## 6. Lift to geo-keyed form + destination weights
#
# Routing returns node-keyed cost ODMs. To get per-cell accessibility
# output and (later) cross-modal aggregation, lift each to a
# `TieredODGeoPairs` via `reindex_by_geo_unit`. Then build one
# destination-value ODM per (mode, destination) — needed because the
# per-mode destination universes differ (different masks → different
# pair sets).

# %%
NETWORK_OF = {
    'walk': 'walk', 'bike_regular': 'bike', 'car_offpeak': 'car',
}

# Per-network: reindex once, build destination-value ODMs once. Variants
# on the same network share these — only the *cost* ODM differs per
# variant.
REINDEXED = {}  # net_label -> (pairs_geo, {destination: dest_vals_geo})
for net_label in ('walk', 'bike', 'car'):
    node_col = f'node_id_{net_label}'
    pairs_geo, _ = od_pairs.reindex_by_geo_unit(
        PAIRS[net_label], PAIRS[net_label], cells,
        cell_node_column=node_col,
        zones=zones, zone_node_column=node_col,
        regions=regions, region_node_column=node_col,
    )
    dest_vals = {
        d: od_pairs.dest_values_geo(
            d, pairs_geo, cells, zones=zones, regions=regions,
        )
        for d in DESTINATIONS
    }
    REINDEXED[net_label] = (pairs_geo, dest_vals)

# Per-variant: lift the cost ODM into geo-space.
GEO_PAIRS = {}
GEO_COSTS = {}
for label, _, pairs, _, _ in ROUTING_PLAN:
    net_label = NETWORK_OF[label]
    node_col = f'node_id_{net_label}'
    pairs_geo, cost_geo = od_pairs.reindex_by_geo_unit(
        pairs, COSTS[label], cells,
        cell_node_column=node_col,
        zones=zones, zone_node_column=node_col,
        regions=regions, region_node_column=node_col,
    )
    GEO_PAIRS[label] = pairs_geo
    GEO_COSTS[label] = cost_geo

CELL_TO_ZONE = cells['zone_id'].to_dict()


# %% [markdown]
# ## 7. Trip overheads — origin + destination first/last-mile
#
# Each trip carries fixed per-mode overheads that don't depend on routing:
#
# - **Constant** per side: door-to-network time (e.g. 49 s for walking,
#   74 s for biking — getting out the bike, unlocking, etc., 52 s for
#   cars — finding parking, walking to/from).
# - **Density**: denser cells are slower to enter / leave (parking,
#   wayfinding). Uses `sqrt((pop + employment_total per km²) / 10_000)`
#   aggregated over a 1 km radius. Walking is unaffected (`coef = 0`);
#   cars are most affected (parking searches scale with density).
#
# **Coefficients from Miotti et al., *Transportation*, 20XX**, hardcoded
# inline below for visibility. The published table also has a constant
# `overhead_*_dist` term times cell-centroid-to-network-node distance —
# we drop it here as a tutorial simplification (cells are small enough
# that snap distances barely move the total), but the production
# pipeline in `projects/lumos/` uses the full per-mode value.
#
# We apply overheads only at the cell tier (origin + destination). The
# zone / region tier overheads (which would weight cell-to-centroid
# distance by an aggregate of nearby cells) are deferred to the
# production version — the showcase keeps things simple.

# %%
# Per-cell density — shared across all modes. Same formula and radius as
# in `prepare/4_edge_weights`: sqrt of (pop+emp / km² within 1 km, then
# normalised by 10_000).
cells['pop_plus_emp'] = cells['population'] + cells['employment_total']
cells_centroids_gdf = cells.set_geometry(cells.geometry.centroid)
raw_per_m2 = geo_processing.aggregate_within_radius(
    targets=cells_centroids_gdf, sources=cells_centroids_gdf,
    radius=1000.0, weight_column='pop_plus_emp', return_density=True,
)
cells['density_norm'] = np.sqrt(raw_per_m2 * 100.0)
print(f"Per-cell density_norm: median {cells['density_norm'].median():.3f}, "
      f"P95 {cells['density_norm'].quantile(0.95):.3f}, "
      f"max {cells['density_norm'].max():.3f}")

# %%
# Published overhead coefficients (Miotti et al., Transportation, 20XX).
# One representative variant per mode for this showcase — see ROUTING_PLAN.
OVERHEAD_COEF = {
    # mode           const(orig)  const(dest)  density(orig)  density(dest)
    'walk':         {'const': 49, 'density':   0},
    'bike_regular': {'const': 74, 'density':  66},   # orig & dest density both 66 for bike
    'car_offpeak':  {'const': 52, 'density': 128},   # orig & dest density differ in paper
}
# Per-paper dest_density values differ slightly from orig_density for some
# modes (e.g. car_offpeak: orig 128, dest 153). Showcase uses one value per
# mode for symmetry; production uses the full split.

# %%
# Bake the per-mode cell-tier overheads into each cost ODM.
GEO_COSTS_WITH_OVERHEAD = {}
for label, _, _, _, _ in ROUTING_PLAN:
    coef = OVERHEAD_COEF[label]

    # Per-cell origin & destination overhead = constant + density * density_norm.
    # No `snap_dist` term: cells are small, the contribution is minor, and the
    # published-paper value isn't pinned down for our updated coefficient table.
    overhead_per_cell = overhead.linear_per_cell_overhead(
        cells, constant=coef['const'],
        feature_coefficients={'density_norm': coef['density']},
    )

    # Cell-tier overheads only. Zone / region tiers get nothing added
    # (the showcase doesn't aggregate dest overhead at coarser tiers —
    # see footer for what production does differently).
    GEO_COSTS_WITH_OVERHEAD[label] = overhead.add_geo_overheads(
        GEO_COSTS[label], GEO_PAIRS[label],
        origin_cell=overhead_per_cell,
        dest_cell=overhead_per_cell,
    )
    print(f"  {label:14s}: per-cell overhead median {overhead_per_cell.median():>5.1f} s "
          f"(P95 {overhead_per_cell.quantile(0.95):>5.1f} s)")


# %% [markdown]
# ## 8. Per-mode gravity accessibility
#
# Exponential decay, one β per mode (rougher than per-mode × per-
# destination but enough to compare modes side-by-side). Cost ODM
# floored at 30 s so intrazonal self-pairs don't dominate.

# %%
# Per-mode decay parameters — half-decay around (walk 2.3 min, bike 3.9 min,
# car 5.8 min). Quick first pass; per-destination tuning comes later.
DECAYS = {
    'walk':         accessibility.exp_decay('walk',         0.005),
    'bike_regular': accessibility.exp_decay('bike_regular', 0.003),
    'car_offpeak':  accessibility.exp_decay('car_offpeak',  0.002),
}

ACC = {}  # (variant_label, destination) -> per-cell pd.Series
for label, _, _, _, _ in ROUTING_PLAN:
    net_label = NETWORK_OF[label]
    _, dest_vals = REINDEXED[net_label]
    cost_floored = routing.set_min_intrazonal_cost(
        GEO_COSTS_WITH_OVERHEAD[label], min_cost=30.0)
    result = accessibility.gravity(
        cost_floored, dest_vals, CELL_TO_ZONE, [DECAYS[label]],
    )
    for d in DESTINATIONS:
        ACC[(label, d)] = result[(label, d)]
        s = ACC[(label, d)]
        print(f"  {label:14s} × {d:24s}: "
              f"median {s.median():>10,.1f}  "
              f"P95 {s.quantile(0.95):>10,.1f}")


# %% [markdown]
# ## 9. Per-mode maps (3 modes × 3 destinations)
#
# Plot only origin cells (destination-only cells have no accessibility
# value and would otherwise show as a grey halo). All maps are square,
# framed, with a height-matched colour bar, and cropped to a 7 × 7 km
# window centred on Bern.

# %%
import _figures as figures   # noqa: E402  — project-local plot helpers

ORIG_CELLS = cells.loc[ORIG_MASK]

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
for row, (label, _, _, _, _) in enumerate(ROUTING_PLAN):
    for col, d in enumerate(DESTINATIONS):
        figures.plot_bern_cell_map(
            axes[row, col], ORIG_CELLS, ACC[(label, d)].loc[ORIG_MASK],
            title=f'{label} × {d}')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 10. Cross-modal logsum
#
# Combine walk + regular-bike + car off-peak into a single combined
# cost ODM via the discrete-choice logsum, then run gravity on top.
# The result is "how reachable is each destination if the agent picks
# the best available mode" — without committing to a single mode for
# the entire trip-set.
#
# This is the multi-modal half of the multi-scale × multi-modal
# combination that the toolkit paper makes the central claim about.

# %%
LOGSUM_SCALE = 60.0  # nest scale (θ); 60 s ≈ "1 minute = 1 utility unit"
logsum_pairs, logsum_costs = od_pairs.aggregate_across_modes(
    {
        'walk':         (GEO_PAIRS['walk'],         GEO_COSTS_WITH_OVERHEAD['walk']),
        'bike_regular': (GEO_PAIRS['bike_regular'], GEO_COSTS_WITH_OVERHEAD['bike_regular']),
        'car_offpeak':  (GEO_PAIRS['car_offpeak'],  GEO_COSTS_WITH_OVERHEAD['car_offpeak']),
    },
    aggregator='logsum', scale=LOGSUM_SCALE,
)
logsum_costs_floored = routing.set_min_intrazonal_cost(logsum_costs, min_cost=30.0)
# Destination weights: any of the per-network dest_vals will do — they're
# the same per-cell counts; geo-pairs alignment is handled internally.
logsum_dest_vals = {
    d: od_pairs.dest_values_geo(d, logsum_pairs, cells, zones=zones, regions=regions)
    for d in DESTINATIONS
}
logsum_decay = accessibility.exp_decay('logsum', 1.0 / LOGSUM_SCALE)
logsum_acc = accessibility.gravity(
    logsum_costs_floored, logsum_dest_vals, CELL_TO_ZONE, [logsum_decay],
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for col, d in enumerate(DESTINATIONS):
    figures.plot_bern_cell_map(
        axes[col], ORIG_CELLS, logsum_acc[('logsum', d)].loc[ORIG_MASK],
        title=f'Cross-modal logsum (walk+bike+car off-peak) × {d}')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## What this notebook does NOT do
#
# This is a tutorial-scope accessibility analysis using published-paper
# default coefficients (Miotti et al., *Transportation*, 20XX), chosen
# to demonstrate aperta's library API on real data with realistic edge
# weights. It deliberately omits things production code would do:
#
# - **Only one variant per mode.** Walk + regular bike + car off-peak.
#   The paper has more (e-bike 25/45, car peak/night) — see
#   [`projects/lumos/`](https://github.com/mmiotti/aperta-lab/tree/main/src/projects/lumos)
#   for the full variant matrix with peak vs off-peak congestion deltas
#   and bike vs e-bike comparisons.
# - **Cell-tier overheads only.** Production aggregates destination
#   overhead at zone / region tiers via centroid-to-centroid distance.
#   Cells are small enough here that the difference is minor for a
#   showcase.
# - **No `overhead_*_dist` term.** The published coefficient table
#   doesn't have it; production may add it back per regional fit.
# - **Edge weights are paper-derived, not re-calibrated** for this
#   region. To re-derive them from ground-truth travel times, see
#   `calibrate_edge_weights.ipynb`. To add a traffic-flow / congestion
#   feature, see `road_stress.ipynb`. None of those are wired into this
#   notebook by design — each showcase notebook stands alone.
# - **Cross-modal logsum uses a coarse `θ = 60 s`.** Production
#   calibrates `θ` per destination class against revealed mode choice.
