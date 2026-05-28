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
# # Multi-modal accessibility — extended example
#
# Default scope: Bern + 25 km. Change `LOCATION_LABEL` and the crop
# constants below to retarget plots; the underlying data is whatever
# `prepare/1_download` produced (driven by `SEED_LOCATION` there).
#
# Tutorial-scope accessibility analysis across three modes (walk, bike,
# car) on real OSM data, with published-paper edge-weight coefficients
# applied inline (Miotti et al., *Transportation*). One representative
# variant per mode for clarity. Ends with a cross-modal logsum that
# combines all three modes into one accessibility surface.
#
# **Inputs** (all from `prepare/`):
#
# - 3 consolidated networks (`walk_graph`, `bike_graph`, `car_graph`)
#   carrying per-edge features from prep: `speed_kph`,
#   `density_norm`, `elev_gain`, `elev_loss`, `is_degree_4`,
#   `is_traffic_signal`. Edge travel times are computed inline in
#   section 2 below by applying published coefficients to these
#   features.
# - H3 cells (res 10) + zones (res 8) with `population`,
#   `employment_*`, `poi_*` columns, plus pre-snapped
#   `node_id_{walk,bike,car}` / `snap_dist_*` from
#   `prepare/3_unit_mapping`.
#
# **Three destinations**, picked to span the temporal-decay spectrum:
#
# | Destination          | Column                  | Decay profile       |
# |----------------------|-------------------------|---------------------|
# | Jobs (FTE)           | `employment_total`      | medium (commute)    |
# | Grocery shopping     | `poi_errands_groceries` | sharp (frequent)    |
# | Hiking POIs          | `poi_leisure_hiking`    | slow (destination)  |
#
# **Three modes** (one published variant each):
#
# | Mode           | Edge attribute       | Distance mask |
# |----------------|----------------------|---------------|
# | walk           | `walk_time_s`        | < 5 km        |
# | bike (regular) | `bike_time_s`        | < 25 km       |
# | car (off-peak) | `car_time_s_offpeak` | none          |

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

# === Plot retargeting knobs =================================================
# Keep in sync with `SEED_LOCATION` / `LOCATION_LABEL` in `prepare/1_download.py`.
# The crop is used by the per-mode / cross-modal map sections at the end.
LOCATION_LABEL = 'Bern'
MAP_CROP_CENTER_XY = (2_600_000, 1_199_000)   # LV95; centred on Bern
MAP_CROP_HALF_M = 3_500                        # 7 × 7 km window
# ============================================================================


# %% [markdown]
# ## 1. Load networks + geo units

# %%
walk_graph = network_processing.load_consolidated_graphml(PREPARED_DIR / 'walk_graph.graphml')
bike_graph = network_processing.load_consolidated_graphml(PREPARED_DIR / 'bike_graph.graphml')
car_graph = network_processing.load_consolidated_graphml(PREPARED_DIR / 'car_graph.graphml')

# `ox.load_graphml` returns edge attributes as strings (except those
# registered in `CONSOLIDATED_EDGE_DTYPES`). Cast the features we use
# in the edge-weight formula below back to float.
FEATURE_COLS = ('length', 'speed_kph', 'density_norm',
                'elev_gain', 'elev_loss',
                'is_degree_3', 'is_degree_4', 'is_traffic_signal')
for g in (walk_graph, bike_graph, car_graph):
    for _, _, d in g.edges(data=True):
        for c in FEATURE_COLS:
            if c in d:
                d[c] = float(d[c])

print(f"Walk: {walk_graph.number_of_nodes():>7,} nodes / "
      f"{walk_graph.number_of_edges():>7,} edges")
print(f"Bike: {bike_graph.number_of_nodes():>7,} nodes / "
      f"{bike_graph.number_of_edges():>7,} edges")
print(f"Car:  {car_graph.number_of_nodes():>7,} nodes / "
      f"{car_graph.number_of_edges():>7,} edges")

# %%
cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')

print(f"Cells: {len(cells):>6,}  (Σ pop {cells['population'].sum():>10,.0f}, "
      f"Σ jobs {cells['employment_total'].sum():>10,.0f})")
print(f"Zones: {len(zones):>6,}")

DESTINATIONS = ['employment_total', 'poi_errands_groceries', 'poi_leisure_hiking']
for d in DESTINATIONS:
    print(f"  Σ {d:25s}: {cells[d].sum():>10,.0f}")

# %%
# Origin / destination split — origins are only cells inside the AOI
# (seed + 5 km from `prepare/1_download`), but every cell in the dest
# polygon (AOI + 25 km) remains a valid destination. This dramatically
# cuts routing cost: each Dijkstra is one-to-many, so the cost scales
# with origin count, not destination count.
aoi_polygon = gpd.read_file(PREPARED_DIR / 'aoi_polygon.gpkg').geometry.iloc[0]
ORIG_MASK = cells.geometry.centroid.within(aoi_polygon)
print(f"\nOrigin cells: {ORIG_MASK.sum():,} of {len(cells):,} "
      f"({100 * ORIG_MASK.mean():.1f}%) inside AOI; "
      f"all {len(cells):,} remain valid destinations.")


# %% [markdown]
# ## 2. Apply published edge-weight coefficients → per-edge travel times
#
# **Coefficients from Miotti et al., *Transportation*, 20XX**, hardcoded
# inline for visibility. One representative variant per mode (walk,
# regular bike, off-peak car).
#
# **Formula** — per directed edge `u → v` of length `L` m:
#
# ```
# effective_kph = max(base_speed_kph · (1 + β_density · density_norm), floor_kph)
# duration_s    = L / (effective_kph / 3.6)
#               + α_up   · elev_gain                     (m climbed u→v)
#               + α_down · elev_loss                     (m descended u→v)
#               + β_intersection_4 · is_degree_4         (∈ {0, 0.5, 1})
#               + β_traffic_signal · is_traffic_signal   (∈ {0, 0.5, 1})
# ```
#
# Bike additionally caps the downhill speed at 50 km/h — without it,
# the negative `α_down` makes short steep descents produce
# negative-duration edges (Dijkstra silently churns).
#
# The simplifications vs the paper (also documented in the
# "what this notebook does NOT do" footer):
#
# - One variant per mode (no e-bike, no peak/night car).
# - `β_intersection` (3-way) absent — published 9-row schema only has
#   the 4-way coefficient.

# %%
def apply_edge_times(graph, attr_name: str, *,
                     base_speed_kph: float | None,
                     alpha_up: float, alpha_down: float,
                     beta_density: float,
                     beta_intersection_4: float, beta_traffic_signal: float,
                     floor_kph: float = 1.0,
                     max_downhill_kph: float | None = None) -> None:
    """Apply the published-coefficient formula and write to `attr_name`.

    `base_speed_kph=None` means "read per-edge `speed_kph`" (used for car).
    `max_downhill_kph` caps the implied speed on edges where the slope
    bonus would otherwise drive duration negative (used for bike).
    """
    min_dur_per_m = (None if max_downhill_kph is None
                     else 1.0 / (max_downhill_kph / 3.6))
    for _, _, _, data in graph.edges(keys=True, data=True):
        base_kph = data['speed_kph'] if base_speed_kph is None else base_speed_kph
        effective_kph = max(base_kph * (1 + beta_density * data['density_norm']),
                            floor_kph)
        base_dur = data['length'] / (effective_kph / 3.6)
        slope_pen = alpha_up * data['elev_gain'] + alpha_down * data['elev_loss']
        intersection_pen = (beta_intersection_4 * data['is_degree_4']
                            + beta_traffic_signal * data['is_traffic_signal'])
        total = base_dur + slope_pen + intersection_pen
        if min_dur_per_m is not None:
            total = max(total, data['length'] * min_dur_per_m)
        data[attr_name] = total


# Walk — slow, no density slowdown, intersection penalties zero in the
# paper (pedestrians don't wait at signals in the same way).
apply_edge_times(walk_graph, 'walk_time_s',
                 base_speed_kph=5.0,
                 alpha_up=2.4, alpha_down=0.1, beta_density=0.0,
                 beta_intersection_4=0.0, beta_traffic_signal=0.0)

# Bike (regular) — steeper climbs, slight downhill bonus (capped at
# 50 km/h to avoid negative durations on steep descents).
apply_edge_times(bike_graph, 'bike_time_s',
                 base_speed_kph=18.0,
                 alpha_up=3.1, alpha_down=-0.3, beta_density=0.0,
                 beta_intersection_4=7.0, beta_traffic_signal=1.0,
                 max_downhill_kph=50.0)

# Car (off-peak) — per-edge baseline from OSM `maxspeed`, density
# slowdown (urban friction), intersection + signal penalties.
apply_edge_times(car_graph, 'car_time_s_offpeak',
                 base_speed_kph=None,
                 alpha_up=0.0, alpha_down=0.0, beta_density=-0.20,
                 beta_intersection_4=6.0, beta_traffic_signal=10.0,
                 floor_kph=15.0)

for label, graph, attr in [('walk', walk_graph, 'walk_time_s'),
                            ('bike', bike_graph, 'bike_time_s'),
                            ('car',  car_graph,  'car_time_s_offpeak')]:
    times = np.array([d[attr] for _, _, d in graph.edges(data=True)])
    lengths = np.array([d['length'] for _, _, d in graph.edges(data=True)])
    implied_kph = (lengths / times) * 3.6
    print(f"  {label:5s} {attr:20s}: median edge time {np.median(times):>6.1f} s, "
          f"implied speed {np.median(implied_kph):.1f} km/h")


# %% [markdown]
# ## 3. Build tiered OD pairs — per network
#
# Per-mode tier cutoffs (Euclidean metres). The three tiers split each
# origin's destination universe into:
#
# - `cells_to_cells` (dest distance `< r_cells`): per-cell origin and
#   dest — the highest-precision tier, kept small to bound storage.
# - `cells_to_zones` (`r_cells ≤ d < r_medium`): per-cell origin,
#   zone-aggregated dest — preserves the origin precision where the
#   cell-to-cell pair count would explode but the relative cost of a
#   specific dest cell within its zone is still meaningful.
# - `zones_to_zones` (`r_medium ≤ d < r_zones`): zone-aggregated both
#   sides — coarse but cheap, sized for long-haul.
#
# Per-mode reasoning: faster modes reach further, so `r_zones` scales
# with mode speed. Walk barely needs a far tier (the 5 km mask in
# section 4 drops anything longer anyway); car runs all the way to the
# dest-polygon edge. The shared geo-unit grid (same cells / zones)
# means the per-mode ODMs can be lifted to `TieredODGeoPairs` and fed
# to `od_pairs.aggregate_across_modes` for the cross-modal logsum at
# the bottom of this notebook.

# %%
TIER_CUTOFFS = {
    'walk': dict(r_cells=1_000.0, r_medium=2_000.0,  r_zones=5_000.0),
    'bike': dict(r_cells=1_000.0, r_medium=5_000.0,  r_zones=25_000.0),
    'car':  dict(r_cells=1_000.0, r_medium=10_000.0, r_zones=100_000.0),
}

PAIRS = {}
for label, graph in [('walk', walk_graph),
                     ('bike', bike_graph),
                     ('car',  car_graph)]:
    pairs = od_pairs.get_pairs(
        cells, node_column=f'node_id_{label}',
        zones=zones, orig_cells=ORIG_MASK,
        **TIER_CUTOFFS[label],
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
    dists = od_pairs.get_euclidean_dists(nodes_xy, pairs)
    return od_pairs.make_mask(dists, lambda d: d < cutoff_m)

MASKS = {
    'walk': make_distance_mask(PAIRS['walk'], walk_graph, cutoff_m=5_000),
    'bike': make_distance_mask(PAIRS['bike'], bike_graph, cutoff_m=25_000),
    # car: no mask (any pair is potentially reachable in reasonable time).
}

def _mask_kept_pct(mask):
    tot = kept = 0
    for tier_name in ('cells_to_cells', 'cells_to_zones', 'zones_to_zones'):
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
#
# `cutoff=` truncates each per-origin Dijkstra once it crosses the
# given network-cost threshold. Picked per mode as a comfortable upper
# bound on plausible trip duration — destinations beyond it return
# `inf` and gravity / cumulative metrics drop them. Large speed-up on
# country-scale graphs without changing any answer.

# %%
ROUTING_PLAN = [
    # (label,             graph,      pairs,         mask,           edge attr,            cutoff_s)
    ('walk',              walk_graph, PAIRS['walk'], MASKS['walk'],  'walk_time_s',         60 * 60),
    ('bike_regular',      bike_graph, PAIRS['bike'], MASKS['bike'],  'bike_time_s',         60 * 60),
    ('car_offpeak',       car_graph,  PAIRS['car'],  None,           'car_time_s_offpeak', 120 * 60),
]
# This showcase uses one representative variant per mode. The published
# paper has more (e-bike 25/45, car peak/night) — see projects/lumos/ for
# the production pipeline that exercises all of them with per-scenario
# coefficient tables.
import time
COSTS = {}
ROUTING_TIMES = {}
for label, graph, pairs, mask, weight, cutoff_s in ROUTING_PLAN:
    t0 = time.perf_counter()
    COSTS[label] = routing.tiered_path_costs(
        pairs, graph, weight=weight, mask=mask, cutoff=cutoff_s,
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
    )
    dest_vals = {
        d: od_pairs.dest_values_geo(d, pairs_geo, cells, zones=zones)
        for d in DESTINATIONS
    }
    REINDEXED[net_label] = (pairs_geo, dest_vals)

# Per-variant: lift the cost ODM into geo-space.
GEO_PAIRS = {}
GEO_COSTS = {}
for label, _, pairs, _, _, _ in ROUTING_PLAN:
    net_label = NETWORK_OF[label]
    node_col = f'node_id_{net_label}'
    pairs_geo, cost_geo = od_pairs.reindex_by_geo_unit(
        pairs, COSTS[label], cells,
        cell_node_column=node_col,
        zones=zones, zone_node_column=node_col,
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
# middle / far tier overheads (which would weight cell-to-centroid
# distance by an aggregate of nearby cells) are deferred to the
# production version — the showcase keeps things simple.

# %%
# Per-cell density — shared across all modes. Same formula and radius as
# in `prepare/4_edge_weights`: sqrt of (pop+emp / km² within 1 km, then
# normalised by 10_000).
cells['pop_plus_emp'] = cells['population'] + cells['employment_total']
cells_centroids_gdf = cells.set_geometry(cells.geometry.centroid)
raw_per_m2 = geo_processing.cross_sum_within_radius(
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
for label, _, _, _, _, _ in ROUTING_PLAN:
    coef = OVERHEAD_COEF[label]

    # Per-cell origin & destination overhead = constant + density * density_norm.
    # No `snap_dist` term: cells are small, the contribution is minor, and the
    # published-paper value isn't pinned down for our updated coefficient table.
    overhead_per_cell = overhead.linear_per_cell_overhead(
        cells, constant=coef['const'],
        feature_coefficients={'density_norm': coef['density']},
    )

    # Cell-tier overheads only. Middle / far tiers get nothing added
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
for label, _, _, _, _, _ in ROUTING_PLAN:
    net_label = NETWORK_OF[label]
    _, dest_vals = REINDEXED[net_label]
    cost_floored = routing.floor_intrazonal_costs(
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
# framed, with a height-matched colour bar, and cropped to a square
# window centred on `MAP_CROP_CENTER_XY` (set at the top of this notebook).

# %%
import _figures as figures   # noqa: E402  — project-local plot helpers

ORIG_CELLS = cells.loc[ORIG_MASK]

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
for row, (label, _, _, _, _) in enumerate(ROUTING_PLAN):
    for col, d in enumerate(DESTINATIONS):
        figures.plot_cell_map_cropped(
            axes[row, col], ORIG_CELLS, ACC[(label, d)].loc[ORIG_MASK],
            crop_center_xy=MAP_CROP_CENTER_XY, crop_half_m=MAP_CROP_HALF_M,
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
logsum_costs_floored = routing.floor_intrazonal_costs(logsum_costs, min_cost=30.0)
# Destination weights: any of the per-network dest_vals will do — they're
# the same per-cell counts; geo-pairs alignment is handled internally.
logsum_dest_vals = {
    d: od_pairs.dest_values_geo(d, logsum_pairs, cells, zones=zones)
    for d in DESTINATIONS
}
logsum_decay = accessibility.exp_decay('logsum', 1.0 / LOGSUM_SCALE)
logsum_acc = accessibility.gravity(
    logsum_costs_floored, logsum_dest_vals, CELL_TO_ZONE, [logsum_decay],
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for col, d in enumerate(DESTINATIONS):
    figures.plot_cell_map_cropped(
        axes[col], ORIG_CELLS, logsum_acc[('logsum', d)].loc[ORIG_MASK],
        crop_center_xy=MAP_CROP_CENTER_XY, crop_half_m=MAP_CROP_HALF_M,
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
#   overhead at middle / far tiers via centroid-to-centroid distance.
#   Cells are small enough here that the difference is minor for a
#   showcase.
# - **No `overhead_*_dist` term.** The published coefficient table
#   doesn't have it; production may add it back per regional fit.
# - **Edge weights are paper-derived, not re-calibrated** for this
#   region. To re-derive them from ground-truth travel times, see
#   `calibrate_edge_weights.ipynb`. To add a traffic-flow / congestion
#   feature, see `traffic_flows.ipynb`. None of those are wired into this
#   notebook by design — each showcase notebook stands alone.
# - **Cross-modal logsum uses a coarse `θ = 60 s`.** Production
#   calibrates `θ` per destination class against revealed mode choice.
