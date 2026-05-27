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
# # `road_stress` — calibration + production estimate
#
# Estimates per-edge AADT for the car network via cost-decay-weighted
# nested-betweenness sampling, calibrated against observed traffic
# counters. Three knobs:
#
# 1. `lognorm_shape` (σ) — width of the trip-time distribution used as
#    the cost-decay weight in `nested_node_sample`.
# 2. `lognorm_scale` (μ) — location of the trip-time distribution.
# 3. `TRIPS_PER_PERSON_PER_DAY` — overall scaling of the betweenness
#    counts to vehicles/day.
#
# Initial values come from fitting a lognormal to ground-truth
# Google-Maps trip times + a default of 1.5 trips/person/day — a
# reasonable prior, but not the same thing as "what weighting produces
# flows that match traffic counters".
#
# ## Two design choices worth flagging upfront
#
# **Zones, not cells, as the granular unit.** Unlike `accessibility.ipynb`
# (which uses H3-res-10 cells, ~95k in the Bern dest polygon), road_stress
# uses H3-res-8 **zones** (~5500). Two reasons:
#
# - Flow estimation is bulk-attributing — thousands of OD pairs accumulate
#   on each road segment, so per-cell origin precision is washed out at
#   the edge level anyway. Zone granularity (~17 cells per zone) gives
#   enough origin resolution to differentiate density patterns without
#   the per-cell overhead.
# - The calibration loop reruns `simulate_flows` many times. Per-call
#   cost scales with the number of unique sampled origins; zone-level
#   sampling keeps each call snappy (~25 s on full Bern + 25 km).
#
# A single-tier OD structure (`cells_to_cells` only, with zones playing
# the role of "cells") suffices — no middle or far tier needed since the
# zone universe is small enough that the all-zones-within-200-km OD
# matrix fits comfortably in memory.
#
# **Origins are NOT restricted to the AOI.** Unlike accessibility (where
# AOI-restriction is fine because we only care about accessibility at
# origins inside the AOI), flow estimation has boundary effects: trips
# originating *outside* the AOI but passing through it contribute to
# observed counter readings. Restricting origins to AOI would
# systematically under-predict flows near the AOI boundary.
#
# ## Flow of the notebook
#
# 1. Load inputs (graph, zones, counters).
# 2. Snap counters to edges (bearing-aware, per-tier eligibility).
# 3. Build the fixed routing inputs (OD pairs, costs, lognormal prior).
# 4. `simulate_flows` helper — one parameter set → per-edge AADT.
# 5. Baseline evaluation at the prior values.
# 6. 1D scan over lognormal shape, pick best.
# 7. Derive `TRIPS_PER_PERSON_PER_DAY` from the slope.
# 8. **Production run** — re-simulate with calibrated params + save
#    `data/prepared/road_stress.csv`.
# 9. Visualise the per-edge stress map.
#
# Sections 5-7 use the library helpers `snap_counters_to_edges` and
# `evaluate_against_counters` from `aperta.calibration`; the scan
# loop + parameter selection are project-specific and live here.
#
# `data/prepared/road_stress.csv` is consumed by
# `calibrate_edge_weights.ipynb` as the `road_stress` feature for the
# BPR-style edge-weight calibration.

# %%
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from aperta import (
    calibration,
    network_processing,
    od_pairs,
    routing,
    traffic_flows,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')

PREPARED_DIR = Path('data/prepared')
GROUND_TRUTH_DIR = Path('data/ground_truth')
CRS_METRIC = 'EPSG:2056'

# Short name for plot titles + prints. Keep in sync with `LOCATION_LABEL`
# in `prepare/1_download.py`.
LOCATION_LABEL = 'Bern'


# %% [markdown]
# ## 1. Load inputs
#
# Graph + zones (the granular unit for flow estimation — see header
# rationale) + counters. The counters file holds directional point
# counters with `traffic_cars`, `bearing_deg`, and per-tier flags
# (`is_highway`, `is_main`, `is_local`).

# %%
car_graph = network_processing.load_consolidated_graphml(
    PREPARED_DIR / 'car_graph.graphml')
print(f"Car graph: {car_graph.number_of_nodes():,} nodes / "
      f"{car_graph.number_of_edges():,} edges")

zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
zones['pop_plus_emp'] = zones['population'] + zones['employment_total']
total_pop = float(zones['population'].sum())
print(f"Zones: {len(zones):,} (Σ pop {total_pop:,.0f})")

counters_all = gpd.read_file(GROUND_TRUTH_DIR / 'traffic_counters.gpkg')
print(f"Counters: {len(counters_all):,} "
      f"(highway: {int(counters_all['is_highway'].sum()):,}, "
      f"main: {int(counters_all['is_main'].sum()):,}, "
      f"local: {int(counters_all['is_local'].sum()):,})")

# Filter to counters inside the "core" area: dest_polygon shrunk by a
# buffer. The dest_polygon defines the analysis area, but counters near
# its edge see a lot of traffic going to / coming from places outside
# the model area (which the simulation can't account for, since trips
# with one endpoint outside aren't generated). Inner-core counters
# avoid that edge-effect bias — fit quality on this subset is ~10
# percentage points of R² better, and highway slope is near 1 even at
# trips=1.5 (the model is essentially correctly-scaled in the core).
dest_polygon = gpd.read_file(PREPARED_DIR / 'dest_polygon.gpkg').geometry.iloc[0]
INNER_BUFFER_M = 5000.0
inner_polygon = dest_polygon.buffer(-INNER_BUFFER_M)
in_inner = counters_all.geometry.within(inner_polygon)
counters = counters_all[in_inner].copy()
print(f"  inside core (dest_polygon − {INNER_BUFFER_M / 1000:.0f} km): "
      f"{len(counters):,} — used for calibration below")

legs = pd.read_csv(GROUND_TRUTH_DIR / 'car_pessimistic.csv')


# %% [markdown]
# ## 2. Snap counters to the car-graph edges
#
# Highway counters get a wider search radius (sparser layout, lower
# risk of catching a parallel local road) but can only snap to highway
# edges. Same for main / local. Bearing tolerance keeps the two
# opposite-direction counters on a two-way road from cross-snapping
# to the wrong directional edge.

# %%
HIGHWAY_RADIUS_M = 120.0
NON_HIGHWAY_RADIUS_M = 50.0
BEARING_TOL_DEG = 20.0

from aperta.network_processing import HIGHWAY_RANKS

# Per-edge tier flags. `_tier` returns the canonical bucket the counter
# expects to match against.
def _edge_tier(d) -> str:
    hwy = d.get('highway')
    if isinstance(hwy, list):
        hwy = hwy[0] if hwy else None
    rank = HIGHWAY_RANKS.get(hwy, -1)
    if rank >= 6: return 'highway'    # motorway / trunk
    if rank >= 3: return 'main'       # primary / secondary / tertiary
    return 'local'                    # residential / service / unknown

for _, _, _, d in car_graph.edges(keys=True, data=True):
    d['_tier'] = _edge_tier(d)

# Per-counter search radius (highway counters → wider).
search_radius = counters['is_highway'].map(
    {1: HIGHWAY_RADIUS_M, 0: NON_HIGHWAY_RADIUS_M})

def _eligible_for_counter(counter_row, candidate_edges):
    if counter_row['is_highway']:
        wanted = 'highway'
    elif counter_row['is_main']:
        wanted = 'main'
    else:
        wanted = 'local'
    return candidate_edges[candidate_edges['_tier'] == wanted]

snapped = calibration.snap_counters_to_edges(
    counters, car_graph,
    search_radius=search_radius,
    bearing_tol_deg=BEARING_TOL_DEG,
    eligible_edges=_eligible_for_counter,
)
counters = counters.join(snapped)
n_matched = counters['u'].notna().sum()
print(f"Snapped: {n_matched:,} of {len(counters):,} counters "
      f"({n_matched / len(counters) * 100:.1f}%)")
for tier_col, label in [('is_highway', 'highway'), ('is_main', 'main'),
                        ('is_local', 'local')]:
    mask = counters[tier_col] == 1
    n_tot = int(mask.sum())
    n_ma = int((mask & counters['u'].notna()).sum())
    if n_tot:
        print(f"  {label:8s}: {n_ma:,} of {n_tot:,} "
              f"({n_ma / n_tot * 100:.1f}%)")


# %% [markdown]
# ## 3. Build the (fixed) routing inputs
#
# Everything that *doesn't* depend on calibrated parameters goes here.
# The expensive routing happens once and gets reused across all
# simulation runs.

# %%
# Initial per-edge durations — features from prep: `speed_kph` (1_download),
# `density_norm`, `is_degree_4`, `is_traffic_signal` (5_density).
KMH_TO_MS = 1.0 / 3.6
INITIAL_MULT = {'density_norm': -0.45}
INITIAL_ADD = {'is_degree_4': 2.6, 'is_traffic_signal': 4.4}
for u, v, k, d in car_graph.edges(keys=True, data=True):
    base = float(d['length']) / (float(d['speed_kph']) * KMH_TO_MS)
    mult_term = base * sum(c * float(d[f]) for f, c in INITIAL_MULT.items())
    add_term = sum(c * float(d[f]) for f, c in INITIAL_ADD.items())
    d['duration_initial'] = max(base + mult_term + add_term, base * 0.2)

# Snap zone centroids to the car graph (zones play the role of "cells"
# in the OD structure below — see header rationale).
zone_centroids = zones.copy()
zone_centroids['geometry'] = zones.geometry.centroid
nid, _ = network_processing.snap_to_network_nodes(zone_centroids, car_graph)
zones['node_id'] = nid

# Single-tier OD pairs: `get_pairs` with no `zones=` kwarg returns a
# TieredODNodePairs with only the `cells_to_cells` slot populated. With
# r_cells=200 km the all-zones-within-200-km matrix is ~5500² entries,
# which fits comfortably at FP32 (~120 MB).
R_ZONE_PAIR_M = 200_000.0
CAR_TIME_CUTOFF_S = 2 * 3600  # 2-hour ceiling on any single trip

pairs = od_pairs.get_pairs(
    zones, r_cells=R_ZONE_PAIR_M, node_column='node_id',
)
costs = routing.tiered_path_costs(
    pairs, car_graph, weight='duration_initial', cutoff=CAR_TIME_CUTOFF_S,
)

zones['_dest_weight'] = zones['employment_total']
orig_weights = od_pairs.node_values(
    'pop_plus_emp', list(pairs.cells_to_cells.keys()),
    zones, 'node_id')
dest_weights = od_pairs.dest_values(
    '_dest_weight', pairs, zones, 'node_id')

# `nested_node_sample` calls `cell_to_zone_node.get(origin)` to look up
# the parent zone for the (absent) zone-tier and middle-tier. With a
# single-tier OD structure the lookup just returns None and the function
# skips the zone-tier code path entirely.
cell_to_zone_node = {}

# Initial lognormal prior fit from ground-truth times.
_pos_times = legs.loc[legs['time_measured'] > 0, 'time_measured']
shape0, _, scale0 = scipy.stats.lognorm.fit(_pos_times, floc=0)
print(f"Initial lognormal (from ground-truth times): "
      f"shape={shape0:.3f}, scale={scale0:.1f}")


# %% [markdown]
# ## 4. `simulate_flows` — one parameter set → per-edge AADT
#
# The expensive bit. Re-runs the betweenness pipeline for one
# `(shape, scale, trips_per_person_per_day)` triple. ~tens of seconds
# per call on this graph.

# %%
N_ORIG, N_DEST = 500, 250
RNG_SEED = 42

def simulate_flows(shape: float, scale: float,
                   trips_per_person_per_day: float = 1.5,
                   n_orig: int = N_ORIG, n_dest: int = N_DEST,
                   seed: int = RNG_SEED) -> pd.Series:
    """Run the full road_stress estimation for one parameter set."""
    def _cost_to_weight(c):
        return scipy.stats.lognorm.pdf(c, shape, 0.0, scale)
    rng = np.random.RandomState(seed)
    nested_sample = traffic_flows.nested_node_sample(
        pairs=pairs, weights=dest_weights, costs=costs,
        cell_to_zone_node=cell_to_zone_node, orig_weights=orig_weights,
        cost_to_weight=_cost_to_weight, n_orig=n_orig, n_dest=n_dest,
        random_state=rng,
    )
    edge_bc = network_processing.get_nested_edge_betweenness(
        car_graph, nested_sample, weights='duration_initial',
        cutoff=od_pairs.max_cost(costs),  # correctness-preserving Dijkstra cap
    )
    aadt_scale = (total_pop * trips_per_person_per_day) / (n_orig * n_dest)
    return edge_bc * aadt_scale


# %% [markdown]
# ## 5. Baseline evaluation at the prior values
#
# Run once with the lognormal-from-times prior + the default
# `trips_per_person_per_day = 1.5`. Sets the bar for the scans below.

# %%
TRIPS_PER_PERSON_PER_DAY_INIT = 1.5

flows_init = simulate_flows(shape0, scale0, TRIPS_PER_PERSON_PER_DAY_INIT)
eval_init = calibration.evaluate_against_counters(flows_init, counters)
eval_hw   = calibration.evaluate_against_counters(
    flows_init, counters[counters['is_highway'] == 1])
print(f"Baseline (shape={shape0:.3f}, scale={scale0:.1f}, "
      f"trips={TRIPS_PER_PERSON_PER_DAY_INIT}):")
print(f"  All inner counters:  R²={eval_init['r2']:.4f}, "
      f"slope={eval_init['slope']:.3f}, "
      f"RMSE={eval_init['rmse']:.0f}, n={eval_init['n_matched']:,}")
print(f"  Highway only:        R²={eval_hw['r2']:.4f}, "
      f"slope={eval_hw['slope']:.3f}, "
      f"RMSE={eval_hw['rmse']:.0f}, n={eval_hw['n_matched']:,}")

# Modeled-vs-observed scatter on log axes (counters span 0–100k AADT
# so a linear plot collapses everything below 5k against the axis).
# Highway counters coloured separately to show whether the high-AADT
# regime fits as well as the bulk.
m = eval_init['merged']
# Join `is_highway` flag back via the shared index (preserved by
# `evaluate_against_counters`'s dropna).
m = m.join(counters['is_highway'], how='left')
is_hw_mask = m['is_highway'] == 1

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(m.loc[~is_hw_mask, 'observed'], m.loc[~is_hw_mask, 'modeled'],
           s=8, alpha=0.3, color='tab:blue', label='main + local')
ax.scatter(m.loc[is_hw_mask, 'observed'], m.loc[is_hw_mask, 'modeled'],
           s=14, alpha=0.6, color='tab:red', label='highway')
lim = float(max(m['observed'].max(), m['modeled'].max())) * 1.1
ax.plot([1, lim], [1, lim], color='black', linewidth=0.5, label='1:1')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(50, lim); ax.set_ylim(50, lim)
ax.set_xlabel('Observed AADT (counter, veh/day)')
ax.set_ylabel('Modeled AADT (road_stress, veh/day)')
ax.set_title(f'Baseline fit — all: R²={eval_init["r2"]:.3f}, '
             f'slope={eval_init["slope"]:.2f}   |   '
             f'highway: R²={eval_hw["r2"]:.3f}, '
             f'slope={eval_hw["slope"]:.2f}')
ax.set_aspect('equal'); ax.legend()
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 6. Scan the lognormal **shape** (σ) — calibration in one parameter
#
# Holds `scale` fixed at the prior; sweeps `shape` over a small grid
# centred on the prior value. Picks the value that maximises R²
# against the counters.
#
# **This is the simplest possible calibration**: one parameter, one
# metric (R²), one pass. Production calibration (see
# `aperta-lab/projects/lumos/`) does coordinate-descent over multiple
# parameters with a scale-invariant combined loss, an inner-vs-outer
# counter filter, and a min-RMSE vs slope=1 trade-off for the final
# scaling. The library's `calibration.evaluate_against_counters`
# returns all the building blocks if you want to wire something
# fancier — `scale` and `trips_per_person_per_day` can be scanned the
# same way as `shape`.

# %%
shape_grid = np.linspace(0.6 * shape0, 1.4 * shape0, 5)
shape_results = []
for s in shape_grid:
    flows = simulate_flows(s, scale0, TRIPS_PER_PERSON_PER_DAY_INIT)
    ev = calibration.evaluate_against_counters(flows, counters)
    shape_results.append({'shape': s, 'r2': ev['r2'], 'slope': ev['slope']})
    print(f"  shape={s:.3f}: R²={ev['r2']:.4f}, slope={ev['slope']:.3f}")
shape_df = pd.DataFrame(shape_results)

best_idx = shape_df['r2'].idxmax()
shape_best = float(shape_df.loc[best_idx, 'shape'])
scale_best = scale0  # scale unscanned in this showcase
print(f"\nBest shape (by R²): {shape_best:.3f} "
      f"(R²={shape_df.loc[best_idx, 'r2']:.4f}, "
      f"slope={shape_df.loc[best_idx, 'slope']:.3f})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(shape_df['shape'], shape_df['r2'], 'o-', color='tab:blue')
ax.axvline(shape0, color='gray', linestyle='--', label=f'prior ({shape0:.3f})')
ax.axvline(shape_best, color='tab:red', label=f'best ({shape_best:.3f})')
ax.set_xlabel('lognormal shape (σ)'); ax.set_ylabel('R² vs counters')
ax.set_title('1D shape scan'); ax.legend()
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 7. Derive `trips_per_person_per_day` from the regression slope
#
# The shape scan picked by R² alone (scale-invariant), so the
# absolute-volume knob remains. Set `trips_per_person_per_day` to
# remove the systematic over/under-prediction:
#
# ```
# trips_new = trips_old × (1 / slope)
# ```
#
# This makes the no-intercept regression slope = 1 — modeled volumes
# match observed counters on average (weighted by observed magnitude).
# A more careful production analysis would compare this against a
# min-RMSE scaling and pick deliberately — see lumos.

# %%
flows_at_best = simulate_flows(shape_best, scale_best, TRIPS_PER_PERSON_PER_DAY_INIT)
eval_at_best = calibration.evaluate_against_counters(flows_at_best, counters)
trips_final = TRIPS_PER_PERSON_PER_DAY_INIT / eval_at_best['slope']
print(f"Slope at trips={TRIPS_PER_PERSON_PER_DAY_INIT} is {eval_at_best['slope']:.3f} "
      f"→ trips_per_person_per_day = {trips_final:.3f}")


# %% [markdown]
# ## 8. Production run + save `road_stress.csv`
#
# Re-runs `simulate_flows` with the calibrated `(shape, trips)` to
# produce the final per-edge AADT. Attaches as `road_stress` edge
# attribute (for the map below) and writes `data/prepared/road_stress.csv`
# for downstream consumers.

# %%
road_stress = simulate_flows(shape_best, scale_best, trips_final)
network_processing.set_nx_edge_attributes_filled(
    car_graph, road_stress.to_dict(), 'road_stress', fill_value=0.0)
print(f"road_stress (veh/day): "
      f"median {road_stress.median():.0f}, "
      f"P95 {road_stress.quantile(0.95):.0f}, "
      f"max {road_stress.max():.0f}")

OUTPUT_PATH = PREPARED_DIR / 'road_stress.csv'
road_stress_df = pd.DataFrame(
    [(u, v, k, float(road_stress.get((u, v, k), 0.0)))
     for u, v, k in car_graph.edges(keys=True)],
    columns=['u', 'v', 'k', 'road_stress'],
)
road_stress_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(road_stress_df):,} rows to {OUTPUT_PATH}")


# %% [markdown]
# ## 9. Visualise — raw flows on the network
#
# Per-edge `road_stress` as a colour map, cropped to 90 % of the
# destination polygon bounding box. Motorways drawn on top (sorted by
# `HIGHWAY_RANKS`) so they stay visible at junctions; colour scale
# clips at P99.
#
# A more sophisticated visualisation is *capacity-normalised stress* —
# `(V/C)^β` with `β` typically 2 or 4 (BPR convention). Divides
# road_stress by `capacity_per_lane[highway] · lanes_per_direction`
# (the per-direction lanes attribute is set by
# `consolidate_intersections`), then raises to `β`. Useful for spotting
# bottlenecks regardless of road class — a 4-lane motorway at 50k AADT
# has more spare capacity than a 2-lane primary at 20k. See
# `aperta-lab/projects/lumos/` for the production version.

# %%
import _figures as figures   # noqa: E402  — project-local plot helpers

stress_arr = np.array([float(d.get('road_stress', 0.0))
                       for _, _, d in car_graph.edges(data=True)])
xlim, ylim = figures.crop_to_polygon(dest_polygon)

fig, ax = plt.subplots(figsize=(12, 11))
figures.plot_network_map(
    ax, car_graph, road_stress,
    cbar_label='road_stress (veh/day, clipped at P99)',
    title=(f'road_stress — {LOCATION_LABEL} + 25 km '
           f'(median {np.median(stress_arr):.0f}, '
           f'P99 {np.quantile(stress_arr, 0.99):.0f}, '
           f'max {stress_arr.max():.0f} veh/day)'),
    xlim=xlim, ylim=ylim,
)
plt.tight_layout(); plt.show()


# %% [markdown]
# ## What this notebook does NOT do
#
# This is a self-contained showcase of *one* library capability —
# estimating per-edge traffic flows via cost-decay-weighted nested
# betweenness, then calibrating against observed counters. It
# deliberately omits things production code would do:
#
# - **Output isn't consumed by `accessibility.ipynb`.** That notebook
#   ships with published-paper edge weights instead of using estimated
#   flows. Each showcase notebook stands alone — they aren't wired into
#   a pipeline. Production *does* wire road_stress into edge-weight
#   calibration as a `(V/C)^β` BPR-style feature.
# - **Calibration is one parameter, one metric.** A real fit would
#   coordinate-descent over `shape`, `scale`, `trips_per_person_per_day`
#   with a combined scale-invariant loss + inner-vs-outer counter
#   filter, and pick the trips scaling deliberately (min-RMSE vs
#   slope=1 trade-off). Here we scan just `shape` and pick by R².
# - **No long-tail diagnostics.** Capacity-normalised `(V/C)²` maps
#   and per-edge outlier tables are useful for QA but are a deep dive.
#
# For an example of all the above wired into a production stack
# (scaffolding scenarios, typed I/O, dependency tracking), see
# [`aperta-lab/src/projects/lumos/`](https://github.com/mmiotti/aperta-lab/tree/main/src/projects/lumos).
