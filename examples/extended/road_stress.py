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
# Estimates per-edge AADT for the car network via nested betweenness,
# with three knobs calibrated against observed traffic counters:
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
# **Flow** of the notebook:
#
# 1. Load inputs (graph, zones, regions, counters).
# 2. Snap counters to edges (bearing-aware, per-tier eligibility).
# 3. Build the fixed routing inputs (OD pairs, costs, lognormal prior).
# 4. `simulate_flows` helper — one parameter set → per-edge AADT.
# 5. Baseline evaluation at the prior values.
# 6. 1D scan over lognormal shape, pick best.
# 7. 1D scan over lognormal scale (using best shape), pick best.
# 8. Derive `TRIPS_PER_PERSON_PER_DAY` from the slope.
# 9. **Production run** — re-simulate with calibrated params + save
#    `data/prepared/road_stress.csv`.
# 10. Visualise (raw flows + capacity-normalised maps).
# 11. Investigate the (V/C)² long tail.
#
# Sections 5-8 use the library helpers `snap_counters_to_edges` and
# `evaluate_against_counters` from `aperta.calibration`; the scan
# loops + parameter selection are project-specific and live here.
#
# `data/prepared/road_stress.csv` is consumed by
# `calibrate_edge_weights.ipynb` as the `road_stress` feature for the
# BPR-style edge-weight calibration.

# %%
import warnings
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import scipy.stats
from shapely.geometry import Point

from aperta import (
    calibration,
    geo_processing,
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


# %% [markdown]
# ## 1. Load inputs
#
# Same graph + zones + regions setup as `road_stress.ipynb`. Plus the
# counters file: directional point counters with `traffic_cars`,
# `bearing_deg`, and per-tier flags (`is_highway`, `is_main`,
# `is_local`).

# %%
car_graph = network_processing.load_consolidated_graphml(
    PREPARED_DIR / 'car_graph.graphml')
print(f"Car graph: {car_graph.number_of_nodes():,} nodes / "
      f"{car_graph.number_of_edges():,} edges")

cells = gpd.read_file(PREPARED_DIR / 'cells.gpkg').set_index('cell_id')
cells['pop_plus_emp'] = cells['population'] + cells['employment_total']
zones = gpd.read_file(PREPARED_DIR / 'zones.gpkg').set_index('zone_id')
regions = gpd.read_file(PREPARED_DIR / 'regions.gpkg').set_index('region_id')
zones['region_id'] = zones.index.map(lambda zid: h3.cell_to_parent(zid, 6))
total_pop = float(zones['population'].sum())

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
# Everything that *doesn't* depend on the three calibrated parameters
# goes here, computed once. The expensive bits — routes, costs — are
# reused across all parameter scans.
#
# Mirrors `road_stress.ipynb` sections 2–4. Could be factored into a
# shared helper if duplication grows; for now it's cheaper to copy.

# %%
HWY_SPEEDS = {
    'motorway': 120, 'motorway_link': 80,
    'trunk': 100,    'trunk_link': 60,
    'primary': 80,   'primary_link': 50,
    'secondary': 50, 'secondary_link': 40,
    'tertiary': 50,  'tertiary_link': 40,
    'unclassified': 50, 'residential': 30,
    'living_street': 20, 'service': 30,
    'road': 30, 'busway': 30,
}
ox.add_edge_speeds(car_graph, hwy_speeds=HWY_SPEEDS)

node_xy = gpd.GeoDataFrame(
    {'node_id': list(car_graph.nodes)},
    geometry=[Point(float(car_graph.nodes[n]['x']),
                    float(car_graph.nodes[n]['y']))
              for n in car_graph.nodes],
    crs=CRS_METRIC,
).set_index('node_id')
raw_per_m2 = geo_processing.aggregate_within_radius(
    targets=node_xy, sources=cells, radius=1000.0,
    weight_column='pop_plus_emp', return_density=True,
)
node_density = np.sqrt(raw_per_m2 * 100.0)
for nid in car_graph.nodes:
    car_graph.nodes[nid]['density_norm'] = float(node_density.loc[nid])

KMH_TO_MS = 1.0 / 3.6
INITIAL_MULT = {'density_norm': -0.45}
INITIAL_ADD = {'is_degree_4': 2.6, 'is_traffic_signal': 4.4}
for u, v, k, d in car_graph.edges(keys=True, data=True):
    u_a, v_a = car_graph.nodes[u], car_graph.nodes[v]
    for f in ('density_norm', 'is_degree_4', 'is_traffic_signal'):
        d[f] = 0.5 * (float(u_a.get(f, 0.0)) + float(v_a.get(f, 0.0)))
    base = float(d['length']) / (float(d['speed_kph']) * KMH_TO_MS)
    mult_term = base * sum(c * d[f] for f, c in INITIAL_MULT.items())
    add_term = sum(c * d[f] for f, c in INITIAL_ADD.items())
    d['duration_initial'] = max(base + mult_term + add_term, base * 0.2)

# OD pairs (zones as cells, regions as zones — see road_stress.ipynb).
R_ZONE_PAIR = 20_000.0
R_REGION_PAIR = 200_000.0
for layer in (zones, regions):
    centroids = layer.copy()
    centroids['geometry'] = centroids.geometry.centroid
    nid, _ = network_processing.snap_to_network_nodes(centroids, car_graph)
    layer['node_id'] = nid

zones_as_cells = zones.copy()
zones_as_cells.index.name = 'cell_id'
zones_as_cells['zone_id'] = zones_as_cells['region_id']

pairs = od_pairs.get_pairs(
    zones_as_cells, r_cells=R_ZONE_PAIR, node_column='node_id',
    zones=regions, r_zones=R_REGION_PAIR,
)
costs = routing.tiered_path_costs(pairs, car_graph, weight='duration_initial')

zones_as_cells['_orig_weight'] = (zones_as_cells['population']
                                  + zones_as_cells['employment_total'])
zones_as_cells['_dest_weight'] = zones_as_cells['employment_total']
regions['_dest_weight'] = (
    zones.groupby('region_id')['employment_total'].sum()
    .reindex(regions.index, fill_value=0.0))

orig_weights = od_pairs.node_values(
    '_orig_weight', list(pairs.cells_to_cells.keys()),
    zones_as_cells, 'node_id')
dest_weights = od_pairs.dest_values(
    '_dest_weight', pairs, zones_as_cells, 'node_id', regions)
cell_to_zone_node = od_pairs.build_cell_to_zone_node_map(
    zones_as_cells, regions, 'node_id')

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
R2_WEIGHT = 2.0  # combined-loss weight: R² gets 2× the weight of slope.

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
    edge_bc = network_processing.get_nested_edge_betweenness_using_igraph(
        car_graph, nested_sample, directed=True, weights='duration_initial')
    aadt_scale = (total_pop * trips_per_person_per_day) / (n_orig * n_dest)
    return edge_bc * aadt_scale


def optimal_scale(eval_result: dict) -> float:
    """Scaling factor that minimises RMSE between `k·modeled` and observed.

    Closed-form solution to `argmin_k Σ(k·m - o)²` = `Σ(o·m) / Σ(m²)`.
    Used both to make the scan loss scale-invariant (see `combined_loss`)
    and to derive the final `trips_per_person_per_day` in section 8.
    """
    m = eval_result['merged']
    o = m['observed'].to_numpy()
    mod = m['modeled'].to_numpy()
    denom = float((mod ** 2).sum())
    return float((o * mod).sum() / denom) if denom > 0 else 1.0


def best_nrmse(eval_result: dict) -> float:
    """NRMSE achievable by optimally rescaling modeled to match observed.

    `NRMSE_min = RMSE(k*·m, o) / mean(o)` where `k*` is from
    `optimal_scale`. This factors out the absolute-scale degree of
    freedom (which `trips_per_person_per_day` covers) so scans isolate
    the contribution of distribution shape.
    """
    m = eval_result['merged']
    o = m['observed'].to_numpy()
    mod = m['modeled'].to_numpy()
    k = optimal_scale(eval_result)
    rmse_min = float(np.sqrt(((k * mod - o) ** 2).mean()))
    obs_mean = float(o.mean())
    return rmse_min / obs_mean if obs_mean > 0 else np.inf


def combined_loss(eval_result: dict, r2_weight: float = R2_WEIGHT) -> float:
    """Scale-invariant combined error metric for picking the best (shape, scale).

    Returns `(1 - R²) · r2_weight + NRMSE_min`, where `NRMSE_min` is the
    NRMSE achievable after optimally rescaling modeled to match observed
    (see `best_nrmse`). Lower is better.

    Why scale-invariant. NRMSE on the raw modeled flows would conflate
    two things: distribution-shape goodness, and whether the current
    `trips_per_person_per_day` happens to be near-optimal for that
    distribution. A shape with the right distribution shape but
    "wrong" optimal trips would lose unfairly. Factoring out the
    optimal rescaling isolates the distribution-fit signal.

    Why combine R² and NRMSE_min rather than either alone:

    1. **R² is scale- and shape-invariant** beyond linear relationships.
       Two distributions can be perfectly correlated (R² = 1) yet have
       very different shapes (e.g. one with the right rank order but
       compressed dynamic range).
    2. **NRMSE_min captures shape under optimal rescaling** — including
       systematic bias, dynamic-range mismatch, and outliers. The
       squared-error weighting makes it dominated by high-AADT edges,
       which is what you want for traffic calibration (getting busy
       roads right matters more than getting cul-de-sacs right).

    Both terms typically land in 0.05-0.5 near the optimum, so the 2:1
    R² weight biases toward correlation without ignoring shape.
    """
    r2 = eval_result['r2']
    if np.isnan(r2) or eval_result['n_matched'] == 0:
        return np.inf
    return (1.0 - r2) * r2_weight + best_nrmse(eval_result)


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
# ## 6. Scan the lognormal **shape** (σ)
#
# Holds `scale` fixed at the prior; sweeps `shape` over a grid centred
# on the prior value.
#
# **Selection metric: combined loss** = `(1 − R²)·R2_WEIGHT + NRMSE_min`,
# minimised, where `NRMSE_min` is the NRMSE achievable after optimally
# rescaling modeled to match observed (see `best_nrmse`). This makes
# the loss **scale-invariant** — what we're comparing across shapes
# is "how good is the distribution shape, ignoring absolute volume",
# not "how good is it at this particular trips=1.5". Trips gets
# derived once at the end (§8).

# %%
shape_grid = np.linspace(0.5 * shape0, 1.5 * shape0, 7)
shape_results = []
for s in shape_grid:
    flows = simulate_flows(s, scale0, TRIPS_PER_PERSON_PER_DAY_INIT)
    ev = calibration.evaluate_against_counters(flows, counters)
    loss = combined_loss(ev)
    nrmse_min = best_nrmse(ev)
    k_opt = optimal_scale(ev)
    shape_results.append({'shape': s, 'r2': ev['r2'], 'slope': ev['slope'],
                          'rmse': ev['rmse'], 'nrmse_min': nrmse_min,
                          'k_opt': k_opt, 'loss': loss})
    print(f"  shape={s:.3f}: R²={ev['r2']:.4f}, slope={ev['slope']:.3f}, "
          f"NRMSE_min={nrmse_min:.3f}, k*={k_opt:.3f}, loss={loss:.4f}")
shape_df = pd.DataFrame(shape_results)

best_shape_idx = shape_df['loss'].idxmin()
shape_best = float(shape_df.loc[best_shape_idx, 'shape'])
print(f"\nBest shape (by combined loss): {shape_best:.3f} "
      f"(loss={shape_df['loss'].min():.4f}, "
      f"R²={shape_df.loc[best_shape_idx, 'r2']:.4f}, "
      f"NRMSE_min={shape_df.loc[best_shape_idx, 'nrmse_min']:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(shape_df['shape'], shape_df['r2'], 'o-', color='tab:blue', label='R²')
axes[0].plot(shape_df['shape'], shape_df['nrmse_min'], 's-', color='tab:orange',
             label='NRMSE_min')
axes[0].axvline(shape0, color='gray', linestyle='--', label=f'prior ({shape0:.3f})')
axes[0].axvline(shape_best, color='tab:red', label=f'best ({shape_best:.3f})')
axes[0].set_xlabel('lognormal shape (σ)'); axes[0].set_ylabel('R² / NRMSE_min')
axes[0].set_title('1D scan — components'); axes[0].legend()
axes[1].plot(shape_df['shape'], shape_df['loss'], 'o-', color='tab:green')
axes[1].axvline(shape0, color='gray', linestyle='--')
axes[1].axvline(shape_best, color='tab:red')
axes[1].set_xlabel('lognormal shape (σ)'); axes[1].set_ylabel('combined loss')
axes[1].set_title(f'(1 − R²)·{R2_WEIGHT} + NRMSE_min')
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 7. Scan the lognormal **scale** (μ)
#
# Uses the best `shape` from §6. Same idea — sweep, plot, pick.

# %%
scale_grid = np.linspace(0.5 * scale0, 1.5 * scale0, 7)
scale_results = []
for sc in scale_grid:
    flows = simulate_flows(shape_best, sc, TRIPS_PER_PERSON_PER_DAY_INIT)
    ev = calibration.evaluate_against_counters(flows, counters)
    loss = combined_loss(ev)
    nrmse_min = best_nrmse(ev)
    k_opt = optimal_scale(ev)
    scale_results.append({'scale': sc, 'r2': ev['r2'], 'slope': ev['slope'],
                          'rmse': ev['rmse'], 'nrmse_min': nrmse_min,
                          'k_opt': k_opt, 'loss': loss})
    print(f"  scale={sc:.1f}: R²={ev['r2']:.4f}, slope={ev['slope']:.3f}, "
          f"NRMSE_min={nrmse_min:.3f}, k*={k_opt:.3f}, loss={loss:.4f}")
scale_df = pd.DataFrame(scale_results)

best_scale_idx = scale_df['loss'].idxmin()
scale_best = float(scale_df.loc[best_scale_idx, 'scale'])
print(f"\nBest scale (by combined loss): {scale_best:.1f} "
      f"(loss={scale_df['loss'].min():.4f}, "
      f"R²={scale_df.loc[best_scale_idx, 'r2']:.4f}, "
      f"NRMSE_min={scale_df.loc[best_scale_idx, 'nrmse_min']:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(scale_df['scale'], scale_df['r2'], 'o-', color='tab:blue', label='R²')
axes[0].plot(scale_df['scale'], scale_df['nrmse_min'], 's-', color='tab:orange',
             label='NRMSE_min')
axes[0].axvline(scale0, color='gray', linestyle='--', label=f'prior ({scale0:.1f})')
axes[0].axvline(scale_best, color='tab:red', label=f'best ({scale_best:.1f})')
axes[0].set_xlabel('lognormal scale (μ)'); axes[0].set_ylabel('R² / NRMSE_min')
axes[0].set_title('1D scan — components'); axes[0].legend()
axes[1].plot(scale_df['scale'], scale_df['loss'], 'o-', color='tab:green')
axes[1].axvline(scale0, color='gray', linestyle='--')
axes[1].axvline(scale_best, color='tab:red')
axes[1].set_xlabel('lognormal scale (μ)'); axes[1].set_ylabel('combined loss')
axes[1].set_title(f'(1 − R²)·{R2_WEIGHT} + NRMSE_min')
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 8. Derive `trips_per_person_per_day` — two options
#
# The shape/scale scans factored out absolute volume (via
# `NRMSE_min`'s optimal-rescaling), so we still need to pick a
# `trips_per_person_per_day`. Two principled choices:
#
# - **Min RMSE.** Pick `trips` to minimise squared error between
#   modeled and observed. The mathematically optimal predictor.
#   Closed form: `k* = Σ(o·m) / Σ(m²)`,
#   `trips_minRMSE = trips_init × k*`.
#
# - **Slope = 1.** Pick `trips` so the no-intercept regression
#   slope equals 1 — i.e. modeled has zero average bias weighted by
#   observed magnitude. `k_slope1 = Σ(o²) / Σ(o·m) = 1 / slope`,
#   `trips_slope1 = trips_init × k_slope1`.
#
# These coincide only when correlation is perfect (R² = 1). At
# imperfect fit, **min-RMSE rescales *down* relative to slope=1**
# (intuition: shrinking modeled toward 0 trades a systematic
# undershoot for shorter total errors on big edges). The choice
# depends on what you want:
#
# - For **prediction** (downstream models consume road_stress as
#   their best estimate of AADT): use min-RMSE.
# - For an **unbiased** scaling (modeled and observed totals match
#   in a weighted sense): use slope=1.
#
# We compute both and let you pick.

# %%
flows_final = simulate_flows(shape_best, scale_best, TRIPS_PER_PERSON_PER_DAY_INIT)
eval_final = calibration.evaluate_against_counters(flows_final, counters)
k_min_rmse = optimal_scale(eval_final)
k_slope_1 = 1.0 / eval_final['slope'] if eval_final['slope'] > 0 else np.nan
trips_min_rmse = TRIPS_PER_PERSON_PER_DAY_INIT * k_min_rmse
trips_slope_1 = TRIPS_PER_PERSON_PER_DAY_INIT * k_slope_1

# RMSE under each option (rescaling modeled by k, recomputing RMSE).
m = eval_final['merged']
o = m['observed'].to_numpy()
mod = m['modeled'].to_numpy()
rmse_at = lambda k: float(np.sqrt(((k * mod - o) ** 2).mean()))
print(f"With best shape={shape_best:.3f}, scale={scale_best:.1f}:")
print(f"  R²    = {eval_final['r2']:.4f}")
print(f"  slope = {eval_final['slope']:.3f}    (regression slope at trips=1.5)")
print()
print(f"  Min RMSE:  k*={k_min_rmse:.3f}  → "
      f"trips_per_person_per_day = {trips_min_rmse:.3f}, "
      f"RMSE = {rmse_at(k_min_rmse):.0f}")
print(f"  Slope=1:   k= {k_slope_1:.3f}  → "
      f"trips_per_person_per_day = {trips_slope_1:.3f}, "
      f"RMSE = {rmse_at(k_slope_1):.0f}")

# Pick which scaling to use for the production road_stress run below.
# Default to min-RMSE (best for prediction). Flip USE_SLOPE_1 if you
# want the unbiased-totals scaling instead.
USE_SLOPE_1 = False
trips_final = trips_slope_1 if USE_SLOPE_1 else trips_min_rmse
print(f"\nUsing trips_per_person_per_day = {trips_final:.3f} "
      f"({'slope=1' if USE_SLOPE_1 else 'min-RMSE'}) for the production run.")


# %% [markdown]
# ## 9. Production run + save `road_stress.csv`
#
# Re-runs `simulate_flows` with the calibrated `(shape, scale, trips)`
# to produce the final per-edge AADT. Attaches as `road_stress` edge
# attribute (needed for the map below) and writes
# `data/prepared/road_stress.csv` for downstream notebooks
# (calibration of edge weights, etc.).

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
# ## 10. Visualise — raw flows vs capacity-normalised stress
#
# Two side-by-side maps, both cropped to 90 % of the destination
# polygon's bounding box. Left: raw `road_stress` (veh/day). Right:
# `(road_stress / capacity)²` — the BPR-style feature the edge-weight
# calibration regresses on, where capacity = `capacity_per_lane[highway]
# · lanes_per_direction`.
#
# **Why two maps.** The raw map foregrounds motorways because they carry
# the most traffic in absolute terms — but a 4-lane motorway at 50k AADT
# has more spare capacity than a 2-lane primary at 20k AADT. Normalising
# reveals "edges actually pushed close to capacity", which is what should
# slow them down in the routing model.
#
# Edge widths follow the standard highway-tier hierarchy
# (motorway/trunk thickest, residential thinnest) so the road class
# stays readable independent of the colour scale. Edges are drawn in
# order of ascending `HIGHWAY_RANKS`, so motorways / trunks land *on
# top* of the residential mesh at junctions. Colour scales clip at
# P99 (extreme bottlenecks compress the rest of the distribution).

# %%
import _figures as figures   # noqa: E402  — project-local plot helpers

# Per-edge capacity + (V/C)². `lanes_per_direction` comes from
# `consolidate_intersections`; fall back to 1.0 for graphmls saved
# before that wiring landed.
vc_sq: dict = {}
edge_cap: dict = {}
for u, v, k, d in car_graph.edges(keys=True, data=True):
    aadt = float(d.get('road_stress', 0.0))
    lpd = float(d.get('lanes_per_direction', 1.0))
    cap = figures.CAPACITY_PER_LANE.get(
        figures.edge_highway(d), figures.DEFAULT_CAPACITY) * lpd
    edge_cap[(u, v, k)] = cap
    vc = aadt / cap if cap > 0 else 0.0
    vc_sq[(u, v, k)] = vc ** 2

stress_arr = np.array([float(d.get('road_stress', 0.0))
                       for _, _, d in car_graph.edges(data=True)])
vc_sq_arr = np.array(list(vc_sq.values()))
xlim, ylim = figures.crop_to_polygon(dest_polygon)

fig, axes = plt.subplots(1, 2, figsize=(22, 11))
figures.plot_network_map(
    axes[0], car_graph, road_stress,
    cbar_label='road_stress (veh/day, clipped at P99)',
    title=(f'Raw flow — road_stress '
           f'(median {np.median(stress_arr):.0f}, '
           f'P99 {np.quantile(stress_arr, 0.99):.0f}, '
           f'max {stress_arr.max():.0f} veh/day)'),
    xlim=xlim, ylim=ylim,
)
figures.plot_network_map(
    axes[1], car_graph, vc_sq,
    cbar_label='(V/C)² (clipped at P99)',
    title=(f'Capacity-normalised — (V/C)² '
           f'(median {np.median(vc_sq_arr):.3f}, '
           f'P99 {np.quantile(vc_sq_arr, 0.99):.3f}, '
           f'max {vc_sq_arr.max():.3f})'),
    xlim=xlim, ylim=ylim,
)
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 11. Investigate the (V/C)² long tail
#
# A few outliers in a feature that enters the edge-weight calibration
# OLS as a multiplier on baseline time can drag the fitted coefficient
# and skew the calibrated edge weights. Inspect the top-N edges to
# decide whether to leave them (true bottlenecks, OLS will downweight),
# cap the feature at a sane V/C, or revisit the capacity table / lanes
# parsing.

# %%
vc_lookup = {key: np.sqrt(v) for key, v in vc_sq.items()}

print(f"Edges with V/C > 0.5: "
      f"{sum(1 for v in vc_lookup.values() if v > 0.5):,} / {len(vc_lookup):,}")
print(f"Edges with V/C > 1.0: "
      f"{sum(1 for v in vc_lookup.values() if v > 1.0):,} / {len(vc_lookup):,}")
print(f"Edges with V/C > 1.5: "
      f"{sum(1 for v in vc_lookup.values() if v > 1.5):,} / {len(vc_lookup):,}")

over_by_tier: dict[str, int] = {}
for (u, v, k), vc in vc_lookup.items():
    if vc > 1.0:
        d = car_graph[u][v][k]
        tier = figures.edge_highway(d) or 'unknown'
        over_by_tier[tier] = over_by_tier.get(tier, 0) + 1
print("\nEdges with V/C > 1, by highway tier:")
for tier, n in sorted(over_by_tier.items(), key=lambda x: -x[1]):
    print(f"  {tier:20s} {n:5d}")

top = sorted(vc_lookup.items(), key=lambda x: -x[1])[:20]
rows = []
for (u, v, k), vc in top:
    d = car_graph[u][v][k]
    rows.append({
        'u': u, 'v': v, 'k': k,
        'highway': figures.edge_highway(d),
        'lanes': d.get('lanes'),
        'lanes_per_direction': d.get('lanes_per_direction', 1.0),
        'road_stress': float(d.get('road_stress', 0.0)),
        'capacity': edge_cap[(u, v, k)],
        'V/C': vc,
        'length_m': float(d.get('length', 0.0)),
    })
top_df = pd.DataFrame(rows)
print("\nTop-20 edges by V/C:")
print(top_df.to_string(index=False,
                       formatters={'V/C': '{:.3f}'.format,
                                   'road_stress': '{:.0f}'.format,
                                   'capacity': '{:.0f}'.format,
                                   'lanes_per_direction': '{:.1f}'.format,
                                   'length_m': '{:.0f}'.format}))


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
#   a pipeline. Production *does* wire road_stress into the edge-weight
#   calibration as a `(V/C)²` BPR-style feature; see
#   [`projects/lumos/`](https://github.com/mmiotti/aperta-lab/tree/main/src/projects/lumos).
# - **Single-pass parameter selection.** No coordinate-descent scan
#   over the lognormal cost-decay shape / scale, no inner-vs-outer
#   counter filter, no min-RMSE-vs-slope-1 trade-off comparison.
#   Production tunes those carefully — the showcase shows the
#   one-parameter-set version to keep runtime reasonable.
# - **(V/C)² capacity normalisation lives in `calibrate_edge_weights.ipynb`
#   if at all** — it's a downstream consumption of the `road_stress`
#   estimate, not part of the estimation itself.
#
# For an example of these pieces wired into a full production stack
# with `aperta_lab` scaffolding (scenario configs, typed I/O, dependency
# tracking), see
# [`aperta-lab/src/projects/lumos/`](https://github.com/mmiotti/aperta-lab/tree/main/src/projects/lumos).
