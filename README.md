# aperta

[![tests](https://github.com/mmiotti/aperta/actions/workflows/test.yml/badge.svg)](https://github.com/mmiotti/aperta/actions/workflows/test.yml)

A Python toolkit for **accessibility analysis on multimodal transport networks** — measuring how open places are to their surroundings via routing, distance/time calculations, and gravity-style metrics on `networkx` / `igraph` graphs.

The name is Latin/Italian for "open."

## Status

**Pre-1.0, alpha.** Published alongside a toolkit paper (in submission). APIs may change without notice until v1.0. The boundary rule for what belongs here: *if the code could run 1:1 on a different country's data, it goes in `aperta`. If it knows the name of a specific input file or schema, it belongs in an application repo built on top* (e.g. the sibling [aperta-lab](https://github.com/mmiotti/aperta-lab) repo with the Swiss "Urban Mobility Atlas" application).

## Workflow

aperta is designed around a six-phase workflow for accessibility analysis. Each phase has clear inputs and outputs and operates on the geo_unit tiers defined in `aperta.geo_units`:

- **Aggregation** units — `cells`, `zones`, `regions` (finest → coarsest; library-canonical).
- **Network** units — `nodes`, `edges` (graph elements; library-canonical).
- **Source** units — region-specific intermediate units like `municipalities`, `cantons`, `buildings` (defined per region in `config/<region>.yml`).

### 1. Load and prepare data

- Load/obtain network(s), depending on modes of interest.
- *Optional:* convert network to dual graph (allows for turn penalties).
- Load/obtain land use data (density maps / points of interest).
- Load/obtain topographical data (if not part of network data already).
- Load/obtain other data of interest (e.g. car ownership rates by municipality).

### 2. Map data and create metrics

- Create `cells`, `zones`, and `regions` (the aggregation hierarchy).
- Map cells to zones and zones to regions.
- Map land use data (density / POIs) to cells.
- Map topographical data to network nodes.
- Map network nodes to the centroids of cells, zones, and regions.
- Map other data to the most appropriate aggregation geo unit.
- Calculate land-use metrics (e.g. population density in a 500 m radius) for each cell.
- Calculate network metrics (e.g. intersection density) for each zone.

### 3. Create sparse OD matrices

- Define asymmetrical cell-to-cell, cell-to-zone, and cell-to-region pairs.
- Based on these pairs, define node OD pairs for each network.

### 4. Estimate traffic flows

- Estimate traffic volumes using modified betweenness centrality.
- *Optional:* calibrate traffic volume estimates using traffic counter data.

### 5. Estimate travel costs

- *Optional:* calibrate network edge weights and overhead weights using travel survey data.
- Estimate travel distances and travel times for each defined OD pair, for each network.
- *Optional:* calibrate utility coefficients using travel survey data.
- Estimate utilities (generalized travel costs) for each defined OD pair.

### 6. Calculate accessibilities

- Calculate accessibility for each cell, for each desired accessibility metric.
- Map cell-based accessibilities to other units of interest (e.g. network nodes, buildings, etc.).

The reference project applying this workflow is **LUMOS Switzerland** (`src/projects/lumos/`, migration in progress); the canonical preparation pipeline that feeds it lives in `src/preparation/switzerland/historic/`. Each refactored module slots into one of the six phases.

## Install

```bash
git clone git@github.com:mmiotti/aperta.git
pip install -e ./aperta
```

(Editable install for development.) Once published to PyPI:

```bash
pip install aperta              # algorithms only
pip install 'aperta[osm]'       # + OSM ingestion (osmnx)
pip install 'aperta[examples]'  # + everything needed to run the example notebooks
```

Requires Python ≥ 3.11.

## Modules

| Module | Purpose |
|---|---|
| `aperta.context` | `Context` dataclass and `init_context()` — entry point for every script using aperta; resolves paths, loads YAML config, tracks dependencies via `status/status.json`. |
| `aperta.data` | Typed I/O helpers: `create_/get_shapes` (`.gpkg`), `create_/get_properties` (`.csv`), `create_/get_odm` (sparse origin-destination dicts as compressed `.npz`), `create_/get_nw` (`.graphml` skeleton + companions), plus generic and results helpers. |
| `aperta.od_pairs` | Tiered OD pair structures (`TieredODNodePairs`, `TieredODGeoPairs`) + builders (`get_pairs`, `dest_values`, `dest_values_geo`, `reindex_by_geo_unit`, `make_mask`, `aggregate_across_modes` for cross-modal alignment). |
| `aperta.routing` | Pure-Python routing on `networkx` / `igraph`. Edge-weighting helpers, single-source / pairwise / one-to-one shortest-path primitives, tiered OD routing (`tiered_path_costs`, `tiered_path_aggregate` with per-edge feature aggregation), intrazonal-cost flooring. |
| `aperta.overhead` | First/last-mile overheads on cost ODMs: per-cell, per-node, per-zone. `add_node_overheads` for node-keyed; `add_geo_overheads` / `add_origin_cell_overhead` for geo-keyed; `aggregate_dest_overhead_per_*` helpers for zone-/region-tier last-mile. |
| `aperta.accessibility` | `count_in_bins` (cumulative-opportunity), `gravity` (decay-based), `nearest_k` (cost to nearest k opportunities). Works on either `TieredODNodePairs` (per-node output) or `TieredODGeoPairs` (per-cell output). |
| `aperta.utility` | Linear utility specs (`Utility`, `RouteFeature`) and pipeline (`route_utility`, `add_endpoint_utility`) for utility-based costs; consumed by `accessibility.gravity` with an exp decay for logsum accessibility. |
| `aperta.traffic_flows` | Traffic-volume estimation via betweenness-with-cutoff or weighted-nested-node-sampling (`get`, `nested_node_sample`). |
| `aperta.network_processing` | Network helpers — networkx↔igraph bridging, betweenness, `snap_to_network_nodes`, `assign_to_eligible_centroid` (tier-aware zone snapping), `aggregate_edges_to_nodes`, node-feature-to-edge propagation. |
| `aperta.geo_processing` | Geometry helpers — hectare grids, bearings, `custom_spatial_lag` (same-set spatial lag via libpysal), `aggregate_within_radius` (cross-set buffer aggregation via scipy KDTree). |
| `aperta.geo_mapping` | Spatial-join wrappers — `map_points_to_polygons`, `map_polygons_to_points`, `map_points_to_points`, `map_points_to_filtered_lines`. |
| `aperta.geo_units` | Registry of the 5 canonical aperta units (`cells`, `zones`, `regions`, `nodes`, `edges`) and their `id_col` conventions. |
| `aperta.visualization` | Plot helpers — `plot_cell_values` (single-panel choropleth), `plot_cell_values_comparison` (multi-panel with shared scale), `plot_tiered_destinations` (origin-cell tier viz). |
| `aperta.params` | Per-mode coefficient tables from CSV (`find_csv`, `load`, `for_mode`). Module-level cached. |
| `aperta.table_processing` | DataFrame helpers — column-aware aggregation, upcasting, integer-column restoration, metric discovery. |
| `aperta.variant` | `Variants` for declaring per-script run-variants, with CLI dispatch via `Variants.run(main)` (selected on the command line via `--variant <name>`). |
| `aperta.pipeline` | YAML-driven pipeline runner (`python -m aperta.pipeline`). Sequential subprocess execution; partial runs via `--from`/`--only`/`--skip`. |
| `aperta.utils` | Misc utilities — `timeit` decorator, `tracked_namedtuple` (used by `Context` for parameter tracking), weighted-aggregation helpers, flatten-list helpers, statsmodels result formatting. |
| `aperta.errors` | `ContextError`, `DataError`, `ProcessingError`. |

## Minimal example

```python
import networkx as nx
from aperta import routing

# Tiny graph: A -> B -> C, plus a more expensive shortcut A -> C
g = nx.MultiDiGraph()
g.add_edge('A', 'B', length=10.0, cost=1.0)
g.add_edge('B', 'C', length=10.0, cost=1.0)
g.add_edge('A', 'C', length=30.0, cost=3.0)

# Single-source distances from A (all reachable nodes)
print(routing.shortest_distances_from(g, 'A', weight='cost'))
# -> {'A': 0, 'B': 1.0, 'C': 2.0}

# One-to-one routing for a list of trips
res = routing.shortest_path_metrics_one_to_one(
    g, trip_ids=['t1'], origins=['A'], destinations=['C'], weight='cost')
print(res)
#     distance  cost
# t1      20.0   2.0
```

## Pipeline runner

A YAML-driven runner is included:

```yaml
# pipelines/lumos.yml
name: lumos-switzerland
working_dir: ../src
stages:
  - name: prepare
    module: applications.switzerland.lumos.analysis.prepare
  - name: calculate
    module: applications.switzerland.lumos.analysis.calculate
    variants: [t3, t4]   # runs twice: once per declared variant
```

```bash
python -m aperta.pipeline run pipelines/lumos.yml
python -m aperta.pipeline run pipelines/lumos.yml --from calculate
```

## Design

aperta is intentionally **lightweight**:

- **No DAG with caching/visualization** — the included pipeline runner is dumb on purpose. For full DAG features, use [DVC](https://dvc.org/) or [Snakemake](https://snakemake.readthedocs.io/) on top.
- **No data versioning** — `aperta.data.Context` records what each script created/used and warns on staleness, but doesn't version data files.
- **Routing profiles are plain Python functions, not config files.** Other routing tools define modes (car, bike, pedestrian) via custom config formats — Lua profiles in OSRM, JSON costing-options in Valhalla, YAML rules in GraphHopper. aperta skips that layer: a profile is a Python callable that takes an edge and returns a duration or cost. See `aperta.network_processing.edge_duration_lumos` for the reference pattern. Trade-off: no shared library of pre-built profiles to pick from, but full Python expressivity (call numpy, look up calibrated coefficients, branch on anything).
- **No global state** — every function takes a `Context` (or graph, or DataFrame) explicitly.

## Contributing

Internal repository for now. External contributions will open after the toolkit-paper publication.

## License

To be confirmed.
