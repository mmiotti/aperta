# aperta

[![tests](https://github.com/mmiotti/aperta/actions/workflows/test.yml/badge.svg)](https://github.com/mmiotti/aperta/actions/workflows/test.yml)

A Python toolkit for **accessibility analysis on multimodal transport networks** — routing, distance/time computation, and gravity-/utility-/logsum-based accessibility metrics on `networkx` / `igraph` graphs.

The name is Latin/Italian for *open*.

## Status

**Pre-1.0, alpha.** Published alongside a toolkit paper (in submission). APIs may change without notice until v1.0.

aperta the library lives here. The sibling [`aperta-lab`](https://github.com/mmiotti/aperta-lab) repo holds (a) an opinionated project-scaffolding package (`aperta_lab` — filesystem layout, typed I/O, per-scenario coefficient tables, optional dependency tracking) and (b) the concrete projects built on it, most prominently the Swiss "Urban Mobility Atlas".

The boundary rule for what belongs in aperta: *if the code could run 1:1 on a different country's data, it goes here. If it knows the name of a specific input file or schema, it belongs in an application repo built on top.*

## Install

```bash
pip install aperta              # algorithms only
pip install 'aperta[osm]'       # + OSM ingestion (osmnx)
pip install 'aperta[examples]'  # + everything needed to run the example notebooks
```

Requires Python ≥ 3.11.

For development:

```bash
git clone git@github.com:mmiotti/aperta.git
cd aperta
pip install -e ".[osm,topo,h3]"
python -m unittest discover -s tests -t .
```

> If you plan to edit the example notebooks under `examples/`, run the
> [notebook editing setup](#editing-notebooks) once after cloning. Not
> needed if you're only using the library or modifying Python source.

## Workflow

aperta is organised around a six-phase workflow. Every module slots into one of these phases.

1. **Load and prepare data** — networks (per mode), land-use rasters and points.
2. **Map data to units + compute shared features** — build the `cells → zones → regions` aggregation hierarchy via `geo_mapping`; snap geo units to network nodes via `network_processing.snap_to_network_nodes` / `assign_to_eligible_centroid`; sample topography rasters (`topography.fetch_copernicus_dem` + `geo_processing.sample_raster_at_points`); compute per-node density via `geo_processing.aggregate_within_radius`.
3. **Build sparse OD pairs** — `od_pairs.get_pairs` returns a `TieredODNodePairs` (cell / zone / region tiers, node-keyed). Lift to `TieredODGeoPairs` (cell/zone/region-keyed) via `od_pairs.reindex_by_geo_unit` for cross-modal alignment.
4. **Estimate traffic flows** — `traffic_flows.nested_node_sample` + betweenness via `network_processing.get_*_betweenness*`. Optional calibration against observed counters via `calibration.snap_counters_to_edges` + `calibration.evaluate_against_counters`.
5. **Estimate travel costs** — `routing.tiered_path_costs` / `routing.tiered_path_aggregate` (Dijkstra on any networkx graph) + `overhead.add_node_overheads` / `add_geo_overheads` / `add_origin_cell_overhead`. Optional calibration of per-edge weights against observed travel times via `calibration.calibrate_edge_weights`. Plus `utility.route_utility` / `add_endpoint_utility` for utility-based costs.
6. **Calculate accessibilities** — `accessibility.count_in_bins`, `accessibility.gravity`, `accessibility.nearest_k`. Cross-modal: `od_pairs.aggregate_across_modes` on per-mode `TieredODGeoPairs`, then any accessibility primitive on the combined ODM.

### Where to see each phase in the examples

The extended example splits its prep across five notebooks and its analysis across three; each analysis notebook stands alone (no cross-dependencies). The minimal example does the whole workflow in one notebook.

| Phase | In `examples/extended/` |
|---|---|
| 1. Load and prepare data | [prep/1_download.ipynb](examples/extended/prepare/1_download.ipynb), [prep/2_dasymetric_employment.ipynb](examples/extended/prepare/2_dasymetric_employment.ipynb) |
| 2. Map data to units + features | [prep/3_unit_mapping.ipynb](examples/extended/prepare/3_unit_mapping.ipynb), [prep/4_topography.ipynb](examples/extended/prepare/4_topography.ipynb), [prep/5_density.ipynb](examples/extended/prepare/5_density.ipynb) |
| 3. Build sparse OD pairs | first cells of [accessibility.ipynb](examples/extended/accessibility.ipynb) and [road_stress.ipynb](examples/extended/road_stress.ipynb) — each builds for its own use |
| 4. Estimate traffic flows | [road_stress.ipynb](examples/extended/road_stress.ipynb) |
| 5. Estimate travel costs | [calibrate_edge_weights.ipynb](examples/extended/calibrate_edge_weights.ipynb) (calibrates the model); each analysis notebook applies edge times inline |
| 6. Calculate accessibilities | [accessibility.ipynb](examples/extended/accessibility.ipynb) |

For the *full* workflow in one notebook, see [examples/minimal/accessibility.ipynb](examples/minimal/accessibility.ipynb) — single-mode, ~10-minute end-to-end Cambridge example.

The toy-world end-to-end test in [tests/test_workflow.py](tests/test_workflow.py) doubles as the smallest possible walk-through (~150 lines, runs in a second).

## Quick example

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

## Modules

| Module | Purpose |
|---|---|
| `od_pairs` | Tiered OD pair structures (`TieredODNodePairs`, `TieredODGeoPairs`) + builders (`get_pairs`, `dest_values`, `reindex_by_geo_unit`, `make_mask`, `aggregate_across_modes` for cross-modal alignment). |
| `routing` | Shortest paths on `networkx` / `igraph` graphs. Edge-weighting helpers, single-source / one-to-one primitives, tiered OD routing (`tiered_path_costs`, `tiered_path_aggregate` with per-edge feature aggregation), intrazonal-cost flooring. |
| `accessibility` | `count_in_bins` (cumulative), `gravity` (decay-based), `nearest_k` (cost to nearest k). Outputs per-node or per-cell depending on input ODM class. |
| `utility` | Linear utility specs (`Utility`, `RouteFeature`) and pipeline (`route_utility`, `add_endpoint_utility`) for utility-based costs; consumed by `accessibility.gravity` with an exp decay for logsum accessibility. |
| `overhead` | First/last-mile overheads on cost ODMs. `add_node_overheads` (node-keyed); `add_geo_overheads` / `add_origin_cell_overhead` (geo-keyed); `aggregate_dest_overhead_per_*` helpers for zone/region-tier last-mile. |
| `traffic_flows` | Traffic-volume estimation via cost-weighted nested-node sampling (`nested_node_sample`). |
| `calibration` | OLS calibration of per-edge weights (`calibrate_edge_weights`) against observed point-to-point travel times; bearing-aware traffic-counter snapping (`snap_counters_to_edges`) + counter-fit evaluation (`evaluate_against_counters`) for traffic-flow calibration. |
| `network_processing` | Network helpers — `networkx ↔ igraph` bridging, `consolidate_intersections` (OSMnx-output cleanup with obstacle re-attachment), betweenness, `snap_to_network_nodes`, `assign_to_eligible_centroid`, `aggregate_edges_to_nodes`, `lanes_per_direction`. |
| `geo_processing` | Geometry helpers — hectare and H3 grids, bearings, `aggregate_within_radius` (cross-set buffer aggregation via scipy KDTree), `custom_spatial_lag`. |
| `geo_mapping` | Spatial-join wrappers — `map_points_to_polygons`, `map_polygons_to_points`, `map_points_to_points`, `map_points_to_filtered_lines`. |
| `geo_units` | Registry of the 5 canonical aperta units (`cells`, `zones`, `regions`, `nodes`, `edges`) and their `id_col` conventions. |
| `osm_helpers` | OSM data fetching + per-edge categorisation via `osmnx` (`fetch_network`, `fetch_pois`, `categorize_edges`). Requires `aperta[osm]`. |
| `topography` | Copernicus GLO-30 DEM download + raster sampling (`fetch_copernicus_dem`). Requires `aperta[topo]`. |
| `visualization` | Plot helpers — `plot_cell_values` (single-panel choropleth), `plot_cell_values_comparison` (multi-panel with shared scale), `plot_tiered_destinations` (origin-cell tier viz), `plot_edge_values` (LineCollection-based with sort/z-order control), `add_styled_colorbar`. |
| `table_processing` | DataFrame helpers — column-aware aggregation, upcasting, integer-column restoration, metric discovery. |
| `utils` | `timeit` decorator, weighted-aggregation helpers, statsmodels result formatting. |
| `errors` | `DataError`, `ProcessingError`. |

## Design

aperta is intentionally lightweight and **agnostic about how you organise your data and pipeline**:

- **No filesystem assumptions.** Algorithm functions take plain `networkx` graphs, `pandas` / `geopandas` frames, and `numpy` arrays. They don't read or write files. The opinionated I/O + project scaffolding layer lives in the sibling [`aperta-lab`](https://github.com/mmiotti/aperta-lab) repo (`aperta_lab` package).
- **No DAG engine.** No caching, no dependency graph, no orchestration. For full DAG features layer [DVC](https://dvc.org/) or [Snakemake](https://snakemake.readthedocs.io/) on top.
- **Routing profiles are plain Python functions, not config files.** Other routing tools define modes (car, bike, pedestrian) via custom formats — Lua profiles in OSRM, JSON costing-options in Valhalla, YAML in GraphHopper. aperta skips that layer: a profile is a Python callable that returns an edge cost. Trade-off: no shared library of pre-built profiles to pick from, but full Python expressivity (call numpy, look up calibrated coefficients, branch on anything).
- **No global state** — every function takes its inputs explicitly.

## Contributing

Internal repository for now. External contributions will open after the toolkit-paper publication.

## Editing notebooks

Notebook outputs (matplotlib figures, dataframe HTML, etc.) bloat git
quickly — a single Bern map can be 2 MB of embedded PNGs. This repo
uses [`nbstripout`](https://github.com/kynan/nbstripout) as a clean
filter to strip output cells on commit. The filter is declared in
[`.gitattributes`](.gitattributes); to activate it in your local clone,
run once after cloning:

```bash
pip install nbstripout
nbstripout --install
```

The filter then runs transparently on every commit — `git diff` and
`git log` on .ipynb files show only code + markdown changes.

Notebooks here are jupytext-paired: each `.ipynb` has a `.py` shadow
under the same basename. Edits in either form propagate to the other
via `jupytext --sync <file>`. Only the .py shadow needs to be human-
readable in code review; the .ipynb travels for the GitHub-rendered
view.

If you skip the `nbstripout --install` step, your commits will include
output cells (large diffs, slow GitHub renders). Just install it.

## License

MIT. See [LICENSE](LICENSE).
