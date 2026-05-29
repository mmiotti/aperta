# aperta

[![tests](https://github.com/mmiotti/aperta/actions/workflows/test.yml/badge.svg)](https://github.com/mmiotti/aperta/actions/workflows/test.yml)
[![docs](https://readthedocs.org/projects/aperta/badge/?version=latest)](https://aperta.readthedocs.io/en/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/github/license/mmiotti/aperta.svg)](LICENSE)

A Python toolkit for **cross-modal accessibility analysis on transport networks** — routing, distance/time computation, utility-based travel costs, and gravity- and logsum-based accessibility metrics on `networkx` graphs (routed via `scipy.sparse.csgraph`).

The name is Latin/Italian for *open* — the condition that accessibility, at root, measures.

## Status

**Pre-1.0, alpha.** Published alongside a toolkit paper (in submission). APIs may change without notice until v1.0.

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

aperta is organised around a six-phase workflow. Phases 4 and 5's calibration sub-step are optional; the rest is the minimum end-to-end pipeline.

1. **Load and prepare data** — networks (one per mode), land use, topography, optional ground-truth data (traffic counters, travel-survey times).
2. **Map data to units** — aggregate source data into the `cells → zones` hierarchy; snap geo units to network nodes.
3. **Build sparse OD pairs** — the tiered OD structure (three distance tiers) with per-cell origins at near range and zone-aggregated destinations at far range, keeping per-origin compute bounded independently of network extent.
4. **(Optional) Estimate traffic flows** — sampled betweenness centrality; optionally calibrate against observed counter data.
5. **Estimate travel costs** — shortest paths on the routing graph plus per-cell trip overheads. Optionally: utility-based generalised costs and edge-weight calibration against observed travel times.
6. **Calculate accessibilities** — cumulative-opportunity, gravity, nearest-k, logsum (and cross-modal aggregation across per-mode results).

See the [Modules](#modules) table below for which module covers each phase, and the [API reference](https://aperta.readthedocs.io/) for the specific functions.

Runnable examples, in increasing depth:

- [examples/minimal/accessibility.ipynb](examples/minimal/accessibility.ipynb) — what aperta does in ~40 lines using only OpenStreetMap. Cambridge MA, ~10 s.
- [examples/walkthrough/accessibility.ipynb](examples/walkthrough/accessibility.ipynb) — guided tour of every primitive; walking + cycling, cross-modal logsum, path-first per-edge feature aggregation. Central Paris, ~1 min end-to-end.
- [examples/extended/](examples/extended/) — production-scale Bern + 25 km: prep pipeline, calibration against observed travel times, traffic-flow estimation, accessibility analysis. ~30 min.

The toy-world end-to-end test in [tests/test_workflow.py](tests/test_workflow.py) doubles as the smallest possible walk-through (~150 lines, runs in a second).

## Quick example

A walking-accessibility map in ~30 lines using only OSM. One aperta call per phase:

```python
import geopandas as gpd
import osmnx as ox
from aperta import (accessibility, geo_mapping, geo_processing,
                    network_processing, od_pairs, routing)

PLACE = 'Cambridge, Massachusetts, USA'

# 1. AOI + walking network
boundary = ox.geocode_to_gdf(PLACE)
crs = boundary.estimate_utm_crs()
graph = ox.project_graph(
    ox.graph_from_place(PLACE, network_type='walk'), to_crs=crs,
).to_undirected()
for _u, _v, _k, data in graph.edges(keys=True, data=True):
    data['walk_time_s'] = data['length'] / 1.4  # 1.4 m/s walking

# 2. H3 cells (origins) + supermarkets (destinations)
cells = geo_processing.build_h3_grid(boundary.geometry.iloc[0], 10,
                                     polygon_crs='EPSG:4326', target_crs=crs)
centroids = gpd.GeoDataFrame(geometry=cells.geometry.centroid, index=cells.index, crs=crs)
cells['node_id'], _ = network_processing.snap_to_network_nodes(centroids, graph)

sm = ox.features_from_place(PLACE, tags={'shop': 'supermarket'}).to_crs(crs)
supermarkets = gpd.GeoDataFrame(geometry=sm.geometry.centroid.values, crs=crs)
supermarkets['cell_id'], _ = geo_mapping.map_points_to_polygons(
    supermarkets, cells, allow_nearest=True)
cells['supermarkets'] = (supermarkets.groupby('cell_id').size()
                         .reindex(cells.index, fill_value=0).astype(float))

# 3. Tiered OD pairs + routing
pairs = od_pairs.get_pairs(cells, r_cells=2000.0, node_column='node_id')
times = routing.tiered_path_costs(pairs, graph, weight='walk_time_s')

# 4. Accessibility — supermarkets reachable within 15 min walk per origin cell.
sm_weights = od_pairs.dest_values('supermarkets', pairs, cells, node_column='node_id')
acc = accessibility.cumulative_opportunities(
    times, {'supermarkets': sm_weights}, {},
    [accessibility.Bin('15min', 0, 15 * 60)],
)
```

Runnable end-to-end version (with plotting): [examples/minimal/accessibility.ipynb](examples/minimal/accessibility.ipynb).

## Modules

| Module | Purpose |
|---|---|
| `od_pairs` | Tiered OD pair structures (`TieredODNodePairs`, `TieredODGeoPairs`) + builders (`get_pairs`, `dest_values`, `reindex_by_geo_unit`, `make_mask`, `aggregate_across_modes` for cross-modal alignment). |
| `routing` | Shortest paths on `networkx` graphs via `scipy.sparse.csgraph.dijkstra`. Edge-weighting helpers, single-source / one-to-one primitives, tiered OD routing (`tiered_path_costs`, `tiered_path_aggregate` with per-edge `PathAggregation` and per-node `NodeAggregation` feature aggregation along realised paths), pure path-walker primitive `aggregate_along_paths` (for prebuilt path lists), intrazonal-cost flooring. |
| `accessibility` | `cumulative_opportunities` (cumulative), `gravity` (decay-based), `nearest_k` (cost to nearest k). Outputs per-node or per-cell depending on input ODM class. |
| `utility` | Linear utility specs (`Utility`, `RouteFeature`) and pipeline (`route_utility`, `add_endpoint_utility`) for utility-based costs; consumed by `accessibility.gravity` with an exp decay for logsum accessibility. |
| `overhead` | First/last-mile overheads on cost ODMs. `add_node_overheads` (node-keyed); `add_geo_overheads` / `add_origin_cell_overhead` (geo-keyed); `aggregate_dest_overhead_per_*` helpers for zone-tier last-mile. |
| `traffic_flows` | Traffic-volume estimation via cost-weighted nested-node sampling (`nested_node_sample`). |
| `calibration` | OLS calibration of per-edge weights (`calibrate_edge_weights`) against observed point-to-point travel times; bearing-aware traffic-counter snapping (`snap_counters_to_edges`) + counter-fit evaluation (`evaluate_against_counters`) for traffic-flow calibration. |
| `network_processing` | Network helpers — `consolidate_intersections` (OSMnx-output cleanup with obstacle re-attachment), `get_nested_edge_betweenness` (sampled edge-usage counts from a `nested_node_sample`), `snap_to_network_nodes`, `assign_to_eligible_centroid`, `aggregate_edges_to_nodes`, `lanes_per_direction`. |
| `geo_processing` | Geometry helpers — H3 grids, line bearings, `sum_within_radius` (same-set neighbourhood sum) and `cross_sum_within_radius` (cross-set buffer aggregation) via scipy KDTree, raster sampling. |
| `geo_mapping` | Spatial-join wrappers — `map_points_to_polygons`, `map_polygons_to_points`, `map_points_to_points`, `map_points_to_filtered_lines`. |
| `osm_helpers` | OSM data fetching + per-edge categorisation via `osmnx` (`fetch_network`, `fetch_pois`, `categorize_edges`). Requires `aperta[osm]`. |
| `topography` | Copernicus GLO-30 DEM download + raster sampling (`fetch_copernicus_dem`). Requires `aperta[topo]`. |
| `visualization` | Plot helpers — `plot_cell_values` (single-panel choropleth), `plot_cell_values_comparison` (multi-panel with shared scale), `plot_tiered_destinations` (origin-cell tier viz), `plot_edge_values` (LineCollection-based with sort/z-order control), `add_styled_colorbar`. |
| `errors` | Aperta-specific exception types (`ContextError`, `DataError`, `ProcessingError`). |

## Design

What aperta is:

- **Path-first.** Every routing call returns the realised route alongside the OD travel cost as a single primitive — so any per-edge or per-node attribute (gradient, perceived safety, surface type, air-pollution exposure, road stress, ...) can be aggregated along each route in the same pass. This is the architectural prerequisite for utility-based travel costs, joint accessibility-and-exposure assessment, route-aware infrastructure-quality metrics, and any other analysis that depends on what happens *along* the route, not just at its endpoints.
- **Cross-modal.** Mode and network are orthogonal: one network per mode, where "mode" generalises to any independently-varying network — walking vs cycling vs driving, but also day-time vs night-time street access, congested vs free-flow edge weights, with vs without a proposed bike-lane scenario. Cross-mode aggregation (`min`, `logsum`) over per-network cost ODMs is a first-class operation. Logsum aggregation closes the utility loop — discrete-choice-consistent accessibility across modes from per-mode utilities.
- **Multi-scale by construction.** The tiered cells / zones / three-distance-tier OD structure bounds per-origin computation independently of the network's geographic extent. Country-scale reach without country-scale destination counts; intermediate cost matrices stay small enough to persist to disk and share.
- **Live-graph routing.** Shortest paths run on the graph directly via `scipy.sparse.csgraph.dijkstra` — no precomputed routing index. Per-query routing is slower than contraction-hierarchy-based tools (OSRM, Pandana/pandarm), but edge-weight changes are immediate, which is what makes iterative calibration, traffic-flow estimation, and scenario comparison practical. Edge weights are written by plain Python callables; no Lua / YAML / JSON profile format to learn.

What aperta is not:

- **No filesystem assumptions.** Algorithm functions take plain `networkx` graphs, `pandas` / `geopandas` frames, and `numpy` arrays. They don't read or write files.
- **No DAG engine, no global state.** No caching, no dependency tracking, no orchestration. Every function takes its inputs explicitly. For DAG features, layer [DVC](https://dvc.org/) or [Snakemake](https://snakemake.readthedocs.io/) on top.

## Interoperability with other accessibility tools

Aperta deliberately doesn't try to do everything in-house. Two interoperability patterns are worth flagging:

- **Public transit via R5.** Aperta has no native public-transit support right now (no GTFS reader, no RAPTOR-style time-dependent routing). Anything that can be expressed as a `networkx` graph with appropriate edge weights — including simplified transit-as-graph models — will route in aperta like any other network. For full GTFS-based transit routing (calendars, transfers, frequency-based services), the pragmatic pattern is to compute the transit OD cost matrix with [R5](https://github.com/conveyal/r5) (via [r5py](https://r5py.readthedocs.io/)), align its origins/destinations to the same cell layer aperta uses, and feed the resulting per-mode cost ODM into `od_pairs.aggregate_across_modes` alongside the walk / cycle / car ODMs computed by aperta. The cross-modal aggregation proceeds identically whether each per-mode ODM came from aperta's router or elsewhere.
- **Faster cost-only routing via Pandana/pandarm.** Aperta's live-graph routing is the right trade-off for path-first, iterative, and scenario-comparative workloads, but for one-shot cost-only accessibility on a large fixed network, contraction-hierarchy backends like [Pandana](https://udst.github.io/pandana/) (and its recent modernized fork pandarm) route faster per query. The calibrated edge weights produced by `calibration.calibrate_edge_weights` are plain per-edge attributes on the `networkx` graph and transfer cleanly to a Pandana/pandarm network built from the same OSM extract — i.e., you can calibrate edge weights in aperta and then route with them in Pandana/pandarm.

## Benchmark vs Pandana

Cumulative-opportunity accessibility to total employment on the consolidated walk and car networks of Bern + 25 km (the same area the extended example notebooks build). End-to-end wall time, including each library's setup phase (Pandana network construction + `precompute`; aperta OD-pair construction + routing + accessibility); lower is better.

| Setup                                                       | Walk (15 min) |  Car (30 min) |
|-------------------------------------------------------------|--------------:|--------------:|
| Pandana — all graph nodes                                   |           …s  |           …s  |
| Aperta A — all graph nodes (single-tier, Euclidean cutoff)  |           …s  |           …s  |
| Aperta B — cell-snap origins, tiered destinations           |           …s  |           …s  |
| Aperta C — AOI-restricted cell origins, tiered destinations |           …s  |           …s  |

Three things this is meant to show:

- **Pandana wins on A** (its design centre: one-shot all-pairs cost on a fixed graph via contraction hierarchies).
- **B narrows the gap** by leveraging the 3-tier destination structure (cell-tier close, zone-tier far) — the tiers replace what would otherwise be a wall of redundant intra-zone routing.
- **C overtakes Pandana** in the realistic production setup (an AOI buffer provides destinations and through-routing but is not itself an origin), where Aperta routes only from the cells that need accessibility values and Pandana still pays its full all-pairs precompute.

Reproduce: run [examples/extended/prepare/](examples/extended/prepare/) end to end (one-time, slow — downloads + consolidates OSM networks), then `python examples/extended/benchmark.py`.

## Engineering

- **CI**: full test suite runs on Python 3.11 / 3.12 / 3.13 (every push + PR), alongside [Ruff](https://github.com/astral-sh/ruff) (lint + format) and mypy (type checking).
- **Tests**: ~310 test methods across 13 files. The end-to-end integration test ([tests/test_workflow.py](tests/test_workflow.py)) doubles as a runnable minimal example (~150 lines).
- **API**: small, parsimonious surface; functions take the minimum arguments needed with sensible defaults. Most of the codebase is type-annotated.
- **Layout**: `src/` layout prevents accidental local-source imports and catches "works on my machine" bugs early.
- **License**: MIT.
- **Dependencies**: core install pulls in only the standard scientific-Python stack; domain extras (`aperta[osm]`, `aperta[topo]`, `aperta[h3]`) keep the core lightweight.
- **Docs**: Sphinx-based API reference hosted on [ReadTheDocs](https://aperta.readthedocs.io/).

## Contributing

Internal repository for now. External contributions will open after the toolkit-paper publication.

## Editing notebooks

Notebooks here are **jupytext-paired**: each `.ipynb` has a `.py` shadow
under the same basename. Edits in either form propagate to the other via
`jupytext --sync <file>`. The `.py` shadow is the human-readable form for
code review; the `.ipynb` carries the executed outputs.

Two pieces of automation handle the friction. Both need to be activated
once per local clone:

```bash
# 1. Auto-sync .py <-> .ipynb on every commit (jupytext via pre-commit).
pip install pre-commit
pre-commit install

# 2. Strip outputs from non-example notebooks on commit
#    (keeps git history clean; example notebooks are exempt and ship
#    with their outputs intact so figures render on GitHub).
pip install nbstripout
nbstripout --install
```

After this setup:

- Edit either the `.py` or the `.ipynb` — the pre-commit hook keeps the
  pair in sync (if your commit changes only one side, the hook updates
  the other and asks you to re-stage).
- `git diff` and `git log` on non-example `.ipynb` files show only code
  and markdown changes (outputs stripped by `nbstripout`).
- Notebooks under [`examples/`](examples/) are exempt from output-stripping
  via [`.gitattributes`](.gitattributes) and ship with their executed
  outputs intact. Before committing changes to an example notebook,
  execute it end-to-end (in VSCode / Jupyter, or via
  `jupytext --sync --execute <file>`) so the committed outputs reflect
  the current code.

If you skip the `nbstripout --install` step, your commits to non-example
notebooks will include output cells (large diffs, slow GitHub renders).
If you skip the `pre-commit install` step, you'll need to run
`jupytext --sync` manually after notebook edits so the `.py` and `.ipynb`
don't drift apart. Both are one-time setups; install them.

## Acknowledgments

Aperta was developed at the [Chair of Ecological Systems Design](https://esd.ifu.ethz.ch/) at [ETH Zurich](https://ethz.ch) in the context of the [BlueCity](https://www.epfl.ch/schools/enac/blue-city-project/) project and [LUMOS](https://csfm.ethz.ch/en/research/projects/lumos.html).

## License

MIT. See [LICENSE](LICENSE).
