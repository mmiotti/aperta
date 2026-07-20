# aperta

[![tests](https://github.com/mmiotti/aperta/actions/workflows/test.yml/badge.svg)](https://github.com/mmiotti/aperta/actions/workflows/test.yml)
[![docs](https://readthedocs.org/projects/aperta/badge/?version=latest)](https://aperta.readthedocs.io/en/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Python toolkit for **cross-modal accessibility analysis on transport networks** — routing, distance/time computation, utility-based travel costs, and gravity- and logsum-based accessibility metrics on `networkx` graphs (routed via `scipy.sparse.csgraph`).

![Three families of aperta capabilities, illustrated on the Bern region: network preparation (estimated traffic volumes and calibrated edge speeds), path feature collection (bike-comfort scores along realized routes and aggregated per origin cell), and accessibility analysis (time-based access to hiking opportunities and cross-modal utility-based access to groceries).](docs/assets/hero.jpg)

The name is Latin/Italian for *open* — the condition that accessibility, at root, measures.

## Status

**Pre-1.0, alpha.** Published alongside a toolkit paper (in submission). APIs may change without notice until v1.0.

## Companion project: aperta-atlas

`aperta` is the algorithm library — the right entry point if you're writing your own accessibility pipeline. **[`aperta-atlas`](https://github.com/mmiotti/aperta-atlas)** is a sibling repo built on `aperta`: a multi-modal accessibility pipeline, fully implemented for Switzerland and transferrable to any other location, plus reusable scaffolding (`aperta_atlas`: context, typed I/O, coefficient system) for building your own atlas-style project. Start there if you're looking for pre-computed accessibility outputs, or if you'd rather adapt an existing full pipeline than write one from scratch.

## Install

```bash
pip install aperta              # algorithms only
pip install 'aperta[osm]'       # + OSM ingestion (osmnx)
pip install 'aperta[examples]'  # + everything needed to run the example notebooks
```

Requires Python ≥ 3.11.

Optional backing libraries are imported lazily: `osmnx` (used by
`osm_helpers`), `rasterio` + `requests` (used by DEM fetch and raster
sampling in `geo_processing`). Install the matching `[osm]` / `[topo]`
extra if you use those features, otherwise an `ImportError` surfaces at
first use.

For development:

```bash
git clone git@github.com:mmiotti/aperta.git
cd aperta
pip install -e ".[osm,topo,h3]"
python -m unittest discover -s tests -t .
```

> If you plan to edit the example notebooks under `examples/`, run the
> [jupytext + nbstripout setup](CONTRIBUTING.md#editing-notebooks) once
> after cloning. Not needed if you're only using the library or
> modifying Python source.

## Workflow

Aperta is organized around a six-phase workflow. Phases 4 and 5's calibration sub-step are optional; the rest is the minimum end-to-end pipeline.

1. **Load and prepare data** — networks (one per mode), land use, topography, optional ground-truth data (traffic counters, travel-survey times).
2. **Map data to units** — aggregate source data into the `cells → zones` hierarchy; snap geo units to network nodes. Snapping is two complementary functions: `insert_projected_nodes` optionally enriches the graph by inserting virtual nodes onto road segments where points would otherwise have no graph node within snap distance (with optional filtering, e.g. main roads only); `snap_to_network_nodes` then does the actual point-to-node match (optionally two-tier with a priority node set for "prefer main-road nodes" semantics).
3. **Build sparse OD pairs** — the tiered OD structure with per-cell origins at near range and zone-aggregated destinations at far range, keeping per-origin compute bounded independently of network extent.
4. **(Optional) Estimate traffic flows** — sampled betweenness centrality (essentially a network-based "2.5-step" travel demand model); optionally calibrate against observed traffic counter data.
5. **Estimate travel costs** — shortest paths on the routing graph. Three optional features: (a) trip overheads for parking search, unlocking a bicycle, etc (usually estimated through correlation with urban characteristics such as density); (b) utility-based generalized costs and (c) edge-weight calibration against observed travel times.
6. **Calculate accessibilities** — cumulative-opportunity, gravity, nearest-k, logsum (and cross-modal aggregation across per-mode results).

See the [API reference](https://aperta.readthedocs.io/en/latest/api/) for which module covers each phase and for the specific functions.

Runnable examples, in increasing depth:

- [examples/minimal/accessibility.ipynb](examples/minimal/accessibility.ipynb) — what aperta does in ~50 lines using only OpenStreetMap. Cambridge MA, ~10 s.
- [examples/walkthrough/accessibility.ipynb](examples/walkthrough/accessibility.ipynb) — guided tour of every primitive; walking + cycling, cross-modal logsum, path-first per-edge feature aggregation. Central Paris, ~1 min end-to-end.
- [examples/calibration/](examples/calibration/) — two standalone calibration demos on production-scale Bern + 40 km: edge-weight calibration against observed travel times, traffic-flow tuning against travel-survey + counter data. Reads pre-prepared inputs from [aperta-atlas](https://github.com/mmiotti/aperta-atlas)'s `bern-public` scenario (set `APERTA_DATA_ROOT`). Each notebook is a standalone showcase — they do NOT chain into a pipeline; for the full integrated calibrate → flows → accessibility chain see aperta-atlas. ~30 min per notebook.
- [examples/benchmarks/](examples/benchmarks/) — aperta vs pandana scaling benchmark on the same Bern + 40 km dataset. Documents the headline scaling numbers in this README.

The toy-world end-to-end test in [tests/test_workflow.py](tests/test_workflow.py) doubles as the smallest possible walk-through (~150 lines, runs in a second).

## Quick example

The three-line core of an accessibility analysis: build the tiered OD pairs, route shortest paths, count opportunities within a travel-time budget.

```python
from aperta import accessibility, od_pairs, routing

pairs = od_pairs.get_pairs(cells, r_cells=2000.0)
times = routing.tiered_path_costs(graph, pairs, weight='walk_time_s')
acc   = accessibility.cumulative_opportunities(
    times, {'supermarkets': weights}, {},
    [accessibility.Bin('15min', 0, 15 * 60)],
)
```

A complete, runnable version (OSM ingestion, plotting): [`examples/minimal/accessibility.ipynb`](examples/minimal/accessibility.ipynb).

## Modules

See the [API reference](https://aperta.readthedocs.io/en/latest/api/) for module-by-module documentation.

## Distinctive features

Two features are unusually first-class in aperta compared to other accessibility libraries in this space:

- **Edge-weight calibration to observed travel times.** Real per-mode speeds (car, bike, walk) vary systematically with road class, density, gradient, intersection topology, and — for car — congestion. `calibration.calibrate_edge_weights` fits a per-edge-attribute weight model against a survey of measured trip times (origin, destination, observed duration), so downstream shortest paths reflect actual travel behavior rather than a naive speed × length. For car, a lightweight sampled-betweenness traffic-flow estimator (`traffic_flows.nested_node_sample` + `network_processing.get_nested_edge_betweenness`) closes the loop against observed traffic-counter data — congestion-aware speeds emerge from calibration without a full traffic-assignment model.
- **Trip overheads for door-to-door realism.** Snapped node-to-node travel time ignores first-mile and last-mile costs (parking search, walking to a bus stop, unlocking a bike, transit egress, etc.) — but those are exactly what push naive routing away from what people actually experience. `overhead.add_geo_overheads` bakes empirically-estimated per-cell / per-zone overheads into the OD cost matrix so accessibility metrics see door-to-door times, not just network distances.

Both integrate with aperta's tiered OD structure and run at country scale without materialising dense distance matrices.

## Design

What aperta is:

- **Path-first.** Routing returns the realized route alongside the cost as a single primitive, so per-edge attributes (gradient, exposure, surface, perceived safety) aggregate along the path natively — the architectural prerequisite for utility-based and route-aware accessibility.
- **Cross-modal.** Mode and network are orthogonal: one network per mode, with `min` / `logsum` aggregation across modes as a first-class operation. Generalizes to any axis of network variation — time-of-day, congestion regime, infrastructure scenario.
- **Multi-scale.** A tiered cell / zone OD structure bounds per-origin computation independently of network extent — country-scale reach without country-scale destination counts.
- **Live-graph routing.** Dijkstra on the graph directly, no precomputed index. Slower per query than contraction-hierarchy tools, but edge-weight changes are immediate — what makes iterative calibration and scenario comparison practical.

What aperta is not:

- **No filesystem assumptions.** Algorithm functions take plain `networkx` graphs, `pandas` / `geopandas` frames, and `numpy` arrays — no file I/O.
- **No DAG engine, no global state.** No caching, no dependency tracking, no orchestration. Every function takes its inputs explicitly. For DAG features, layer [DVC](https://dvc.org/) or [Snakemake](https://snakemake.readthedocs.io/) on top.

## Interoperability with other accessibility tools

Aperta deliberately doesn't try to do everything in-house. Two interoperability patterns are worth flagging:

- **Public transit via R5.** Aperta has no native public-transit support right now (no GTFS reader, no RAPTOR-style time-dependent routing). Anything that can be expressed as a `networkx` graph with appropriate edge weights — including simplified transit-as-graph models — will route in aperta like any other network. For full GTFS-based transit routing (calendars, transfers, frequency-based services), the pragmatic pattern is to compute the transit OD cost matrix with [R5](https://github.com/conveyal/r5) (via [r5py](https://r5py.readthedocs.io/)), align its origins/destinations to the same cell layer aperta uses, and feed the resulting per-mode cost ODM into `od_pairs.aggregate_across_modes` alongside the walk / cycle / car ODMs computed by aperta. The cross-modal aggregation proceeds identically whether each per-mode ODM came from aperta's router or elsewhere.
- **Faster cost-only routing via Pandana/pandarm.** Aperta's live-graph routing is the right trade-off for path-first, iterative, and scenario-comparative workloads, but for one-shot cost-only accessibility on a large fixed network, contraction-hierarchy backends like [Pandana](https://udst.github.io/pandana/) (and its recent modernized fork pandarm) route faster per query. The calibrated edge weights produced by `calibration.calibrate_edge_weights` are plain per-edge attributes on the `networkx` graph and transfer cleanly to a Pandana/pandarm network built from the same OSM extract — i.e., you can calibrate edge weights in aperta and then route with them in Pandana/pandarm.

## Benchmark vs Pandana

Ultimate speed for the full accessibility stack was not aperta's goal. On production-shape cumulative-opportunity workloads (AOI cells as origins, tiered destinations covering a wider buffer), aperta runs within ~2–6× of Pandana — a modest constant-factor cost for the extra capabilities aperta provides (path-first routing, cross-modal aggregation, live-graph iterative calibration). For iterative workloads (re-routing after edge-weight calibration or scenario changes) Pandana would pay its contraction-hierarchy preprocess cost on every change, whereas aperta re-routes directly on the mutated graph — a different comparison not measured by this one-shot benchmark. See the [benchmark](https://aperta.readthedocs.io/en/latest/benchmark.html) for the full setup and numbers, or run [`examples/benchmarks/benchmark.py`](examples/benchmarks/benchmark.py) to reproduce.

## Acknowledgments

Aperta was developed at the [Chair of Ecological Systems Design](https://esd.ifu.ethz.ch/) at [ETH Zurich](https://ethz.ch) in the context of the [BlueCity](https://www.epfl.ch/schools/enac/blue-city-project/) project and [LUMOS](https://csfm.ethz.ch/en/research/projects/lumos.html).

## Cite this

If you use aperta in a publication, please cite the archived release:

```bibtex
@software{miotti_aperta_2026,
  author    = {Miotti, Marco},
  title     = {{Aperta: Path-first, cross-modal accessibility analysis in Python}},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20473787},
  url       = {https://doi.org/10.5281/zenodo.20473787}
}
```

## License

MIT. See [LICENSE](LICENSE).
