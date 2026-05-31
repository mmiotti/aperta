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

The `osm_helpers` and `topography` modules import their backing libraries
(`osmnx`, `rasterio`, `requests`) lazily — install the matching `[osm]` /
`[topo]` extra if you use them, otherwise an `ImportError` surfaces at first use.

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
2. **Map data to units** — aggregate source data into the `cells → zones` hierarchy; snap geo units to network nodes.
3. **Build sparse OD pairs** — the tiered OD structure (three distance tiers) with per-cell origins at near range and zone-aggregated destinations at far range, keeping per-origin compute bounded independently of network extent.
4. **(Optional) Estimate traffic flows** — sampled betweenness centrality; optionally calibrate against observed counter data.
5. **Estimate travel costs** — shortest paths on the routing graph plus per-cell trip overheads. Optionally: utility-based generalized costs and edge-weight calibration against observed travel times.
6. **Calculate accessibilities** — cumulative-opportunity, gravity, nearest-k, logsum (and cross-modal aggregation across per-mode results).

See the [Modules](#modules) table below for which module covers each phase, and the [API reference](https://aperta.readthedocs.io/) for the specific functions.

Runnable examples, in increasing depth:

- [examples/minimal/accessibility.ipynb](examples/minimal/accessibility.ipynb) — what aperta does in ~50 lines using only OpenStreetMap. Cambridge MA, ~10 s.
- [examples/walkthrough/accessibility.ipynb](examples/walkthrough/accessibility.ipynb) — guided tour of every primitive; walking + cycling, cross-modal logsum, path-first per-edge feature aggregation. Central Paris, ~1 min end-to-end.
- [examples/extended/](examples/extended/) — production-scale Bern + 40 km: prep pipeline, calibration against observed travel times, traffic-flow estimation, accessibility analysis. ~30 min.

The toy-world end-to-end test in [tests/test_workflow.py](tests/test_workflow.py) doubles as the smallest possible walk-through (~150 lines, runs in a second).

## Quick example

The three-line core of an accessibility analysis: build the tiered OD pairs, route shortest paths, count opportunities within a travel-time budget.

```python
from aperta import accessibility, od_pairs, routing

pairs = od_pairs.get_pairs(cells, r_cells=2000.0, node_column='node_id')
times = routing.tiered_path_costs(pairs, graph, weight='walk_time_s')
acc   = accessibility.cumulative_opportunities(
    times, {'supermarkets': weights}, {},
    [accessibility.Bin('15min', 0, 15 * 60)],
)
```

A complete, runnable version (OSM ingestion, plotting): [`examples/minimal/accessibility.ipynb`](examples/minimal/accessibility.ipynb).

## Modules

See the [API reference](https://aperta.readthedocs.io/en/latest/api/) for module-by-module documentation.

## Design

What aperta is:

- **Path-first.** Every routing call returns the realized route alongside the OD travel cost as a single primitive — so any per-edge or per-node attribute (gradient, perceived safety, surface type, air-pollution exposure, road stress, ...) can be aggregated along each route in the same pass. This is the architectural prerequisite for utility-based travel costs, joint accessibility-and-exposure assessment, route-aware infrastructure-quality metrics, and any other analysis that depends on what happens *along* the route, not just at its endpoints.
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

Ultimate speed for the full accessibility stack was not aperta's goal. Nonetheless, aperta typically runs within 1–5× of Pandana on equivalent cumulative-opportunity workloads, especially when the area of interest (for which to calculate accessibilities) is substantially smaller than the buffer zone (destinations to consider). Aperta can be much faster than Pandana when aiming to recalculate accessibilities for a select subset of locations after a graph topology or edge weight change. See the [benchmark](https://aperta.readthedocs.io/en/latest/benchmark/) for the full setup and numbers, or run [`examples/extended/benchmark.py`](examples/extended/benchmark.py) to reproduce.

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
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## License

MIT. See [LICENSE](LICENSE).
