# API reference

Aperta is organized as a flat collection of algorithm modules. Each module
focuses on one phase of the six-phase accessibility workflow and operates
on plain `numpy` / `pandas` / `networkx` inputs without any framework
scaffold.

## At a glance

| Module | Purpose |
|---|---|
| `od_pairs` | Tiered OD pair structures + builders, including cross-modal alignment. |
| `routing` | Shortest paths on `networkx` graphs via `scipy.sparse.csgraph.dijkstra`: tiered OD routing, per-edge / per-node feature aggregation along realized paths, intrazonal-cost flooring. |
| `accessibility` | Cumulative-opportunity, gravity, nearest-k metrics. Per-node or per-cell output depending on input ODM class. |
| `utility` | Linear utility specifications and the routing-+-endpoint pipeline for utility-based costs; consumed by `gravity` with an exp decay for logsum accessibility. |
| `overhead` | First/last-mile overheads on geo-keyed cost ODMs, with helpers for zone-tier last-mile aggregation. |
| `traffic_flows` | Traffic-volume estimation via cost-weighted nested-node sampling. |
| `calibration` | OLS calibration of per-edge weights against observed travel times; bearing-aware traffic-counter snapping + counter-fit evaluation for flow calibration. |
| `network_processing` | Network helpers — sampled edge betweenness, node/edge attribute aggregation, feature snapping. |
| `network_snap` | Snap-target resolution — `snap_to_network_nodes` (point-to-node match, optionally two-tier with a priority set), `insert_projected_nodes` (virtual-node insertion). |
| `routing_prep` | Mode-aware graph preparation — `prepare_network`, `compute_snap_eligibility`, `PreparedGraph`. |
| `geo_processing` | Geometry helpers — H3 grids, line bearings, KDTree-based buffer aggregations, raster sampling, Copernicus GLO-30 DEM download (`fetch_copernicus_dem` requires `aperta[topo]`). |
| `geo_mapping` | Spatial-join wrappers between points, polygons, and filtered lines. |
| `data_processing` | Tabular helpers — dedup indices after a spatial join, add straight-line distance, weighted mean per group. |
| `visualization` | Plot helpers — choropleth panels, multi-panel comparisons, per-edge `LineCollection` rendering, styled colorbars. |
| `errors` | aperta-specific exception types. |

```{toctree}
:caption: Core OD + routing
:maxdepth: 1

od_pairs
routing
routing_prep
overhead
```

```{toctree}
:caption: Accessibility metrics
:maxdepth: 1

accessibility
utility
```

```{toctree}
:caption: Traffic flows + calibration
:maxdepth: 1

traffic_flows
calibration
```

```{toctree}
:caption: Geographic + network helpers
:maxdepth: 1

geo_processing
geo_mapping
network_processing
network_snap
data_processing
```

```{toctree}
:caption: Misc
:maxdepth: 1

visualization
errors
```
