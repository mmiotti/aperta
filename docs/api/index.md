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
| `overhead` | First/last-mile overheads on cost ODMs (node-keyed and geo-keyed), with helpers for zone-tier last-mile aggregation. |
| `traffic_flows` | Traffic-volume estimation via cost-weighted nested-node sampling. |
| `calibration` | OLS calibration of per-edge weights against observed travel times; bearing-aware traffic-counter snapping + counter-fit evaluation for flow calibration. |
| `network_processing` | Network helpers — intersection consolidation, sampled edge betweenness, node snapping, per-direction lane counts. |
| `geo_processing` | Geometry helpers — H3 grids, line bearings, KDTree-based buffer aggregations, raster sampling. |
| `geo_mapping` | Spatial-join wrappers between points, polygons, and filtered lines. |
| `osm_helpers` | OSM data fetching + per-edge categorization via `osmnx`. Requires `aperta[osm]`. |
| `topography` | Copernicus GLO-30 DEM download + raster sampling. Requires `aperta[topo]`. |
| `visualization` | Plot helpers — choropleth panels, multi-panel comparisons, per-edge `LineCollection` rendering, styled colorbars. |
| `errors` | aperta-specific exception types. |

```{toctree}
:caption: Core OD + routing
:maxdepth: 1

od_pairs
routing
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
osm_helpers
topography
```

```{toctree}
:caption: Misc
:maxdepth: 1

visualization
errors
```
