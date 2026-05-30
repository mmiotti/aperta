# Changelog

All notable changes to aperta are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com); versions follow
[Semantic Versioning](https://semver.org). Until v1.0, minor breaking
changes may occur in 0.x releases.

## [Unreleased]

## [0.1.0a0] — 2026-05-30

Initial public release, alongside the toolkit-paper submission.

### Architecture
- Three architectural primitives: path-first routing, tiered origin–destination
  structure, orthogonal mode–network separation.
- Six-phase analysis workflow (load → map → OD pairs → flows → costs → accessibility),
  composable from notebooks or scripts.

### Modules
- `od_pairs` — tiered OD-pair primitives (`TieredODNodePairs`,
  `TieredODGeoPairs`); `get_pairs`, `reindex_by_geo_unit`,
  `aggregate_across_modes`, `dest_values_geo`.
- `routing` — `tiered_path_costs`, `tiered_path_aggregate` (Dijkstra on any
  `networkx` graph via `scipy.sparse.csgraph`, with per-edge / per-node
  feature aggregation), `floor_intrazonal_costs`.
- `accessibility` — `cumulative_opportunities`, `gravity`, `nearest_k`;
  `Bin`, `Decay` named-tuple specs.
- `utility` — `Utility` linear spec, `route_utility`, `add_endpoint_utility`;
  used for utility-based generalized travel costs and logsum accessibility.
- `overhead` — per-cell first/last-mile overheads (node-keyed and geo-keyed).
- `traffic_flows` — demand-weighted nested-node sampling for traffic-volume
  estimation via modified betweenness.
- `calibration` — iterative OLS edge-weight calibration against traffic
  counters and travel-survey times.
- `network_processing` — OSMnx wrappers preserving intersection-attribute
  tags through consolidation; node snapping; sparse Brandes-style betweenness.
- `geo_processing`, `geo_mapping` — H3 grids, KDTree buffers, raster sampling,
  point/polygon/line spatial joins.
- `visualization` — choropleth panels, per-edge `LineCollection` rendering,
  basemap-aware figures.
- `osm_helpers`, `topography` — optional OSM and DEM ingestion utilities.
- `errors` — aperta-specific exception types.

### Distribution
- MIT-licensed.
- Published on PyPI.
- 13 test modules, ~320 test methods; CI on Python 3.11–3.13.
- Sphinx documentation hosted on ReadTheDocs.

[Unreleased]: https://github.com/mmiotti/aperta/compare/v0.1.0-alpha...HEAD
[0.1.0a0]: https://github.com/mmiotti/aperta/releases/tag/v0.1.0-alpha
