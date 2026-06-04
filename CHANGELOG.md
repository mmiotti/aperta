# Changelog

All notable changes to aperta are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com); versions follow
[Semantic Versioning](https://semver.org). Until v1.0, minor breaking
changes may occur in 0.x releases.

## [Unreleased]

## [0.2.0a0] — 2026-06-03

Two methodologically substantial changes:

- The snap subsystem is refactored into two single-purpose functions
  with clean separation between graph mutation and point-to-node
  matching. Bern-scale unit-mapping snap is ~6× faster end-to-end.
- Traffic-flow sampling drops the lognormal parametric assumption
  about trip-time shape in favor of a non-parametric percentile-
  binned target. Sampling weights now follow whatever shape the
  calibration data actually shows, with higher R² against observed
  flows and lower sensitivity to input data and random state.

### Added
- `network_snap.insert_projected_nodes` — graph-enrichment function that
  inserts virtual nodes onto eligible edges where points would otherwise
  have no snap target. Replaces the mutation half of the removed
  `snap_to_network_tiered`. Single straight pass via an in-memory
  child-lineage walk (no batched-rebuild loop). Composable: callable
  multiple times on the same graph with different `edge_filter` values
  to accumulate virtuals for different purposes (cells, zones, POIs).
- `network_snap.nodes_incident_to_edges` — helper to derive a node set
  from an edge-tag predicate. Typical use: build the priority-node set
  for `snap_to_network_nodes`.
- `routing_prep.compute_snap_eligibility` — prep-stage entry point that
  writes per-edge `cost_excluded_<mode>` and returns the snap-eligible
  node set + cost-mask flag name. Decoupled from the consumer-stage
  `prepare_network` so the prep workflow can hand outputs to
  `insert_projected_nodes` / `snap_to_network_nodes` without
  over-promising a full `PreparedGraph`.
- `snap_to_network_nodes` two-tier behavior: optional
  `priority_node_ids` / `priority_node_flag` + `priority_node_radius`
  for "prefer main-road nodes" semantics. Tier-1 priority then
  tier-2 eligible-node fallback within `max_radius`.
- `prepare_network` writes per-node `is_snap_eligible_<mode>` and
  per-edge `cost_excluded_<mode>` decorations that survive graphml
  round-trip, eliminating trapped-node snap regressions on reloaded
  graphs.
- `is_virtual=1` decoration on every inserted synthetic node;
  `load_consolidated_graphml`'s prefix-scan dtype handling preserves
  the int through graphml round-trip.

### Changed
- `traffic_flows`: weighted node sampling now targets a **non-parametric
  percentile-binned trip-time distribution** instead of a parametric
  lognormal fit (`LOGNORM_SHAPE` / `LOGNORM_SCALE` removed). Eliminates
  the a priori assumption about the shape of the travel-time
  distribution that drives the weighted betweenness; sampling
  weights now follow whatever empirical shape the calibration data
  shows. User-verified on Swiss traffic-counter data: higher R²
  against observed flows, lower sensitivity to both input data and
  random state.
- Split `aperta.network_processing` into three focused modules
  (back-compat re-exports preserved):
  - `aperta.network_processing` — graph-processing primitives
    (consolidation, intersection topology, OSM classification, lanes,
    GraphML round-trip).
  - `aperta.network_snap` — snap-target resolution
    (`snap_to_network_nodes`, `insert_projected_nodes`,
    `nodes_incident_to_edges`, `transport_centroid`, edge-split
    primitives).
  - `aperta.routing_prep` — mode-aware preparation
    (`prepare_network`, `compute_snap_eligibility`, `PreparedGraph`,
    `MODE_DEFAULTS`).
- `snap_to_network_nodes`: parameter `max_distance` → `max_radius`
  (terminology consistency with the rest of the snap API).
- The insert-time spacing knob is now `node_spacing` (per-edge minimum
  spacing between graph nodes along an edge), separated cleanly from
  the snap-time radii.
- Example pipeline ([examples/extended/prepare/](examples/extended/prepare/)):
  step 1 outputs `<mode>_consolidated.graphml`; step 3 reads
  `_consolidated.graphml` and writes `<mode>_graph.graphml`
  (snap-mutated). Step 3 is now an explicit
  `insert_projected_nodes` → `snap_to_network_nodes` two-phase
  pattern. Adds a Bern crop snap-visualization appendix
  (original vs virtual nodes per network).

### Removed
- `network_snap.snap_to_network_tiered` (replaced by the
  `insert_projected_nodes` + `snap_to_network_nodes` two-function
  pattern).
- `snap_to_network_nodes`: `max_distance` parameter (renamed to
  `max_radius`, no deprecation alias kept).
- `routing_prep.MODE_PRIORITY_DEFAULTS` and the related node-priority
  predicate machinery (`_priority_walk_bike`, `_priority_car`).
  Priority is now expressed via `priority_node_ids` at snap call time.

### Fixed
- Multi-zone snap-node double-counting in tiered OD reindexing.
- Self-loop edges (`u == v`) in snap insertion: `_find_reverse_edge_key`
  returns None so split falls back to single-direction split, avoiding
  the double-split corruption that previously crashed
  `split_two_way_edge_at_point`.
- Mirror-asymmetric forward/reverse OSM geometries in two-way edge
  insertion: endpoint-projection check considers both directions, so
  cells projecting near a reverse-edge endpoint fall back to endpoint
  snap instead of raising on degenerate split.

### Performance
- Bern-scale unit-mapping snap pipeline (142k cells × 3 networks +
  8.9k zones × 3 networks): **~47 min → ~8 min (~6× speedup)**.
  Concentrated in the insert step via the single-pass lineage walk
  replacing the old per-cell STRtree rebuild and the dropped tier-1
  KDTree machinery.

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

[Unreleased]: https://github.com/mmiotti/aperta/compare/v0.2.0-alpha...HEAD
[0.2.0a0]: https://github.com/mmiotti/aperta/compare/v0.1.0-alpha...v0.2.0-alpha
[0.1.0a0]: https://github.com/mmiotti/aperta/releases/tag/v0.1.0-alpha
