# Changelog

All notable changes to aperta are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com); versions follow
[Semantic Versioning](https://semver.org). Until v1.0, minor breaking
changes may occur in 0.x releases.

## [Unreleased]

_Nothing yet._

## [0.3.0a0] — 2026-07-20 — the aperta / aperta-atlas split becomes real

This cycle is dominated by one arc: **the aperta / aperta-atlas split becoming real**. [aperta-atlas](https://github.com/mmiotti/aperta-atlas) (formerly aperta-lab) is now a stable public companion repo, and with it as a home for scenario-bound, project-shaped, and OSM-tag-aware code, aperta itself can be trimmed to its algorithm-library essence.

The two roles going forward:

- **aperta** (this repo) — the algorithm library. Pure primitives on `numpy` / `pandas` / `networkx`. No filesystem, no scenarios, no OSM-tag semantics.
- **[aperta-atlas](https://github.com/mmiotti/aperta-atlas)** — an opinionated multi-modal accessibility pipeline for Switzerland (published as the "Urban Mobility Atlas") plus reusable `aperta_atlas` scaffolding (context, typed I/O, coefficient system, variant runner) for building your own atlas-style project on top of aperta.

Everything below follows from that split. Helpers moved out to aperta-atlas or got removed outright when live code migrated to better patterns; the remaining public surface got an API-consistency pass (naming, argument order, defaults, keyword-only cleanups). Two earlier sub-cycles that belong to the same arc are covered in their own sub-sections below: the sweeping OSM-boundary cleanup (2026-06-07) and the extended-example restructure (2026-06-27).

Callers upgrading from `0.2.0a0` will hit multiple breaking signature / import changes. All migrations are mechanical.

### Added
- `aperta.data_processing` — new module for tabular helpers (DataFrame /
  GeoDataFrame). Contains `remove_duplicate_indices`,
  `add_straight_line_dist`, `weighted_group_mean`. The split with
  `geo_processing` is functional: `data_processing` = tabular column
  transforms; `geo_processing` = geometry math, spatial-lag
  aggregation, raster sampling, grid building, DEM fetch.
- `aperta.data_processing.weighted_group_mean` — NaN-aware weighted
  mean per group, indexed by group ID. Used by aperta-atlas main
  pipeline for reducing per-cell values to per-zone summaries
  (public-transit access, OD-time aggregation).
- `aperta.geo_processing.fetch_copernicus_dem` — Copernicus GLO-30 DEM
  downloader (tiles from AWS Open Data → mosaic → clip → optional
  reproject). Moved in from the retired `aperta.topography` module.
- `aperta.geo_processing.count_within_radius` — count features within
  a radius of each point. Companion to `sum_within_radius` /
  `cross_sum_within_radius`; same scipy-KDTree backend, returns counts
  rather than weighted sums.
- `aperta.geo_processing.simplify_geometry` and
  `aperta.geo_processing.get_hectare_geometries` — pure geometry
  helpers moved in from earlier scaffolding.
- `aperta.calibration.apply_edge_durations` — promoted from private
  helper to public. Writes the per-edge duration formula
  `α · base + base · Σ multipliers + Σ additives` to a graph in place,
  with `[min_speed_kph, max_speed_kph]` clamping. Callers can apply
  pre-fitted coefficients to a fresh graph without re-running the OLS —
  the production pattern in aperta-atlas.
- `aperta.routing.shortest_path_costs_one_to_one` — fast cost-only
  per-trip helper. Same routing engine as
  `shortest_path_metrics_one_to_one`, but skips the path walk /
  per-edge feature aggregation. For cases where only the scalar cost
  is needed (e.g. scatter-plotting predicted vs observed times).
- `aperta.network_processing.attach_node_properties` /
  `attach_edge_properties` — public helpers for layering a CSV-typed
  property DataFrame onto a graph (clean alternative to graphml dtype
  coercion, which now lives in `aperta_atlas.osm`).
- `aperta.network_processing.parse_edge_id` — parse `'u:v:k'` →
  `(u, v, k)` tuple. Lightweight utility used by callers that store
  edge IDs as strings.
- `aperta.network_processing.verify_odm_against_network` — cheap
  pre-flight check that every (origin, destination) pair in an OD
  matrix has endpoints present in the graph's node set.
- `aperta.network_processing.smooth_node_attribute` —
  distance-weighted smoothing of a per-node attribute over the network.
- `aperta.routing_prep.compute_snap_eligible_nodes` — standalone
  helper returning the largest-connected-component's node set for a
  given mode. Same logic `prepare_network` runs internally, but
  exposed for callers that want the eligibility set without building
  a full `PreparedGraph`.

### Renamed
- `od_pairs.dest_values` → `od_pairs.lookup_dest_column_node`.
  `od_pairs.dest_values_geo` → `od_pairs.lookup_dest_column_geo`.
  Both are renamed for descriptiveness (verb-forward "look up a column
  at destinations") and parallel naming (the `_node`/`_geo` suffix
  pairing signals the two are peers, not a base + variant).
  **Breaking**.
- `routing.apply_edge_weights` — parameter `weight_name` → `weight`.
  Every other routing helper (routing, network_processing, utility,
  overhead) already used `weight` for the same concept; the outlier is
  fixed. **Breaking** for callers passing it as `weight_name=`.
- Snap-distance kwargs unified as `max_distance` (matching the
  geo_mapping / snap_features_to_nodes convention):
  - `network_snap.snap_to_network_nodes`: `max_radius` →
    `max_distance`; `priority_node_radius` →
    `priority_node_max_distance`.
  - `network_snap.insert_projected_nodes`: `max_radius` →
    `max_distance`.
  - `calibration.snap_counters_to_edges`: `search_radius` →
    `max_distance`.
  - `geo_mapping.map_points_to_filtered_lines`: `search_radius` →
    `max_distance` (and promoted to keyword-only).
  - `calibration.calibrate_edge_weights`: `snap_max_distance` →
    `max_distance` (dropping the `snap_` prefix that was only there to
    disambiguate from the earlier `max_trip_distance` kwarg — which
    has itself been dropped).

  Note: `0.2.0a0` renamed `snap_to_network_nodes.max_distance` →
  `max_radius` for "consistency with the rest of the snap API." The
  wider audit here found `max_distance` used in more places (4+
  geo_mapping helpers + `snap_features_to_nodes`) than `max_radius`
  (2 network_snap helpers), so consistency now goes the other way.
  **Breaking** for callers using any of the previous kwarg names.

### Changed
- Signature reorder — `routing.tiered_path_costs`,
  `routing.tiered_path_aggregate`, `routing.aggregate_along_paths`,
  `utility.route_utility` now take `graph` as the first positional
  argument (was `pairs` / `paths`). Matches
  `shortest_path_*_one_to_one`, `get_nested_edge_betweenness`, and
  the networkx convention. **Breaking** for positional callers;
  kwarg-based callers unaffected.
- `traffic_flows.nested_node_sample` — first 3 args stay positional
  (`pairs`, `weights`, `costs`); the remaining 9 promoted to
  keyword-only. Also decouples `n_picks` / `n_dest` semantics — the
  former controls independent draws per origin, the latter caps unique
  destinations sampled (with dedup). **Breaking** for positional
  callers and for callers passing both `n_picks` + `n_dest`.
- `network_processing.get_nested_edge_betweenness.weight` — now
  required positional (was `str | None = None` with a runtime raise on
  `None`). Runtime behavior identical.
- `od_pairs.get_pairs`, `build_cell_to_zone_node_map`, `node_values`,
  `lookup_dest_column_node` — `node_column` now defaults to
  `"node_id"` (the documented convention). Non-breaking for existing
  positional callers; simplifies new-caller ergonomics.
- `od_pairs.node_values` — argument order swapped: `node_list` moves
  before `node_column` so the latter can carry a default. **Breaking**
  for positional callers.
- `od_pairs.get_pairs` — always includes a self-pair (`origin →
  origin`) for every origin, regardless of `r_cells`. Required for
  downstream consumers (accessibility metrics) that treat the origin
  cell itself as a destination. **Breaking** for callers relying on
  the previous behavior.
- `network_processing.aggregate_nodes_to_edges` — aggregator arg
  aligned with `aggregate_edges_to_nodes`. Now accepts `'max'`,
  `'min'`, `'sum'`, `'mean'`, `'median'`, or a callable (previously
  `'sum'` / `'mean'` / `'median'` only). Non-breaking.
- `network_processing.aggregate_edges_to_nodes` — accepts `'median'`
  (previously `'max'` / `'min'` / `'mean'` / `'sum'` + callable).
  Non-breaking.
- `remove_duplicate_indices` moved from `aperta.geo_processing` to
  `aperta.data_processing`. **Breaking** for callers importing via the
  old path.
- `network_processing.clean_consolidated_edges` + `lanes_per_direction`
  moved to `aperta_atlas.osm` (formerly `aperta_atlas.osm`). The
  `lanes` / `oneway` semantics they encode are OSM tag conventions,
  so their public surface belongs in the OSM-aware layer.
  **Breaking**.
- `network_processing.collapse_osm_highway_lists_by_rank` — removed;
  the equivalent logic is private in `aperta_atlas.osm`. **Breaking**.
- `CalibrationResult` field rewrite. The flat fields
  `r_squared_baseline` / `r_squared_calibrated` / `r_squared_regression`
  / `rmse` / `rmse_by_distance` / `r2_by_distance_*` /
  `predicted_times` / `observed_times` / `iter_log` /
  `edge_duration_attr` are all removed. Replaced by three
  per-distance-band DataFrames `metrics_baseline` /
  `metrics_calibrated` / `metrics_regression` (each indexed by
  `"all"` / `"< 5 km"` / `"5-25 km"` / `">= 25 km"`, with columns
  `r2`, `rmse`, `bias`) plus `n_used`. Quick overall-fit access:
  `result.metrics_calibrated.loc['all', 'r2']`. Per-trip
  predictions/observations are no longer carried in the result;
  recompute via `routing.shortest_path_costs_one_to_one` on the
  calibrated graph if needed. **Breaking**.
- `calibration.calibrate_edge_weights` signature. Dropped:
  `min_trip_distance`, `max_trip_distance`. Added: `min_speed_kph`
  (default 1.0), `max_speed_kph` (default 120.0) — these clamp each
  edge's calibrated duration to a plausible speed range, replacing
  the previous trip-level distance filter. `constant` semantics
  changed from `bool` to `float | None` (initial constant value, or
  None to omit the OLS constant term). **Breaking**.
- `routing.shortest_path_metrics_one_to_one` — `cutoff` parameter
  removed. Self-pair (origin = destination) trips now zero-fill all
  metrics instead of routing them. **Breaking**.

### Consolidated
- `overhead.aggregate_dest_overhead_per_group_euclidean` +
  `aggregate_dest_overhead_per_group_routed` → single
  `aggregate_dest_overhead_per_group` with `distance='euclidean'` /
  `'routed'` dispatch. The two variants differed only in the
  last-mile distance model (Euclidean centroid distance vs. routed
  Dijkstra); everything else — group-by, weighted mean, per-cell
  overhead handling, output shape — was identical. Callers pick the
  distance mode via the `distance=` kwarg and supply the corresponding
  mode-specific kwargs (`speed=` for euclidean; `graph=`, `weight=`,
  `node_column=`, `cutoff=` for routed). **Breaking** for callers of
  either variant; migration is mechanical.

### Moved to aperta-atlas

Helpers whose only live callers were in project-specific code, or which
carried assumptions (OSM tag semantics, Swiss survey conventions) that
belong on aperta-atlas's side of the split. Nothing lost — everything is
one clone / import away.

- `aperta.data_processing.add_lat_lon`, `data_processing.add_group_id` →
  `aperta-atlas/src/preparation/switzerland/private/surveys/common.py`.
  Only live callers were the Swiss survey preprocessing scripts; no
  library-tier justification for keeping them in aperta. **Breaking**
  for callers importing from `aperta.data_processing`.
- `aperta.data_processing.add_internal_distance`, `filter_columns`,
  `get_col_agg_fns`, `aggregate`, `upcast`, `restore_integer_columns`,
  `get_available_metrics` → `aperta-atlas/src/aperta_atlas/_shelf.py`
  (a per-machine, gitignored "purgatory" for functions with no live
  callers but which might still be useful someday). Zero live callers in
  aperta OR aperta-atlas main pipeline; kept locally so functions can
  be resurrected without a git-archaeology pass if a genuine need
  surfaces. **Breaking** for `aperta.data_processing` importers.
- (Earlier in the cycle:) OSM-aware code — `clean_consolidated_edges`,
  `lanes_per_direction`, `OSM_HIGHWAY_RANKS`, `consolidate_intersections`,
  `flag_node_osm_classification`, `load_consolidated_graphml`,
  `categorize_pois`, and companions — moved to `aperta_atlas.osm`. See
  the "Sweeping OSM-boundary cleanup" sub-section below for the full
  list.

### Removed
- `aperta.topography` module — folded into `aperta.geo_processing`.
  Its two public entries (`fetch_copernicus_dem`,
  `_copernicus_tile_name`) live there now. `aperta[topo]` still
  installs the same `rasterio` + `requests` extras.
  **Breaking** for `from aperta.topography import …` callers.
- `overhead.linear_per_cell_overhead`, `overhead.add_node_overheads`,
  `overhead.add_origin_cell_overhead` — the node-keyed overhead API.
  Removed as part of the migration to geo-keyed overheads, which fixes
  the multi-zone-snap-node double-counting bug that the node-keyed
  variant carried. Live code now uses `overhead.add_geo_overheads`.
  The natural node-keyed "twin" of `add_geo_overheads`
  (`add_node_overheads`, lookup-based) can be recovered from git if
  ever needed; the correct pattern in the meantime is to precompute
  per-cell overheads and feed them into `add_geo_overheads`.
  **Breaking**.
- `routing.add_trip_overhead` — callable-based node-keyed overhead
  applier. Same family as the removed node-keyed API above; no live
  callers. **Breaking**.
- `routing.combine_edge_weights` — sum of per-edge components in
  place. LUMOS pattern that no longer has any live caller.
  **Breaking**.
- `routing.shortest_distances_from`,
  `routing.shortest_distances_pairwise` — zero live callers. The
  tiered API (`tiered_path_costs`) uses identical scipy internals and
  covers every atlas / lumos / notebook workflow. Callers wanting
  quick single-origin routing can either build a single-origin
  `TieredODNodePairs` or drop to `scipy.sparse.csgraph.dijkstra`
  directly (~3 lines). **Breaking**.
- `accessibility.flatten_index` — collapsed 2-level MultiIndex column
  outputs into `__`-joined strings. Zero live callers. **Breaking**.
- `traffic_flows.estimate_edge_flows` — one-shot scaling wrapper
  (`bc * expected_km / Σ(bc·length)`). No live callers; every
  downstream user (atlas, lumos, calibration notebook) calls the
  two-step (`nested_node_sample` + `get_nested_edge_betweenness`)
  directly and applies its own scaling. **Breaking**.
- `network_processing` back-compat re-export block. The `network_snap`
  / `routing_prep` names (`snap_to_network_nodes`,
  `insert_projected_nodes`, `prepare_network`, `PreparedGraph`,
  `MODE_DEFAULTS`, etc.) were also re-exported from
  `network_processing` after the `0.2.0a0` split. No live caller
  used those re-exports; the block is gone. **Breaking** for anyone
  still importing them via `network_processing` — import from the
  dedicated modules directly.

### Performance
- `network_snap.snap_to_network_nodes` — the priority-node-set
  construction is hoisted out of the per-point loop. At Bern-scale
  (~250 k cells, ~70 k nodes) this drops the call from hours to
  seconds. Result unchanged.
- `network_snap.insert_projected_nodes` — the "next available node
  ID" computation is hoisted out of the per-insertion loop. Material
  for callers inserting many virtual nodes on a country-scale graph.
- `geo_processing.sample_raster_at_points` — switched to per-point
  windowed raster reads instead of loading the full raster array.
  Drops memory pressure substantially on country-scale DEM rasters;
  runtime equivalent.

### Fixed
- `geo_mapping.map_polygons_to_points` and
  `geo_mapping.map_points_to_polygons` now return `distance = 0.0` for
  polygons / points matched by containment (the natural "contained =
  zero proximity" reading), rather than `NaN`. `NaN` is now reserved
  exclusively for TRULY unmatched inputs (`allow_nearest=False` or
  beyond `max_distance`). Prevents a silent trap where downstream
  arithmetic on the distance column was NaN-poisoned for every row
  whose polygon contained (or point was inside) the target — a
  pattern that only surfaces far downstream (e.g. digitizing
  NaN-inflated travel times to out-of-range bin indices, producing
  systematically all-zero output rows for those geometries). Three
  cleanly distinguishable states after the fix: containment → `0.0`,
  nearest-fallback → finite positive, unmatched → `NaN`. **Breaking**
  for callers that special-cased NaN as a "point-inside-polygon"
  sentinel; those callers should now check `distance == 0.0` instead
  (or drop the special case if they were `fillna(0)`-ing anyway).
- `od_pairs.lookup_dest_column_geo`,
  `od_pairs.lookup_dest_column_node`, and
  `utility.add_endpoint_utility` (origin-feature lookup) now raise
  `KeyError` on a missing destination / origin id instead of silently
  falling back to `np.nan` (or, for the utility origin lookup,
  silently dropping the term from the sum). Missing keys always
  signal a structural mismatch — most often the sign that a cost ODM
  (per-origin arrays of float values) was passed where a pair index
  (per-origin arrays of geo-unit / node ids) was expected. The
  previous silent fallback made that class of bug produce plausible-
  looking mostly-empty ODMs that only surfaced far downstream.
  Present-but-NaN values in a column still propagate as NaN — only
  missing keys raise. **Breaking** for callers that relied on the
  silent NaN fallback for partial-coverage scenarios (filter
  unreachable destinations upstream before calling).
- `network_processing.aggregate_nodes_to_edges` produced wrong results
  on `MultiDiGraph` inputs (treated parallel edges as a single edge).
  Now correctly emits one aggregated value per `(u, v, k)`.
- `geo_processing._copernicus_tile_name` (formerly
  `topography._copernicus_tile_name`) mis-formed tile names in the
  southern + western hemispheres (sign + zero-padding bugs). Affected
  any caller using `fetch_copernicus_dem` outside N+E.
- `calibration.calibrate_edge_weights` raised `UnboundLocalError` on
  the return statement when the first iteration's calibrated R² fell
  below `r2_tolerance` — `final_coefs` was only assigned inside a
  conditional branch that didn't fire. Now records the latest fit
  unconditionally on every iteration.
- `test_routing.SetMinIntrazonalCostTestCase` — stale identity check
  fixed to match the current `floor_intrazonal_costs` behavior (returns
  fresh dicts for all tiers, values unchanged when already above the
  floor). Renamed to
  `test_other_tiers_values_unchanged_when_above_floor`.

### Documentation
- Numerous docstring fixes for stale function references — the
  overhead trio no longer points at the removed `add_node_overheads`;
  `power_decay` refers to `floor_intrazonal_costs` (not the
  non-existent `add_intrazonal_cost`); `tiered_path_costs` drops the
  reference to the non-existent `tiered_path_costs_mp`; `utility`
  return-dtype claims corrected to match the FP32 default (previously
  claimed float64).
- README — added a "Companion project: aperta-atlas" section
  positioning the two repos (library vs. atlas pipeline + scaffolding)
  and a "Distinctive features" section highlighting edge-weight
  calibration against observed travel times (with sampled-betweenness
  flow estimation for car) and trip overheads for door-to-door
  realism.
- `CalibrationResult` docstring updated to describe the
  per-distance-band `metrics_*` DataFrames (previously still described
  the removed flat `r_squared_*` attributes).

### Sweeping OSM-boundary cleanup (2026-06-07)

Aperta is now strictly OSM-agnostic — the algorithm library consumes
whatever edge/node attributes the caller provides and never reads OSM-
specific tag content. All OSM-aware code (modules that recognise
specific OSM tag values like `'motorway'`, `'lanes'` semantics,
`'oneway'` conventions, OSMnx-wrapped fetchers) has been moved to
`aperta_atlas.osm` and `aperta_atlas.osm_helpers`. Aperta's
`network_processing.py` now contains only network-agnostic helpers
(`flag_node_intersection_topology`, `snap_features_to_nodes`,
`aggregate_*`, `get_nested_edge_betweenness`, `set_nx_edge_attributes_filled`).

Moved from `aperta.network_processing` → `aperta_atlas.osm`:

- `OSM_HIGHWAY_RANKS`, `_osm_highway_rank`
- `flag_node_osm_classification`
- `consolidate_intersections` (legacy OSMnx wrapper — refactored to
  use the canonical `clean_consolidated_edges` + `lanes_per_direction`
  helpers; the previous inlined `_*_osm` private duplicates removed).
  Also: the unused `flag_node_osm_classification` call at the end of
  this wrapper was dropped (nothing in the extended notebook reads
  the per-node OSM-classification flags it wrote). Callers that need
  those flags should now call `flag_node_osm_classification` directly
  on the result.
- `extract_obstacle_locations`
- `load_consolidated_graphml` + `_CONSOLIDATED_NODE_DTYPES` +
  `_CONSOLIDATED_EDGE_DTYPES` + `_PREFIX_SCAN_*` + `_scan_graphml_keys`
  + `_int_via_float` + `_int_via_bool_or_float` (graphml round-trip
  dtype handling — `aperta_atlas.osm` handles all of it now)

Moved from `aperta.osm_helpers` → `aperta_atlas.osm_helpers` (whole file),
then **further cleaned up same day** — and then trimmed further still:

- `osm_tag_query_for_categories`, `categorize_pois` (pure logic) →
  consolidated into `aperta_atlas.osm` (alongside the other OSM-aware
  helpers). Kept because the extended notebook's small-scale API path
  uses them.
- `fetch_pois`, `fetch_network` deleted entirely — replaced by inline
  `ox.graph_from_polygon` + `ox.project_graph` (and
  `ox.features_from_polygon` + `categorize_pois`) in the extended
  notebook. Three to five lines per call site, no library wrapper.
- `aperta_atlas.osm_helpers` module deleted.
- The Overpass-API alternative scripts in aperta-atlas
  (`networks_download_via_api.py` and a freshly-created
  `pois_download_via_api.py`) were **both deleted in the same wave**.
  aperta-atlas's mission is the country-scale PBF production pipeline;
  alternative-data-source scripts that documentation explicitly tells
  users to avoid don't belong there. The API path lives only as
  inlined snippets in `aperta/examples/extended/prepare/1_download.py`
  — the right home for "simplest way to get started" pedagogical
  examples.

aperta's CHANGELOG "Deprecated (scheduled for removal post-atlas)"
section is now empty for the OSM-helper subset — those items aren't
deprecated *in aperta* anymore; they've simply moved.

`aperta.visualization` is unchanged (it was already OSM-agnostic in
code; one docstring example updated to refer to
`aperta_atlas.osm.OSM_HIGHWAY_RANKS`).

**Breaking** for anyone importing the moved items from
`aperta.network_processing` or `aperta.osm_helpers`. The migration is
mechanical — `from aperta.X import Y` → `from aperta_atlas.osm[_helpers] import Y`.

The `examples/extended/` notebook source files were updated to the
new import paths in the same commit; users running the notebook now
need `aperta-atlas` installed (it was previously aperta-only). Since
the notebook is itself slated for retirement once `aperta-atlas/projects/atlas/`
takes over, the soft aperta-atlas dependency just moves the transition
forward.

### Extended-example restructure (2026-06-27)

The `examples/extended/` notebook stack has been substantially slimmed
in preparation for aperta-atlas going public, and — as a follow-up
step — the surviving showcase notebooks have been relocated from
`examples/extended/` to `examples/calibration/` (the surviving
notebooks are all calibration-focused; the extended-example folder
name no longer fit).

- **Removed** `examples/extended/prepare/` entirely (5 .py + 5 .ipynb,
  ~2.1k LOC of OSM-download / dasymetric / unit-mapping / topography /
  density prep). The canonical preparation pipeline is now
  [aperta-atlas](https://github.com/mmiotti/aperta-atlas) — much more
  consistent, more transferable, and properly scenario-aware. Trying
  to maintain two parallel prep paths was duplicative.

- **Removed** `examples/extended/coefficients/` (CSV stubs +
  hand-maintained tables). Coefficients are now SHOWN inline by each
  showcase notebook rather than persisted — they illustrate how the
  pipeline works but no longer represent it. The pipeline is
  aperta-atlas.

- **Refactored 2 keepers** (`calibrate_edge_weights`, `traffic_flows`)
  to read prepared inputs from aperta-atlas's `bern-public` scenario
  output. Configurable via `APERTA_DATA_ROOT` env var (defaults to
  `./data`). File-layout convention follows aperta-atlas's
  `<scenario>/shapes/`, `<scenario>/nw/`, `<scenario>/properties/`
  split, with raw POIs / buildings sourced from
  `<root>/preparation/world/osm/`. Each notebook now opens with a
  "showcase, not a pipeline stage" framing block and points at
  aperta-atlas for the integrated chain.

- **Removed** `accessibility.ipynb` / `.py` outright — aperta-atlas is
  now the canonical showcase for accessibility computation, and
  maintaining a parallel accessibility notebook here duplicated
  scope. The remaining aperta examples focus on the two showcase
  primitives (traffic flows and edge-weight calibration) that
  aperta-atlas builds on but doesn't itself re-demonstrate in
  standalone-notebook form.

- **Stopped writing files** from the surviving keepers. They're
  pure-demo: compute + show, no persistence. No notebook reads
  another's output.

- **Added** a consolidated `examples/calibration/calibration.ipynb` —
  end-to-end showcase notebook combining traffic-flow estimation and
  edge-weight calibration into a single runnable narrative.

- **Archived** `diagnose_*.py` (9 files) + `verify_filter_fix.py` to
  `examples/extended/_archive/` — they were dev artifacts from past
  debugging sessions, not part of the showcase surface.

- **Inlined** the `OSM_HIGHWAY_RANKS` rank table in `_figures.py` so
  the showcase notebooks no longer reach into aperta-atlas or
  aperta-atlas for OSM-specific conventions. aperta examples now depend
  ONLY on aperta (+ osmnx + geopandas + pandas + matplotlib).

- **Updated** `benchmark.py` + `paper_tier_figure.py` to the same
  aperta-atlas-shaped input layout.

> **Required setup change.** The notebooks now require aperta-atlas's
> `bern-public` scenario outputs at `<APERTA_DATA_ROOT>/atlas/bern-public/`.
> Until a pre-built artifact is hosted, users run aperta-atlas's
> preparation locally. Ground-truth files (proprietary Swiss data) still
> live under the override-able `APERTA_EXAMPLES_GROUND_TRUTH_DIR`.

### Deprecated (scheduled for removal post-atlas)

Empty as of 2026-06-07. The OSM-specific transitional surface that
previously lived here (`consolidate_intersections`,
`extract_obstacle_locations`, `load_consolidated_graphml`,
`fetch_network`, `OSM_HIGHWAY_RANKS`, `flag_node_osm_classification`,
the `_CONSOLIDATED_*_DTYPES` + `_PREFIX_SCAN_*` graphml dtype
machinery, the `_*_osm` private helpers) is no longer in aperta — all
of it moved to `aperta_atlas.osm` (formerly `aperta_atlas.osm`) in the
sweeping cleanup above.

When updating this list, also update the project-memory entries
[project-extended-notebooks-post-atlas] and
[project-aperta-vs-aperta-atlas-boundary] used by Claude Code.

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

[Unreleased]: https://github.com/mmiotti/aperta/compare/v0.3.0-alpha...HEAD
[0.3.0a0]: https://github.com/mmiotti/aperta/compare/v0.2.0-alpha...v0.3.0-alpha
[0.2.0a0]: https://github.com/mmiotti/aperta/compare/v0.1.0-alpha...v0.2.0-alpha
[0.1.0a0]: https://github.com/mmiotti/aperta/releases/tag/v0.1.0-alpha
