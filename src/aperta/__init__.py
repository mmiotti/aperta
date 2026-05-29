"""
aperta — cross-modal accessibility analysis on transport networks.

The library is organized around a six-phase workflow:

  1. Load and prepare data  — networks (per mode), land use, topography.
  2. Map data to units      — `cells → zones` aggregation hierarchy
                              (`geo_mapping`, `network_processing.snap_to_network_nodes`,
                              `network_processing.assign_to_eligible_centroid`).
  3. Build sparse OD pairs  — `od_pairs.get_pairs` returns a `TieredODNodePairs`
                              (three distance tiers: cells_to_cells,
                              cells_to_zones, zones_to_zones). Lift to
                              `TieredODGeoPairs` via
                              `od_pairs.reindex_by_geo_unit` for cross-modal
                              alignment / geo-unit-keyed overheads.
  4. Estimate traffic flows — `traffic_flows.nested_node_sample` +
                              `network_processing.get_*_betweenness*`.
  5. Estimate travel costs  — `routing.tiered_path_costs` /
                              `routing.tiered_path_aggregate` (Dijkstra on any
                              networkx graph; the latter also aggregates
                              per-edge / per-node features along realised
                              routes via `PathAggregation` / `NodeAggregation`)
                              + the `overhead` module
                              (`add_node_overheads` for node-keyed,
                              `add_geo_overheads` / `add_origin_cell_overhead`
                              for geo-keyed). `utility.route_utility` +
                              `add_endpoint_utility` for utility-based costs.
                              `routing.aggregate_along_paths` is the path-walker
                              primitive when you have a pre-computed list of
                              paths rather than a `TieredODPairs`.
  6. Calculate accessibility — `accessibility.cumulative_opportunities` (cumulative),
                               `accessibility.gravity`, `accessibility.nearest_k`.
                               Cross-modal: combine per-mode `TieredODGeoPairs`
                               with `od_pairs.aggregate_across_modes` first.

All algorithm modules (`od_pairs`, `routing`, `overhead`, `accessibility`,
`utility`, `traffic_flows`, `geo_processing`, `geo_mapping`,
`network_processing`, `visualization`, `osm_helpers`, `calibration`,
`topography`) operate on plain numpy / pandas / networkx inputs — no
filesystem assumptions, no opinionated project structure.
See `tests/test_workflow.py` for the ~150-line end-to-end toy-world
example, `examples/minimal/accessibility.ipynb` for a ~40-line OSM
quickstart, `examples/walkthrough/accessibility.ipynb` for the full
guided tour, and `examples/extended/` for a multi-notebook showcase
with published-paper calibration (Bern + 40 km).

Key types:
  - `od_pairs.TieredODNodePairs` — three-tier OD dict-of-arrays keyed by network
                                   node IDs. Output of routing.
  - `od_pairs.TieredODGeoPairs`  — three-tier OD dict-of-arrays keyed by
                                   geo-unit IDs (cells / zones).
                                   Mode-agnostic; required for cross-modal
                                   accessibility and geo-unit-keyed overhead.
  - `od_pairs.TieredODPairs`     — abstract base of the two above; use as a
                                   type hint when key space doesn't matter.
  - `accessibility.Bin`       — half-open cost bin for `cumulative_opportunities`.
  - `accessibility.Decay`     — named cost-decay callable for `gravity`.
  - `utility.Utility`         — linear utility spec (constant + cost + route
                                + origin + destination feature coefficients).
"""
