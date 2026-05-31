# Benchmark vs Pandana

Pandana is built for raw speed on one-shot all-pairs cost: contraction-hierarchy preprocessing + a C++ inner loop, both purpose-built for exactly that workload. Aperta is built for path-first routing (it returns realized paths, not just costs), reusable tiered ODMs that survive across scenarios on the same area, and cross-modal aggregation. Speed isn't the primary design goal — but it's useful to know how much the extra capabilities cost. On accessibility workloads at this scale the gap turns out to be small (1–2×), and aperta is faster in some regimes. For further context, the upstream OSMnx download + consolidation for these two networks takes well over an hour — a one-time cost shared across every scenario run on the same area, dwarfing the per-query routing differences below.

## Setup

Cumulative-opportunity accessibility to total employment on the consolidated walk and car networks of Bern + 40 km (10 km AOI buffer + 30 km destination buffer — the same area the extended example notebooks build). End-to-end wall time, including each library's setup phase (Pandana network construction + `precompute`; aperta OD-pair construction + routing + accessibility); lower is better.

## Results

| Setup                                                       | Walk (15 min) |  Car (30 min) |
|-------------------------------------------------------------|--------------:|--------------:|
| Pandana — all graph nodes                                   |        16.2 s |        19.0 s |
| Aperta A — all graph nodes (single-tier, Euclidean cutoff)  |       130.9 s |       275.7 s |
| Aperta B — cell-snap origins, tiered destinations           |        62.5 s |       118.7 s |
| Aperta C — AOI-restricted cell origins, tiered destinations |        11.3 s |        28.3 s |

Three variants step through aperta's algorithmic levers against Pandana's baseline:

- **A — all-nodes, single-tier.** Apples-to-apples Dijkstra on the same problem. Pandana wins by ~8–15×, its design center.
- **B — cell-snap origins, tiered destinations.** ~2× faster than A by using the 3-tier destination structure (cell tier for close pairs at cell resolution; zone tier for far pairs, replacing redundant intra-zone routing).
- **C — AOI cell origins, tiered destinations.** The realistic production setup. Aperta runs in 11.3 s vs Pandana's 16.2 s on walk and 28.3 s vs 19.0 s on car — within 1–2× either way. The cross-over depends on workload: aperta wins when origins can be restricted relative to graph size (walk: 21 k AOI cells out of 204 k graph nodes); Pandana wins when its CH precompute amortizes well over a long routing cutoff on a smaller graph (car: 30 min on 34 k nodes).

## Reproduce

Run [`examples/extended/prepare/`](https://github.com/mmiotti/aperta/tree/main/examples/extended/prepare) end to end (one-time, slow — downloads + consolidates OSM networks), then `python examples/extended/benchmark.py`.
