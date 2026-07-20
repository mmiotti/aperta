# Benchmark vs Pandana

Pandana is built for raw speed on one-shot all-pairs cost: contraction-hierarchy preprocessing + a C++ inner loop, both purpose-built for exactly that workload. Aperta is built for path-first routing (it returns realized paths, not just costs), reusable tiered ODMs that survive across scenarios on the same area, and cross-modal aggregation. Speed isn't the primary design goal — but it's useful to know how much the extra capabilities cost. On production-shape workloads (variant C below) the gap is ~2–6×; the constant-factor cost is dwarfed by the one-time OSMnx download + consolidation shared across every scenario run on the same area.

## Setup

Self-contained script — fetches the Bern OSM networks (walk, car) inline via `osmnx`; no pre-prepared inputs required. Cumulative-opportunity accessibility to a uniform per-cell weight (1 per cell — routing-engine benchmark, so realistic destination weights don't matter and using uniform weights sidesteps OSMnx's flaky building-count fetch on canton-scale queries):

- **Walk**: inner = Bern city, outer = city + 5 km. 15-minute time budget.
- **Car**: inner = Bern city, outer = city + 30 km. 15-minute time budget.

End-to-end wall time — Pandana's network construction + `precompute`, aperta's OD-pair construction + routing + accessibility. Lower is better.

## Results

| Setup                                                       | Walk (15 min) | Car (15 min) |
|-------------------------------------------------------------|--------------:|-------------:|
| Pandana — all graph nodes                                   |         0.9 s |        1.2 s |
| Aperta A — all graph nodes (single-tier, Euclidean cutoff)  |        62.8 s |      648.6 s |
| Aperta B — cell-snap origins, tiered destinations           |         7.5 s |      115.5 s |
| Aperta C — AOI-restricted cell origins, tiered destinations |         2.0 s |        7.0 s |

Graph size context: walk = 51 k nodes / 66 k edges; car = 44 k nodes / 105 k edges. AOI (Bern city) is 15.5 % of walk cells and 1.5 % of car cells. Fetch + graph build (dominated by OSMnx download on first run, cached thereafter): 26 s walk, 187 s car.

Three variants step through aperta's algorithmic levers against Pandana's baseline:

- **A — all-nodes, single-tier.** Apples-to-apples Dijkstra on the same problem. Pandana's contraction-hierarchy preprocess wins decisively — ~70× on walk, ~540× on car. Pandana's design center.
- **B — cell-snap origins, tiered destinations.** ~5–8× faster than A by using the 3-tier destination structure (cells for close pairs at cell resolution, aggregated zones for far pairs — replacing redundant intra-zone routing).
- **C — AOI cell origins, tiered destinations.** The realistic production setup — only cells inside the AOI are origins; the wider buffer supplies destinations and through-routing. Aperta runs at 2.0 s vs Pandana's 0.9 s on walk (~2×) and 7.0 s vs 1.2 s on car (~6×). The remaining gap reflects the constant-factor overhead of aperta's tiered OD structure + Python-side per-origin loops vs. Pandana's C++ inner loop.

**Not measured here**: iterative workloads. When edge weights change (calibration loops, scenario comparison, time-of-day variants), Pandana pays its CH preprocessing cost again on every graph change; aperta re-routes directly on the mutated live graph. That's aperta's story — it's why the library targets iterative and cross-modal workloads, not one-shot cost. A dedicated iterative-workload benchmark would need to be its own thing.

## Reproduce

```bash
cd examples/benchmarks && python benchmark.py
```

Self-contained — no external data required. First run fetches + caches the OSM networks (~25 s walk, ~3 min car); subsequent runs reuse the cache. Requires `pandana` (not in aperta's own extras — install separately: `pip install pandana`).
