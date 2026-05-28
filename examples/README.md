# Aperta examples

Three tiers, in increasing depth.

| Tier | What | Run time |
|---|---|---|
| [`minimal/accessibility.ipynb`](minimal/accessibility.ipynb) | ~100 lines. Every aperta primitive exercised exactly once, on OSM data only (Cambridge MA). The "what does aperta do?" demo. | A few seconds. |
| [`walkthrough/accessibility.ipynb`](walkthrough/accessibility.ipynb) | ~1000 lines. Guided tour of key aperta features on real OSM data (Central Paris): tiered ODs, geo-keyed reindex, overheads, three accessibility metrics, path-first per-edge feature aggregation, cross-modal logsum. | About a minute. |
| [`extended/`](extended/) | Near-production-scale, multi-mode showcase on Bern + 25 km: full prep pipeline ([`extended/prepare/`](extended/prepare/)), three-mode accessibility ([`extended/accessibility.ipynb`](extended/accessibility.ipynb)), traffic-flow estimation against observed counters ([`extended/traffic_flows.ipynb`](extended/traffic_flows.ipynb)), and edge-weight calibration against ground-truth travel times ([`extended/calibrate_edge_weights.ipynb`](extended/calibrate_edge_weights.ipynb)). | Over an hour for a full run — most of it OSM downloads and network preprocessing in `extended/prepare/`. |

Each `.ipynb` is paired with a `.py` via [Jupytext](https://jupytext.readthedocs.io/) (one source of truth in git diffs, the `.ipynb` is what GitHub renders).
