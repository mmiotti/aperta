# aperta examples

Three tiers, in increasing depth.

| Tier | Notebook | Rendered view | Run time |
|---|---|---|---|
| Minimal | [`minimal/accessibility.ipynb`](minimal/accessibility.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/minimal.html) | A few seconds |
| Walkthrough | [`walkthrough/accessibility.ipynb`](walkthrough/accessibility.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/walkthrough.html) | About a minute |
| Extended | [`extended/accessibility.ipynb`](extended/accessibility.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/extended_accessibility.html) | ~30 min |
| Extended | [`extended/calibrate_edge_weights.ipynb`](extended/calibrate_edge_weights.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/extended_calibration.html) | ~30 min |
| Extended | [`extended/traffic_flows.ipynb`](extended/traffic_flows.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/extended_traffic_flows.html) | ~30 min |

What each notebook covers:

- **Minimal** — every aperta primitive exercised exactly once, OSM data only (Cambridge MA). The "what does aperta do?" demo.
- **Walkthrough** — guided tour of every primitive on real OSM data (Central Paris): tiered ODs, geo-keyed reindex, overheads, three accessibility metrics, path-first per-edge feature aggregation, cross-modal logsum.
- **Extended** — near-production-scale, multi-mode showcase on Bern + 40 km. Full prep pipeline under [`extended/prepare/`](extended/prepare/), three-mode accessibility, traffic-flow estimation against observed counters, edge-weight calibration against ground-truth travel times. A full run (including OSM downloads) takes over an hour.

Each `.ipynb` is paired with a `.py` via [Jupytext](https://jupytext.readthedocs.io/) (one source of truth in git diffs; the `.ipynb` is rendered on ReadTheDocs and GitHub).

> 💡 **Note:** GitHub's notebook renderer is currently experiencing an outage affecting many repositories (see [community discussion](https://github.com/orgs/community/discussions/197350)). The **View on RTD ↗** links above are the recommended way to view rendered notebooks; they work regardless of GitHub's renderer status.

### Data availability for the calibration notebooks

`extended/prepare/` and `extended/accessibility.ipynb` are fully reproducible from OpenStreetMap and other public sources: all data downloads happen inside the notebooks themselves. The two calibration notebooks — `extended/calibrate_edge_weights.ipynb` (edge-weight calibration against observed car travel times) and `extended/traffic_flows.ipynb` (flow-estimation tuning against traffic-counter readings) — additionally require ground-truth data (Google-Maps-derived OD travel times and Swiss ASTRA counter readings) whose redistribution terms do not permit shipping inside this repo. These two notebooks are included primarily as documentation of the calibration *method*; their cells are not runnable end-to-end without the private inputs. A public-data version is planned for a future release.
