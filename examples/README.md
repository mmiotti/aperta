# aperta examples

Notebooks + scripts, in increasing depth.

| Kind | Notebook / script | Rendered view | Run time |
|---|---|---|---|
| Minimal | [`minimal/accessibility.ipynb`](minimal/accessibility.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/minimal.html) | A few seconds |
| Walkthrough | [`walkthrough/accessibility.ipynb`](walkthrough/accessibility.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/walkthrough.html) | About a minute |
| Calibration | [`calibration/calibration.ipynb`](calibration/calibration.ipynb) | [View on RTD ↗](https://aperta.readthedocs.io/en/latest/examples/calibration.html) | ~30 min |
| Benchmark | [`benchmarks/benchmark.py`](benchmarks/benchmark.py) | — | ~5–10 min |

What each covers:

- **Minimal** — every aperta primitive exercised exactly once, OSM data only (Cambridge MA). The "what does aperta do?" demo.
- **Walkthrough** — guided tour of every primitive on real OSM data (Central Paris): tiered ODs, geo-keyed reindex, overheads, three accessibility metrics, path-first per-edge feature aggregation, cross-modal logsum. Covers the full accessibility surface end-to-end and is self-contained.
- **Calibration** — one consolidated calibration workflow on the Canton of Zurich (fetched inline from OSM): first estimate per-edge traffic flows via nested-betweenness sampling, then calibrate edge-weight durations against observed peak-hour Google-Maps travel times with the flow-derived `vc_beta` congestion feature. Requires proprietary ground-truth files (counters + trip times) to run end-to-end.
- **Benchmark** — aperta vs pandana scaling benchmark on the canton of Bern (fetched inline from OSM, walk + car). Documents the headline scaling numbers in the top-level README.

Each `.ipynb` is paired with a `.py` via [Jupytext](https://jupytext.readthedocs.io/) (one source of truth in git diffs; the `.ipynb` is rendered on ReadTheDocs and GitHub).

> 💡 **Note:** GitHub's notebook renderer is currently experiencing an outage affecting many repositories (see [community discussion](https://github.com/orgs/community/discussions/197350)). The **View on RTD ↗** links above are the recommended way to view rendered notebooks; they work regardless of GitHub's renderer status.

### Data availability for the calibration + benchmark

- **Calibration** fetches its network (Canton of Zurich) inline from OSM via `osmnx`. It additionally requires ground-truth files — Google-Maps-derived OD travel times and Swiss ASTRA counter readings — whose source terms preclude redistribution. The notebook is included primarily as documentation of the calibration + flow-estimation *methods*; cells past §1 are not runnable end-to-end without the private ground-truth inputs. A public-data version of the ground-truth inputs is planned for a future release.
- **Benchmark** fetches its networks inline from OSM via `osmnx`: walk on city of Bern + 5 km, car on city of Bern + 30 km. Both modes use the city as the AOI — the different buffer sizes reflect each mode's realistic reach in the metric's time cutoff, and (for car) make the "small AOI inside a much larger network" story that motivates aperta's design especially visible. Fully public-data, no additional inputs required. First-run OSM fetch dominates wall-clock (~5–15 min total); OSMnx's built-in file cache speeds up subsequent runs. Absolute node counts in headline numbers drift slowly as OSM evolves — see the timestamp on the current README numbers.
