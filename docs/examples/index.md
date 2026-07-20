# Example notebooks

Three tiers of runnable example notebooks ship with aperta. Each is paired
with a `.py` source file via [Jupytext](https://jupytext.readthedocs.io/) in
the repository; the rendered versions below are the committed `.ipynb`
outputs.

| Tier | Notebook | Run time | What it covers |
|---|---|---|---|
| Minimal | [`minimal`](minimal) | ~10 s | Every aperta primitive exercised exactly once, OSM data only (Cambridge MA). The "what does aperta do?" demo. |
| Walkthrough | [`walkthrough`](walkthrough) | ~1 min | Guided tour of every primitive on real OSM data (Central Paris): tiered ODs, geo-keyed reindex, overheads, three accessibility metrics, path-first per-edge feature aggregation, cross-modal logsum. Covers the full accessibility surface end-to-end. |
| Calibration | {doc}`calibration` | ~30 min | Consolidated calibration workflow on the Canton of Zurich (fetched inline from OSM): traffic-flow estimation via nested-betweenness sampling with survey-derived P(C), then flow-aware edge-weight calibration against Google-Maps peak-hour travel times. |

The **calibration notebook** fetches the OSM car network for the Canton
of Zurich inline via `osmnx`; the only external inputs are ground-truth
files whose source terms preclude redistribution — Google-Maps-derived
car travel times and Swiss ASTRA counter readings. It renders here
from committed outputs; re-running locally requires the private
ground-truth inputs. A public-data version is planned.

```{toctree}
:hidden:
:maxdepth: 1

minimal
walkthrough
calibration
```
