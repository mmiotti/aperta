# Example notebooks

Three tiers of runnable example notebooks ship with aperta. Each is paired
with a `.py` source file via [Jupytext](https://jupytext.readthedocs.io/) in
the repository; the rendered versions below are the committed `.ipynb`
outputs.

| Tier | Notebook | Run time | What it covers |
|---|---|---|---|
| Minimal | [`minimal`](minimal) | ~10 s | Every aperta primitive exercised exactly once, OSM data only (Cambridge MA). The "what does aperta do?" demo. |
| Walkthrough | [`walkthrough`](walkthrough) | ~1 min | Guided tour of every primitive on real OSM data (Central Paris): tiered ODs, geo-keyed reindex, overheads, three accessibility metrics, path-first per-edge feature aggregation, cross-modal logsum. |
| Extended | [`extended_accessibility`](extended_accessibility) | ~30 min | Production-scale Bern + 40 km cross-modal accessibility (walking, cycling, car at peak hours; three destination types). Generates the figures used in the toolkit paper. |
| Extended | [`extended_calibration`](extended_calibration) | ~30 min | Iterative edge-weight calibration against Google-Maps-derived car travel times. |
| Extended | [`extended_traffic_flows`](extended_traffic_flows) | ~30 min | Per-edge AADT estimation via cost-decay-weighted nested-betweenness sampling, with counter-fit diagnostics. |

The two **extended-example calibration notebooks**
(`extended_calibration` and `extended_traffic_flows`) require
ground-truth inputs (Google-Maps-derived car travel times and traffic-counter
readings) whose source terms preclude redistribution. They render here from
committed outputs, but re-running them locally requires the private inputs.
A public-data version is planned.

```{toctree}
:hidden:
:maxdepth: 1

minimal
walkthrough
extended_accessibility
extended_calibration
extended_traffic_flows
```
