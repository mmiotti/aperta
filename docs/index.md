---
hide-toc: true
---

# aperta

**Path-first, multi-modal accessibility analysis on transport networks.**

A Python library for routing, distance/time computation, and gravity- /
utility- / logsum-based accessibility metrics on
[NetworkX](https://networkx.org/) graphs (routed via
[`scipy.sparse.csgraph`](https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html)).
Designed around a path-first routing primitive that returns the realised
route alongside the OD travel cost, enabling utility-based accessibility,
cross-modal logsum, and per-route feature aggregation as first-class
operations.

The name is Latin/Italian for *open*.

## Status

**Pre-1.0, alpha.** Published alongside a toolkit paper (in submission).
APIs may change without notice until v1.0.

## Install

```bash
pip install aperta              # algorithms only
pip install 'aperta[osm]'       # + OSM ingestion (osmnx)
pip install 'aperta[examples]'  # + everything needed to run the example notebooks
```

Requires Python ≥ 3.11.

## Get started

- **Quickest read** — the [README](https://github.com/mmiotti/aperta/blob/main/README.md)
  has a 30-line walking-accessibility example showing one aperta call per
  workflow phase.
- **Runnable minimal example** —
  [`examples/minimal/accessibility.ipynb`](https://github.com/mmiotti/aperta/blob/main/examples/minimal/accessibility.ipynb)
  (Cambridge MA, ~10 s).
- **Guided tour of every primitive** —
  [`examples/walkthrough/accessibility.ipynb`](https://github.com/mmiotti/aperta/blob/main/examples/walkthrough/accessibility.ipynb)
  (Central Paris, walk + bike, cross-modal logsum, path-first per-edge
  feature aggregation, ~40 s).
- **Production-scale demo** —
  [`examples/extended/`](https://github.com/mmiotti/aperta/tree/main/examples/extended)
  (Bern + 25 km, calibration, traffic flows, ~30 min).

## API reference

The [API reference](api/index) covers every public function and class
across aperta's algorithm modules: routing, accessibility metrics, OD
pair structures, trip overheads, utility-based costs, traffic flow
estimation, edge-weight calibration, and the supporting geo / network /
OSM / topography helpers.

```{toctree}
:hidden:
:caption: Reference

api/index
```

```{toctree}
:hidden:
:caption: Project

GitHub <https://github.com/mmiotti/aperta>
PyPI <https://pypi.org/project/aperta/>
```
