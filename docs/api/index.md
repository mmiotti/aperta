# API reference

Aperta is organised as a flat collection of algorithm modules. Each module
focuses on one phase of the six-phase accessibility workflow and operates
on plain `numpy` / `pandas` / `networkx` inputs without any framework
scaffold.

```{toctree}
:caption: Core OD + routing
:maxdepth: 1

od_pairs
routing
overhead
```

```{toctree}
:caption: Accessibility metrics
:maxdepth: 1

accessibility
utility
```

```{toctree}
:caption: Traffic flows + calibration
:maxdepth: 1

traffic_flows
calibration
```

```{toctree}
:caption: Geographic + network helpers
:maxdepth: 1

geo_processing
geo_mapping
network_processing
osm_helpers
topography
```

```{toctree}
:caption: Misc
:maxdepth: 1

visualization
errors
```
