"""Mode-aware preparation of a transport network for routing.

`prepare_network` is the entry point: it takes a freshly-loaded (typically
consolidated) graph and produces a `PreparedGraph` that's ready for trap-free
routing under a specific transport mode. Two preparation axes:

- **Directedness**: undirected (for walk / bike) or directed_scc-restricted
  (for car, where one-ways must be respected).
- **Cost-mask**: per-edge `cost_excluded_<mode>` flags for highway-tag values
  that don't apply to this mode (e.g. motorways for walk).

The output also carries the cost-masked largest connected (or strongly
connected) component as the **snap-eligible** node set — passed to
`insert_projected_nodes` and `snap_to_network_nodes` in `aperta.network_snap`
to prevent points from being mapped to trapped-but-not-routable nodes.

Per-mode defaults live in `MODE_DEFAULTS` (walk / bike / car). Subtype labels
(`car_night`, `ebike25`, etc.) work too — pass `base_mode` to inherit
defaults and override individual flags.
"""

import warnings
from dataclasses import dataclass
from typing import Iterable, Literal, cast

import networkx as nx

# Type alias for the cost-mask overwrite policy used by compute_snap_eligibility
# and the internal _write_cost_excluded helper. "always" overwrites the per-edge
# flag unconditionally (the right policy for the prep stage). "if_missing"
# preserves any existing value (the right policy for the consumer-side
# prepare_network, which runs on graphs whose cost-mask flag was already
# persisted by the prep stage).
OverwritePolicy = Literal["always", "if_missing"]

BaseMode = Literal["walk", "bike", "car"]
Directedness = Literal["undirected", "directed_scc"]


# Per-mode default configuration for `prepare_network`. The rationale for each
# choice is documented in the design memo `project_aperta_node_snap_trap_fix`:
#   - walk: undirected (walking does not respect one-ways) +
#     `network_type='all'` (avoids stripping pedestrian paths topologically
#     connected through highway) + motorway/trunk excluded at cost time.
#   - bike: undirected (untagged contraflow is more common than legit-bike-only
#     one-ways) + `'bike'`.
#   - car:  directed with snap-to-largest-SCC (one-ways are semantically real
#     for cars; flipping them produces wrong-direction routes) + `'drive'`.
# OSM-specific. `network_type` values (`'all'`, `'bike'`, `'drive'`) are OSMnx
# network filters, and `cost_excluded_tags` reads the OSM `highway` tag.
# Projects working with non-OSM road networks should override these settings
# explicitly when calling `prepare_network` (or skip it and write the
# `is_snap_eligible_*` / `cost_excluded_*` decorations directly).
#
# ⚠ WARNING: ASYMMETRIC PER-EDGE COSTS require `directed_scc` for ALL modes.
# Undirected routing collapses an edge's two orientations into one in the
# cost function — any per-direction asymmetry (uphill/downhill elevation
# penalties, time-of-day flow asymmetry, signed traffic regulations) gets
# averaged or arbitrarily picked, producing nonsense results. This is the
# classic OSRM trap: it defaults to undirected for active modes and silently
# breaks elevation-aware cycling/walking routing.
#
# If your project needs per-direction costs for walk or bike (e.g.
# elevation-aware active-mode routing), override the default:
#
#     prepare_network(graph, mode='walk', directedness='directed_scc')
#     prepare_network(graph, mode='bike', directedness='directed_scc')
#
# AND ensure the input graph carries genuinely directed edges per mode.
# OSMnx's `oneway` is car-oriented; for proper per-mode direction handling
# at extract time, see `aperta_atlas.osm.emit_directions` (which respects
# `oneway:bicycle`, `oneway:foot`, `cycleway*=opposite*` tags).
MODE_DEFAULTS: dict[BaseMode, dict] = {
    "walk": {
        "network_type": "all",
        "directedness": "undirected",
        "cost_excluded_tags": frozenset({"motorway", "motorway_link", "trunk", "trunk_link"}),
    },
    "bike": {
        "network_type": "bike",
        "directedness": "undirected",
        "cost_excluded_tags": frozenset(),
    },
    "car": {
        "network_type": "drive",
        "directedness": "directed_scc",
        "cost_excluded_tags": frozenset(),
    },
}


@dataclass(frozen=True)
class PreparedGraph:
    """Routing-ready graph plus the in-session metadata derived by `prepare_network`.

    `prepare_network` decorates the underlying `graph` in place with two
    boolean attributes — one per node (`snap_eligible_flag`) and one per
    edge (`cost_excluded_flag`) — so the trap-fix information survives
    `.graphml` roundtripping and is available to downstream consumers that
    receive only the graph (e.g., Pandana). The `PreparedGraph` itself is a
    thin session-time wrapper that memoizes the snap-eligible node set for
    direct 1:1 hand-off to `insert_projected_nodes` (as `eligible_node_ids=`)
    and `snap_to_network_nodes` (same), and records the resolved-config
    metadata for introspection.

    Attributes:
        graph: The routing graph. `MultiGraph` (undirected) when
            `directedness='undirected'`, `MultiDiGraph` (directed) when
            `directedness='directed_scc'`.
        snap_eligible_nodes: Node IDs from which every node in the set is
            mutually reachable under the chosen directedness — the largest
            connected component for `'undirected'`, the largest strongly
            connected component for `'directed_scc'`. Pass as
            `eligible_node_ids=` to `insert_projected_nodes` and
            `snap_to_network_nodes` to prevent points from being mapped to
            trapped nodes. Memoized; the same set is also recoverable from
            the per-node `snap_eligible_flag` attribute on `graph`.
        snap_eligible_flag: Name of the per-node boolean attribute written
            onto `graph` (default `f"is_snap_eligible_{mode}"`).
        cost_excluded_flag: Name of the per-edge boolean attribute written
            onto `graph` (default `f"cost_excluded_{mode}"`). Mode-specific
            cost functions consult this flag to assign cost = ∞.
        mode: The user-supplied mode label (free-form string).
        directedness: Resolved directedness setting.
        network_type: Resolved OSMnx network-type string (recorded for
            metadata / warnings; `prepare_network` does not re-fetch).
    """

    graph: nx.Graph
    snap_eligible_nodes: frozenset
    snap_eligible_flag: str
    cost_excluded_flag: str
    mode: str
    directedness: Directedness
    network_type: str


# Resolved-config carrier shared by `compute_snap_eligibility` and
# `prepare_network`. Both entry points need to turn the same set of caller-
# supplied args (plus per-mode defaults) into the same set of resolved
# values; factoring the resolution into one helper keeps the two functions
# from drifting.
@dataclass(frozen=True)
class _ResolvedConfig:
    directedness: Directedness
    network_type: str
    cost_excluded_tags: frozenset
    snap_eligible_flag: str
    cost_excluded_flag: str
    effective_base: BaseMode | None


def _resolve_mode_config(
    mode: str,
    base_mode: BaseMode | None,
    directedness: Directedness | None,
    network_type: str | None,
    cost_excluded_tags: Iterable[str] | None,
    snap_eligible_flag: str | None,
    cost_excluded_flag: str | None,
) -> _ResolvedConfig:
    """Resolve mode-aware defaults + flag names. Single source of truth for
    both `compute_snap_eligibility` and `prepare_network`.
    """
    if base_mode is not None and base_mode not in MODE_DEFAULTS:
        raise ValueError(f"`base_mode` must be one of {list(MODE_DEFAULTS)}; got {base_mode!r}.")

    defaults: dict | None
    effective_base: BaseMode | None
    if base_mode is not None:
        defaults = MODE_DEFAULTS[base_mode]
        effective_base = base_mode
    elif mode in MODE_DEFAULTS:
        defaults = MODE_DEFAULTS[cast(BaseMode, mode)]
        effective_base = cast(BaseMode, mode)
    else:
        defaults = None
        effective_base = None

    def _resolve(flag_value, default_key: str, flag_name: str):
        if flag_value is not None:
            return flag_value
        if defaults is None:
            raise ValueError(
                f"`{flag_name}` not provided and no defaults are available for "
                f"mode={mode!r}. Pass `base_mode` to inherit from a known base, "
                f"or supply `{flag_name}` explicitly."
            )
        return defaults[default_key]

    return _ResolvedConfig(
        directedness=_resolve(directedness, "directedness", "directedness"),
        network_type=_resolve(network_type, "network_type", "network_type"),
        cost_excluded_tags=frozenset(
            _resolve(cost_excluded_tags, "cost_excluded_tags", "cost_excluded_tags")
        ),
        snap_eligible_flag=(
            snap_eligible_flag if snap_eligible_flag is not None else f"is_snap_eligible_{mode}"
        ),
        cost_excluded_flag=(
            cost_excluded_flag if cost_excluded_flag is not None else f"cost_excluded_{mode}"
        ),
        effective_base=effective_base,
    )


def _write_cost_excluded(
    graph: nx.Graph,
    excluded_tags: frozenset,
    flag_name: str,
    *,
    overwrite: OverwritePolicy,
) -> None:
    """Mutate `graph` in place: write `flag_name` (bool) per edge based on the
    edge's `highway` tag and the excluded-tag set.

    `overwrite='if_missing'` skips edges that already carry the attribute,
    which is what `prepare_network` wants when running on a graph whose
    cost-mask flag was already persisted by a prior `compute_snap_eligibility`
    call (and may have been propagated to split-child edges by
    `insert_projected_nodes`).
    """
    if graph.is_multigraph():
        for u, v, k, edata in graph.edges(keys=True, data=True):
            if overwrite == "if_missing" and flag_name in edata:
                continue
            graph.edges[u, v, k][flag_name] = _osm_highway_in_excluded(
                edata.get("highway"), excluded_tags
            )
    else:
        for u, v, edata in graph.edges(data=True):
            if overwrite == "if_missing" and flag_name in edata:
                continue
            graph.edges[u, v][flag_name] = _osm_highway_in_excluded(
                edata.get("highway"), excluded_tags
            )


def compute_snap_eligibility(
    graph: nx.MultiDiGraph,
    mode: str,
    *,
    base_mode: BaseMode | None = None,
    directedness: Directedness | None = None,
    network_type: str | None = None,
    cost_excluded_tags: Iterable[str] | None = None,
    cost_excluded_flag: str | None = None,
    overwrite_cost_excluded: OverwritePolicy = "always",
) -> tuple[frozenset, str]:
    """Prep-stage entry point: write the per-edge cost-mask flag and return the
    snap-eligible node set, ready to hand to `insert_projected_nodes` and
    `snap_to_network_nodes`.

    Step 3 of the canonical preparation pipeline (cf. the snap-across-barrier
    7-step workflow). Operates on the ORIGINAL directed graph — does NOT
    apply the directedness transform — because the downstream node-insertion
    (`insert_projected_nodes`) needs the directed `MultiDiGraph` for two-way
    edge detection via `split_two_way_edge_at_point`. For walk / bike modes,
    eligibility is computed on an internal undirected *view*
    (`graph.to_undirected(as_view=True)`) without mutating the caller's
    graph.

    What this function DOES:

    - Resolves mode-aware defaults (directedness, network_type,
      cost_excluded_tags) the same way as `prepare_network`.
    - Writes `cost_excluded_<mode>` (bool) per edge on the input graph.
      With `overwrite_cost_excluded='always'` (the default), every edge is
      rewritten; with `'if_missing'`, edges that already carry the flag
      are preserved (matches the policy `prepare_network` uses when
      reloading a graph whose cost-mask was already persisted).
    - Computes the largest connected component (undirected) or strongly
      connected component (directed) of the cost-masked subgraph; returns
      it as a frozenset.
    - Emits `_check_combination` warnings for known-bad mode / config
      combinations.

    What this function does NOT do (intentionally):

    - It does NOT call `graph.to_undirected()` as a transformation; the
      caller's graph stays a `MultiDiGraph`.
    - It does NOT write `is_snap_eligible_<mode>` per node. That decoration
      belongs to the post-insertion consumer-side `prepare_network` call,
      which reclassifies any virtual nodes inserted by
      `insert_projected_nodes` in step 4.
    - It does NOT return a `PreparedGraph`. The caller wants only the two
      pieces `insert_projected_nodes` and `snap_to_network_nodes` consume
      (`eligible_node_ids=` and `cost_excluded_flag=`), and returning a
      `PreparedGraph` here would over-promise (the per-node flag isn't
      written yet).

    Args:
        graph: Routing graph, typically the output of `fetch_network` +
            `consolidate_intersections`. Must be directed (`nx.MultiDiGraph`).
        mode: Free-form mode label. If it matches a `BaseMode`
            (`'walk'`/`'bike'`/`'car'`), pulls defaults from `MODE_DEFAULTS`.
        base_mode, directedness, network_type, cost_excluded_tags,
            cost_excluded_flag: same semantics as `prepare_network`.
        overwrite_cost_excluded: `'always'` (default) rewrites
            `cost_excluded_<mode>` on every edge; `'if_missing'` preserves
            existing values.

    Returns:
        Tuple `(snap_eligible_nodes, cost_excluded_flag)`. Pass directly as
        `eligible_node_ids=` and `cost_excluded_flag=` to
        `insert_projected_nodes` (and the same `eligible_node_ids=` to the
        subsequent `snap_to_network_nodes` call).

    Raises:
        ValueError: same conditions as `prepare_network` (missing defaults,
            wrong directedness).
    """
    cfg = _resolve_mode_config(
        mode,
        base_mode,
        directedness,
        network_type,
        cost_excluded_tags,
        snap_eligible_flag=None,
        cost_excluded_flag=cost_excluded_flag,
    )

    if cfg.directedness == "directed_scc" and not graph.is_directed():
        raise ValueError(
            "`directedness='directed_scc'` requires a directed input graph; "
            "got an undirected graph."
        )

    if cfg.effective_base is not None:
        _check_combination(
            cfg.effective_base,
            cfg.directedness,
            cfg.network_type,
            cfg.cost_excluded_tags,
        )

    _write_cost_excluded(
        graph,
        cfg.cost_excluded_tags,
        cfg.cost_excluded_flag,
        overwrite=overwrite_cost_excluded,
    )

    # For walk/bike, eligibility is computed on an undirected VIEW (no copy,
    # no mutation of the caller's graph). For car, directly on the directed
    # graph (largest SCC of cost-masked subgraph).
    if cfg.directedness == "undirected":
        view: nx.Graph = graph.to_undirected(as_view=True)
    else:
        view = graph
    snap_nodes = compute_snap_eligible_nodes(view, cfg.directedness, cfg.cost_excluded_flag)

    return snap_nodes, cfg.cost_excluded_flag


def prepare_network(
    graph: nx.MultiDiGraph,
    mode: str,
    *,
    base_mode: BaseMode | None = None,
    directedness: Directedness | None = None,
    network_type: str | None = None,
    cost_excluded_tags: Iterable[str] | None = None,
    snap_eligible_flag: str | None = None,
    cost_excluded_flag: str | None = None,
) -> PreparedGraph:
    """Apply mode-aware graph preparation: directedness + snap-eligibility + cost-exclusion flags.

    Wraps an already-fetched (and typically already-consolidated) graph to
    produce a `PreparedGraph` ready for trap-free routing. The two axes —
    `directedness` and `network_type` — each have per-mode defaults
    (`MODE_DEFAULTS`) that callers can override. The trap-fix outputs
    (`snap_eligible_flag` per node, `cost_excluded_flag` per edge) are
    written onto the underlying graph in place so they survive `.graphml`
    roundtripping and are available to downstream consumers that don't
    know about `PreparedGraph`.

    The `mode` argument is a free-form string label used for default flag
    names and for warning-policy lookup. Common cases (`'walk'`, `'bike'`,
    `'car'`) pick up defaults directly from `MODE_DEFAULTS`. Subtype /
    context labels (e.g., `'car_night'`, `'ebike25'`) work too; pass
    `base_mode` to inherit defaults from a known base, and override
    individual flags as needed. Default flag names embed `mode`, so
    multiple calls on the same graph with different mode labels accumulate
    independent decorations (`is_snap_eligible_car_peak`,
    `is_snap_eligible_car_night`, etc.) instead of overwriting each other.

    Args:
        graph: Routing graph, typically the output of `fetch_network` +
            `consolidate_intersections`. Must be directed
            (`nx.MultiDiGraph`) when `directedness='directed_scc'` is
            chosen.
        mode: Free-form mode label. Used to derive default flag names and,
            if it matches a `BaseMode` (`'walk'`/`'bike'`/`'car'`), to pull
            defaults from `MODE_DEFAULTS`.
        base_mode: Optional `BaseMode` for `MODE_DEFAULTS` lookup when
            `mode` itself is not a `BaseMode` (e.g., `mode='car_night'`,
            `base_mode='car'`). Resolution order: `base_mode` if given,
            else `mode` if in `MODE_DEFAULTS`, else no defaults available
            and all flags must be supplied explicitly.
        directedness: `'undirected'` applies `to_undirected()` and snaps to
            the largest connected component. `'directed_scc'` keeps the
            graph directed and snaps to the largest strongly connected
            component. `None` resolves from defaults.
        network_type: The OSMnx `network_type` used to fetch `graph`. Used
            only for warning-policy decisions and recorded as metadata;
            this function does not re-fetch. `None` resolves from defaults.
        cost_excluded_tags: Highway-tag values to mark for cost = ∞
            downstream (e.g., `{'motorway', 'trunk'}` for walk on
            `network_type='all'`). `None` resolves from defaults; an empty
            iterable is a valid explicit override.
        snap_eligible_flag: Name of the per-node bool attribute to write.
            `None` defaults to `f"is_snap_eligible_{mode}"`.
        cost_excluded_flag: Name of the per-edge bool attribute to write.
            `None` defaults to `f"cost_excluded_{mode}"`.

    Returns:
        A `PreparedGraph` carrying the (possibly transformed) routing
        graph, the snap-eligible node set (also written per-node onto the
        graph), and the resolved metadata.

    Warns:
        `UserWarning` on combinations known to silently produce wrong
        answers (e.g., `walk + network_type='walk'` strips pedestrian
        paths; `car + directedness='undirected'` routes the wrong way down
        one-way streets). Warnings are keyed by the resolved base mode, so
        they fire for subtypes too (`'car_night'` with `base_mode='car'`
        still warns on `directedness='undirected'`). Deviations from a
        mode default that are not known-problematic do NOT warn.

    Raises:
        ValueError: if `mode` has no defaults available (neither
            `base_mode` nor `mode in MODE_DEFAULTS`) and any of
            `directedness` / `network_type` / `cost_excluded_tags` is
            unspecified; or if `directedness='directed_scc'` is requested
            on an already-undirected graph.
    """
    cfg = _resolve_mode_config(
        mode,
        base_mode,
        directedness,
        network_type,
        cost_excluded_tags,
        snap_eligible_flag,
        cost_excluded_flag,
    )

    if cfg.effective_base is not None:
        _check_combination(
            cfg.effective_base,
            cfg.directedness,
            cfg.network_type,
            cfg.cost_excluded_tags,
        )

    if cfg.directedness == "undirected":
        prepared_graph: nx.Graph = graph.to_undirected()
    else:
        if not graph.is_directed():
            raise ValueError(
                "`directedness='directed_scc'` requires a directed input graph; "
                "got an undirected graph."
            )
        prepared_graph = graph

    # Write per-edge cost-mask flag with `if_missing` semantics: edges that
    # already carry the flag (because the prep stage's
    # `compute_snap_eligibility` persisted it, and `insert_projected_nodes`
    # propagated it to split-child edges) keep their values. Edges that
    # don't have the flag get it computed now. This makes prepare_network
    # idempotent: calling it twice — once at prep time, once at consumer
    # time — is a no-op on the second pass for edges, and re-derives the
    # eligible set fresh (picking up any virtual nodes that got inserted
    # between the two calls).
    _write_cost_excluded(
        prepared_graph,
        cfg.cost_excluded_tags,
        cfg.cost_excluded_flag,
        overwrite="if_missing",
    )

    snap_nodes = compute_snap_eligible_nodes(
        prepared_graph, cfg.directedness, cfg.cost_excluded_flag
    )

    # Decorate the graph in place so the trap-fix information rides along
    # with the graph (e.g., through .graphml roundtripping) and downstream
    # consumers can use it without knowing about `PreparedGraph`.
    for n in prepared_graph.nodes:
        prepared_graph.nodes[n][cfg.snap_eligible_flag] = n in snap_nodes

    return PreparedGraph(
        graph=prepared_graph,
        snap_eligible_nodes=snap_nodes,
        snap_eligible_flag=cfg.snap_eligible_flag,
        cost_excluded_flag=cfg.cost_excluded_flag,
        mode=mode,
        directedness=cfg.directedness,
        network_type=cfg.network_type,
    )


def compute_snap_eligible_nodes(
    graph: nx.Graph,
    directedness: Directedness,
    cost_excluded_flag: str,
) -> frozenset:
    """Largest CC (undirected) or SCC (directed) of the *cost-masked* subgraph.

    Pure compute — does NOT mutate `graph`. Companion to
    `compute_snap_eligibility` (which writes the per-edge cost-mask AND
    computes the SCC in one call); use this directly when you've already
    written the cost-mask elsewhere (e.g., a post-insertion recompute after
    `insert_projected_nodes` that needs the updated SCC without rewriting
    the cost-mask).

    The cost-masked subgraph contains only edges where the per-edge attribute
    `cost_excluded_flag` is absent or `False`. Computing eligibility on this
    subgraph (rather than on the raw topology) is what distinguishes
    "topologically reachable" from "reachable at finite routing cost." A node
    connected to the rest of the largest CC only via cost-excluded edges
    (e.g., a pedestrian-network node at a motorway interchange) is in the
    largest topological CC but practically unreachable at routing time — the
    cost-masked subgraph correctly identifies it as ineligible.

    When `cost_excluded_tags` is empty (the default for bike + car), no edges
    carry `cost_excluded_flag=True`, so the cost-masked subgraph equals the
    full graph and the result matches plain topological analysis.
    """
    ok_edges: list
    if graph.is_multigraph():
        ok_edges = [
            (u, v, k)
            for u, v, k, d in graph.edges(keys=True, data=True)
            if not d.get(cost_excluded_flag, False)
        ]
    else:
        ok_edges = [
            (u, v) for u, v, d in graph.edges(data=True) if not d.get(cost_excluded_flag, False)
        ]
    subgraph = graph.edge_subgraph(ok_edges)
    if directedness == "undirected":
        components = nx.connected_components(subgraph)
    else:
        components = nx.strongly_connected_components(subgraph)
    largest: set = max(components, key=len, default=set())
    return frozenset(largest)


def _osm_highway_in_excluded(highway_value, excluded: frozenset) -> bool:
    """True if the edge's `highway` tag is in `excluded`. Tolerates list-valued
    tags (post-`consolidate_intersections` edges can carry a list of highway
    strings) and `None`.
    """
    if highway_value is None:
        return False
    if isinstance(highway_value, list):
        return any(v in excluded for v in highway_value)
    return highway_value in excluded


def _check_combination(
    base_mode: BaseMode,
    directedness: Directedness,
    network_type: str,
    cost_excluded_tags: frozenset,
) -> None:
    """Emit `UserWarning` for combinations known to silently produce wrong
    answers. Keyed by the resolved base mode, so warnings fire for subtypes
    (e.g., `'car_night'` with `base_mode='car'` triggers car-rule warnings).
    Deviations from defaults that are not known-problematic stay silent —
    warnings are reserved for actual risk, not "this is not the recommended
    setting".
    """
    if base_mode == "walk":
        # Historical walk-specific warnings (`directed_scc` "unnecessarily
        # restrictive", `network_type='walk'` "Cambridge pitfall",
        # `network_type='all'` + empty cost_excluded_tags "walking will be
        # routed across motorways") were dropped: they each assumed a
        # specific OSMnx-fetched data shape. For custom PBF pipelines the
        # walk network is pre-filtered at extraction time and asymmetric
        # per-edge costs (elevation gradients) make `directed_scc` the
        # correct choice — exactly the case the asymmetric-cost note at
        # the top of this module flags. Reintroduce only with an opt-out
        # for callers who manage their own filtering.
        pass
    elif base_mode == "car":
        if directedness == "undirected":
            warnings.warn(
                "car + directedness='undirected' treats one-way streets as "
                "bidirectional. Was this intentional? Routes may go the wrong way "
                "down one-way streets.",
                UserWarning,
                stacklevel=3,
            )
        if network_type == "all" and not cost_excluded_tags:
            warnings.warn(
                "car + network_type='all' with empty cost_excluded_tags: car routing "
                "may traverse footways, pedestrian paths, or steps. Consider "
                "excluding non-drivable highway tags.",
                UserWarning,
                stacklevel=3,
            )
