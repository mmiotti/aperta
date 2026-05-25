"""
Helpers for downloading and categorizing OpenStreetMap data via osmnx.

Two concerns, separated:

1. **Categorized POI downloads** — accessibility analyses typically classify
   OSM features into user-defined categories (`groceries`, `schools`, …) that
   may bundle multiple OSM `tag:value` pairs, optionally with per-pair
   weights (a `convenience` shop might count as 0.5 of a `supermarket` for
   the `groceries` category). `fetch_pois` automates the full pipeline:
   build the OSM tag query from the category map, download via osmnx, tag
   each feature with per-category count + weighted-count columns, drop
   features matching no category.

2. **Network downloads** — `fetch_network` wraps
   `osmnx.graph_from_polygon` + `osmnx.project_graph` so the standard
   "fetch in EPSG:4326, project to a metric CRS" pattern is one line.

Categorization without download (when POIs come from another source —
cached, a different provider, etc.) is also exposed as `categorize_pois`.

`osmnx` is an optional dependency (`aperta[osm]` or `aperta[examples]`);
calls to `fetch_*` import it lazily and raise a helpful error if missing.

Example category map:

    POI_CATEGORIES = {
        'groceries': [
            ('shop:supermarket', 1.0),
            ('shop:convenience', 0.5),
            ('shop:bakery',      0.5),
        ],
        'education_school': [('amenity:school', 1.0)],
        'transit_rail':     [('railway:station', 3), ('railway:halt', 1)],
    }

    pois = osm_helpers.fetch_pois(
        polygon=dest_polygon, polygon_crs='EPSG:2056',
        category_map=POI_CATEGORIES, target_crs='EPSG:2056',
    )
    # pois now has columns: geometry, name, amenity, shop, railway, ...
    # plus per-category columns: groceries, groceries_weight,
    # education_school, education_school_weight, transit_rail, transit_rail_weight.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely.geometry.base


# A category map: `{user_category -> [(osm_tag_pair, weight), ...]}`
# where `osm_tag_pair` is a `'key:value'` string like `'shop:supermarket'`.
CategoryMap = dict[str, list[tuple[str, float]]]


# ---------------------------------------------------------------------------
# Pure logic — no network access
# ---------------------------------------------------------------------------

def osm_tag_query_for_categories(category_map: CategoryMap) -> dict[str, list[str]]:
    """Build the osmnx `tags=` argument from a category map.

    Unions every `key:value` pair across all categories and groups by key.
    The result is the minimal query that returns *every* feature any
    category could match.

    Args:
        category_map: `{user_category -> [(tag_pair, weight), ...]}` where
            `tag_pair` is a `'key:value'` string.

    Returns:
        `{osm_tag_key -> [values, ...]}` with values sorted (deterministic
        for caching / hashing). Pass directly as the `tags` argument to
        `osmnx.features_from_polygon` / `osmnx.features_from_place`.
    """
    out: dict[str, set[str]] = {}
    for tags in category_map.values():
        for tag_pair, _weight in tags:
            if ':' not in tag_pair:
                raise ValueError(
                    f"Tag pair {tag_pair!r} must be 'key:value' "
                    f"(e.g. 'shop:supermarket').")
            key, value = tag_pair.split(':', 1)
            out.setdefault(key, set()).add(value)
    return {k: sorted(v) for k, v in out.items()}


def categorize_pois(
    pois: gpd.GeoDataFrame,
    category_map: CategoryMap,
    *,
    weight_suffix: str = '_weight',
    drop_unmatched: bool = True,
) -> gpd.GeoDataFrame:
    """Add per-category count + weighted-count columns to a POI GeoDataFrame.

    For each `(category, [(tag:value, weight), …])` entry, two new columns
    are appended:

    - **`{category}`** (int): number of listed `(tag:value)` pairs this row
      matches. Usually 0 or 1; can be ≥ 2 if multiple of the listed pairs
      match for one feature (a feature with both `amenity=school` and
      `school=primary` for a `schools` category that lists both).
    - **`{category}{weight_suffix}`** (float): sum of weights across all
      matching pairs. Equal to the count if every weight is 1.

    Features matching no category at all are dropped if
    `drop_unmatched=True` (the typical case — saves carrying around OSM
    features that aren't of interest).

    Args:
        pois: GeoDataFrame containing the OSM tag columns referenced by
            `category_map` (e.g. `amenity`, `shop`, `leisure`). Tags missing
            from the DataFrame are silently treated as never-matching, so
            partial input works.
        category_map: `{category -> [(tag:value, weight), …]}`.
        weight_suffix: suffix for the per-category weight column. Default
            `'_weight'`. Set to e.g. `'_w'` for shorter columns.
        drop_unmatched: drop rows matching no listed `(tag:value)` pair.
            Default `True`.

    Returns:
        A copy of `pois` with two new columns per category. Original
        columns + index preserved.
    """
    pois = pois.copy()
    count_cols: list[str] = []
    for category, tags in category_map.items():
        weight_col = f'{category}{weight_suffix}'
        if category in pois.columns or weight_col in pois.columns:
            raise ValueError(
                f"Category {category!r} would overwrite an existing column "
                f"(have {category!r} / {weight_col!r}). Rename the category "
                f"or use a different `weight_suffix`.")
        pois[category] = 0
        pois[weight_col] = 0.0
        for tag_pair, weight in tags:
            key, value = tag_pair.split(':', 1)
            if key not in pois.columns:
                continue
            match = pois[key] == value
            pois.loc[match, category] += 1
            pois.loc[match, weight_col] += float(weight)
        count_cols.append(category)
    if drop_unmatched and count_cols:
        any_match = pois[count_cols].sum(axis=1) > 0
        pois = pois[any_match]
    return pois


# ---------------------------------------------------------------------------
# End-to-end fetchers — require osmnx
# ---------------------------------------------------------------------------

def fetch_pois(
    polygon: shapely.geometry.base.BaseGeometry,
    polygon_crs: str,
    category_map: CategoryMap,
    *,
    target_crs: str | None = None,
    use_centroid: bool = True,
    weight_suffix: str = '_weight',
    drop_unmatched: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch OSM POIs within `polygon` and tag them with category columns.

    End-to-end pipeline:
      1. Reproject `polygon` to EPSG:4326 for the osmnx query.
      2. Build the OSM tag query via `osm_tag_query_for_categories`.
      3. Call `osmnx.features_from_polygon`.
      4. Drop non-Point/Polygon geometries (lines, etc.).
      5. (Optional) reduce polygon footprints to point centroids.
      6. (Optional) reproject to `target_crs`.
      7. Categorize via `categorize_pois`.

    Args:
        polygon: shapely polygon (or multipolygon) describing the fetch area.
        polygon_crs: CRS of `polygon` (e.g. `'EPSG:2056'`).
        category_map: `{category -> [(tag:value, weight), …]}`.
        target_crs: optional CRS to project the result to. `None` keeps the
            EPSG:4326 of the OSM source.
        use_centroid: reduce polygon footprints to point centroids. Centroid
            is computed in `target_crs` if given (geometrically meaningful)
            else in EPSG:4326 (warns).
        weight_suffix, drop_unmatched: as in `categorize_pois`.

    Returns:
        GeoDataFrame indexed by OSM ID with the original OSM tag columns
        plus per-category count + weight columns.
    """
    import osmnx as ox

    polygon_4326 = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs('EPSG:4326').iloc[0]
    tags_query = osm_tag_query_for_categories(category_map)
    raw = ox.features_from_polygon(polygon_4326, tags=tags_query)
    raw = raw[raw.geometry.type.isin(['Point', 'Polygon', 'MultiPolygon'])]
    if target_crs is not None:
        raw = raw.to_crs(target_crs)
    if use_centroid:
        # Now (after the optional reprojection) centroid is computed in
        # target_crs (typically metric) — geometrically meaningful.
        raw = raw.copy()
        raw['geometry'] = raw.geometry.centroid
    return categorize_pois(
        raw, category_map,
        weight_suffix=weight_suffix,
        drop_unmatched=drop_unmatched,
    )


def fetch_network(
    polygon: shapely.geometry.base.BaseGeometry,
    polygon_crs: str,
    network_type: str,
    *,
    target_crs: str | None = None,
    simplify: bool = True,
) -> nx.MultiDiGraph:
    """Fetch an OSM network within `polygon` and project to `target_crs`.

    One-line wrapper around `osmnx.graph_from_polygon` + `project_graph`.
    Removes the standard reproject-polygon-to-4326, fetch, reproject-graph
    boilerplate.

    Args:
        polygon: shapely polygon describing the fetch area.
        polygon_crs: CRS of `polygon`.
        network_type: passed through to `osmnx.graph_from_polygon` —
            `'walk'`, `'bike'`, `'drive'`, `'all'`, `'all_public'`, etc.
        target_crs: optional CRS to project the resulting graph to. `None`
            keeps EPSG:4326.
        simplify: passed through. `True` (default) consolidates degree-2
            nodes — usually what you want for routing.

    Returns:
        `networkx.MultiDiGraph`. Nodes carry `x` / `y` attributes in
        `target_crs` (or EPSG:4326 if no target).
    """
    import osmnx as ox

    polygon_4326 = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs('EPSG:4326').iloc[0]
    graph = ox.graph_from_polygon(
        polygon_4326, network_type=network_type, simplify=simplify,
    )
    if target_crs is not None:
        graph = ox.project_graph(graph, to_crs=target_crs)
    return graph
