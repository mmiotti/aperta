"""
Registry of geographical units recognized by `aperta.data` for shapes, properties,
and ODMs.

Five canonical units, organized in two groups:

  AGGREGATION units (`cells`, `zones`, `regions`):
      The library-canonical hierarchy. Input data (land use, car ownership rates,
      etc.) maps to these. Each transition `cells → zones → regions` is a
      many-to-one aggregation.

  NETWORK units (`nodes`, `edges`):
      Belong to a specific network instance (driving, walking, transit). Names
      are universal even though the underlying graphs may differ per mode.

Anything else (municipalities, cantons, buildings, locations, …) is *project-
specific source data* that gets mapped into one of the five canonical units
during preparation — usually in `preparation/<country>/...` or in the project's
first prep script. The library knows about the five and only the five.

Each unit declares its **id_col** — the canonical name of the ID column / index
name for DataFrames keyed by that unit. Convention is "singular + _id" (e.g.
`cells → cell_id`), declared explicitly per unit. `aperta.data` enforces the
mapping at I/O time: `df.index.name` must match the registered `id_col`.
"""
import logging

from aperta.errors import DataError


CANONICAL_UNITS: dict[str, dict[str, str]] = {
    'cells':   {'tier': 'aggregation', 'id_col': 'cell_id',
                'description': "Finest spatial unit (typically 100-150m square or hexagonal grid). "
                               "The smallest analysis unit and the only level routed cell-to-cell."},
    'zones':   {'tier': 'aggregation', 'id_col': 'zone_id',
                'description': "Intermediate aggregation — one zone contains many cells. "
                               "Often corresponds to traffic analysis zones (TAZ) in transport modeling, "
                               "but any sensible mid-scale partition of the study area works "
                               "(e.g. census tracts, municipalities)."},
    'regions': {'tier': 'aggregation', 'id_col': 'region_id',
                'description': "Coarsest aggregation — one region contains many zones. "
                               "Often a political / statistical macro-unit (cantons in Switzerland, "
                               "NUTS-2 in Europe). Used as the fallback resolution for long-distance "
                               "OD pairs where cell- or zone-level precision isn't needed."},
    'nodes':   {'tier': 'network',     'id_col': 'node_id',
                'description': "Network nodes (graph vertices) — typically OSM nodes for the mode's "
                               "routable network."},
    'edges':   {'tier': 'network',     'id_col': 'edge_id',
                'description': "Network edges (graph edges) — typically OSM ways for the mode's "
                               "routable network."},
}

AGGREGATION_HIERARCHY: list[str] = ['cells', 'zones', 'regions']


def is_known(unit: str) -> bool:
    """True iff `unit` is a canonical aperta unit."""
    return unit in CANONICAL_UNITS


def tier_of(unit: str) -> str | None:
    """Return `'aggregation'` or `'network'` for a known unit, else `None`."""
    return CANONICAL_UNITS[unit]['tier'] if unit in CANONICAL_UNITS else None


def description_of(unit: str) -> str | None:
    """Human-readable description of `unit`, or `None` if unknown."""
    return CANONICAL_UNITS[unit]['description'] if unit in CANONICAL_UNITS else None


def id_col(unit: str) -> str:
    """ID column / index name for `unit`. Raises `DataError` on unknown units."""
    if unit not in CANONICAL_UNITS:
        raise DataError(
            f"No registered id_col for geo_unit '{unit}'. "
            f"Known units: {sorted(CANONICAL_UNITS)}.")
    return CANONICAL_UNITS[unit]['id_col']


def unit_for_id_col(col: str) -> str | None:
    """Reverse lookup: unit whose `id_col` is `col`, else `None`.

    Used by `aperta.data` to derive the geo_type from a DataFrame's index name.
    """
    for name, meta in CANONICAL_UNITS.items():
        if meta['id_col'] == col:
            return name
    return None


def warn_if_unknown(unit: str, where: str = '') -> None:
    """Log a warning if `unit` is not canonical. Doesn't raise — preserves
    flexibility for ad-hoc geo_types — but flags typos at the call site.
    """
    if is_known(unit):
        return
    suffix = f" in {where}" if where else ""
    logging.warning(
        f"Unknown geo_unit '{unit}'{suffix}. "
        f"Known: {sorted(CANONICAL_UNITS)}. "
        f"Map your source data to one of these in preparation, or add a new "
        f"canonical unit to `aperta.geo_units.CANONICAL_UNITS` if generally useful.")
