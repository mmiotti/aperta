"""Tabular helpers that flow through aperta's (Geo)DataFrame pipeline.

Small composable column-level transforms — dedupe an index after a join,
add a straight-line origin→destination distance, take a weighted mean per
group. None of these reason about geometry as the primary concern (for
that see `geo_processing`); they operate on tables that may or may not
carry a geometry column.
"""

import numpy as np
import pandas as pd


def remove_duplicate_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate indices from DataFrame or Series. Often useful after spatial join."""
    df = df.loc[~df.index.duplicated(keep="first"), :]
    return df


def add_straight_line_dist(
    df: pd.DataFrame,
    orig_prefix: str = "orig",
    dest_prefix: str = "dest",
    out_col: str = "dist_line",
) -> pd.DataFrame:
    """Add euclidean origin→destination distance, in CRS units (typically metres)."""
    dx = df[f"{orig_prefix}_x"] - df[f"{dest_prefix}_x"]
    dy = df[f"{orig_prefix}_y"] - df[f"{dest_prefix}_y"]
    df[out_col] = np.hypot(dx, dy)
    return df


def weighted_group_mean(
    values: pd.Series,
    weights: pd.Series,
    group_id: pd.Series,
) -> pd.Series:
    """Weighted mean of `values` per group, indexed by group ID.

    NaN-aware: rows with NaN in either `values` or `weights` are dropped
    before aggregation. Groups with no surviving rows (or with all weights
    ≤ 0) yield NaN.

    All three inputs must share the same index (per-row alignment); the
    result is indexed by the unique group IDs. Typical use case: reduce
    per-cell values to per-zone (`group_id = cells['zone_id']`,
    `weights = cells['population']` or similar).

    Args:
        values: per-row numeric Series.
        weights: per-row non-negative weight Series (same index as `values`).
        group_id: per-row group-membership Series (same index as `values`).
            Values become the result's index.

    Returns:
        `pd.Series` indexed by the unique group IDs, with the weighted
        mean of `values` per group.
    """
    valid = values.notna() & weights.notna()
    eff_w = weights.where(valid, 0.0)
    eff_v = values.where(valid, 0.0)
    num = (eff_v * eff_w).groupby(group_id).sum()
    den = eff_w.groupby(group_id).sum()
    return num / den.where(den > 0)
