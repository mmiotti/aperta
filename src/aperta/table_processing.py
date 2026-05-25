import pandas as pd
import geopandas as gpd
import numpy as np
import logging


def add_internal_distance(df: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame | gpd.GeoDataFrame:
    """Add travel distance for when origin and destination lie within same geographical unit."""
    df['internal_distance'] = np.sqrt(df.area_m2_geometry)
    return df


def add_group_id(df: pd.DataFrame, id_col: str, out_col: str = 'group_id') -> pd.DataFrame:
    """Add `out_col` with sequential integer IDs based on unique values of `id_col`.

    Uses `pd.factorize` — O(n) and avoids a materialised replacement dict.
    """
    codes, _ = pd.factorize(df[id_col], sort=False)
    df[out_col] = codes
    return df


def filter_columns(df: pd.DataFrame | gpd.GeoDataFrame,
                   col_types: str | list[str] | None = None,
                   prefixes: str | list[str] | tuple[str, ...] | None = None,
                   suffixes: str | list[str] | tuple[str, ...] | None = None,
                   contains: str | list[str] | tuple[str, ...] | None = None) -> list[str]:

    cols = df.columns.tolist()
    if col_types:
        if isinstance(col_types, str):
            col_types = [col_types]
        for col_type in col_types:
            if col_type == 'numeric':
                cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
            elif col_type == 'string':
                cols = [col for col in cols if pd.api.types.is_string_dtype(df[col])]
            elif col_type == 'int':
                cols = [col for col in cols if pd.api.types.is_integer_dtype(df[col])]
            elif col_type == 'float':
                cols = [col for col in cols if pd.api.types.is_float_dtype(df[col])]
            else:
                raise ValueError(f"Unknown column type `{col_type}`")
    if prefixes:
        cols = [col for col in cols if col.startswith(tuple(prefixes))]
    if suffixes:
        cols = [col for col in cols if col.startswith(tuple(suffixes))]
    if contains:
        if isinstance(contains, str):
            contains = [contains]
        for s in contains:
            cols = [col for col in cols if s in col]
    return cols


def get_col_agg_fns(col_list: list[str]) -> dict[str, callable]:
    """Return dictionary with values being correct aggregation callable for each metric (key) in list."""

    fns = {}
    for col in col_list:
        if col == 'geometry' or col.endswith('_id'):
            fns[col] = 'first'
        elif col.startswith(('population_', 'employment_', 'combined_', 'poi_', 'mobility_')):
            fns[col] = 'sum'
        elif 'centroid' in col:
            fns[col] = 'mean'
        elif '_std' in col or '_avg' in col or 'topography' in col:
            fns[col] = 'mean'
        elif '_flag' in col:
            fns[col] = 'max'
    return fns


def aggregate(
    df: pd.DataFrame | gpd.GeoDataFrame,
    by: str | list[str] | None = None,
    level: int | list[int] | None = None,
    agg_cols: list[str] | None = None,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Aggregate dataframe by `by`, automatically choosing agg functions based on column names by default.

    Works with GeoDataFrames (first geometry of aggregated cells will be returned).
    """

    if not agg_cols:
        agg_cols = [col for col in df.columns if col != df.index.name]
    fns = get_col_agg_fns(agg_cols)
    res = df.groupby(by=by, level=level).agg(fns)
    if 'geometry' in res.columns:
        res = res.set_geometry('geometry', crs=df.crs)
    return res


def upcast(
    from_df: pd.DataFrame,
    to_df: pd.DataFrame,
    metrics: list[str],
    missing: int | float,
) -> pd.DataFrame:
    """Aggregate metrics in `from_df` by index of `to_df` and add aggregated values to `to_df`."""

    if any([metric in to_df.columns for metric in metrics]):
        logging.warning(f"At least one metric from `metrics` is already present in `to_df`: "
                        f"{[metric for metric in metrics if metric in to_df.columns]}")
        metrics = [metric for metric in metrics if metric not in to_df.columns]
        if len(metrics) == 0:
            return to_df

    agg_fn = get_col_agg_fns(metrics)
    by = to_df.index.name
    res = from_df.groupby(by).agg(agg_fn)
    to_df = to_df.join(res[metrics])
    # TODO: could this be better integrated with other auto-rules-based aggregation stuff?
    for metric in metrics:
        if pd.api.types.is_integer_dtype(from_df[metric]):
            if agg_fn[metric] in (max, 'max'):
                fill_value = 0
            elif agg_fn[metric] in (min, 'min'):
                fill_value = missing
            else:
                fill_value = -1
            to_df[metric] = to_df[metric].fillna(fill_value).astype(int)
    return to_df


def restore_integer_columns(
    new_df: pd.DataFrame,
    old_df: pd.DataFrame,
    nan_value: int,
) -> pd.DataFrame:
    """Make sure integer columns in `old_df` are also integers in `new_df`, filling NaNs with `nan_value`.

    Fills float columns with nan_value as well. This function is usually used after applying some kind of join that
    can introduce NaNs.

    The index of `old_df` is excluded (otherwise, if old_df has an integer-based index, it would also be filled with
    nan_value, which may be unintended).

    TODO: separate nan_values for int/float, rename function to restore_original_dtypes or similar
    """
    for col in new_df.columns:
        if col in old_df.columns and col != old_df.index.name:
            if pd.api.types.is_integer_dtype(old_df[col]):
                new_df[col] = np.round(new_df[col].fillna(nan_value)).astype(int)
            elif pd.api.types.is_float_dtype(old_df[col]):
                new_df[col] = new_df[col].fillna(nan_value).astype(float)
    return new_df


def get_available_metrics(
    df: pd.DataFrame | gpd.GeoDataFrame,
    operation_type: str | None = None,
    included_groups: list[str] | None = None,
) -> list[str]:
    """Get list of columns of all metrics present in DataFrame or GeoDataFrame, identified by column name prefix.

    Data can either be DataFrame or class Properties (containing a DataFrame).
    """

    if operation_type is not None and operation_type not in ('sum', 'mean'):
        raise ValueError(f"Cannot get available metrics in data for unknown operation_type `{operation_type}`")
    if included_groups is None:
        included_groups = ('people', 'buildings', 'mobility', 'poi', 'nw')
    else:
        included_groups = tuple(included_groups)
    cols = []

    sums_prefix = ('people', 'area')
    sums_suffix = ('total', 'count')

    for col in df.columns:
        if operation_type == 'sum':
            if col.startswith(included_groups) and (col.startswith(sums_prefix) or col.endswith(sums_suffix)):
                cols.append(col)
        elif operation_type == 'mean':
            if '_per_' in col:
                cols.append(col)
        elif operation_type is None:
            if col.startswith(included_groups):
                cols.append(col)
    return cols
