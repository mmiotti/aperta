import logging
import math
import sys
import time

import numpy as np
import pandas as pd

from typing import NamedTuple


def tracked_namedtuple(labels_and_types: list[tuple[str, type]]) -> type:
    """Factory for creating named tuples that track access to its fields through self._accessed_fields.

    There may be some unexpected behavior when using a debugger/IDE, which may access object attributes outside the
    actual code. This implementation using named tuples works better in PyCharm than an implementation using
    (frozen) data classes, as starting the debug console will access all attributes of any instances of the latter.

    This feature is used to keep track of dependencies (more specifically, flagging outdated output data if input
    parameters that were used to create that data have changed). False positives are therefore not a big deal; they
    will mostly lead to false positive dependency flags.
    """

    nt = NamedTuple('nt', labels_and_types)

    class TrackedNamedTuple(nt):
        __slots__ = ()
        def __new__(cls, *args, **kwargs):
            cls._accessed_fields = set()
            return super().__new__(cls, *args, **kwargs)
        def __getattribute__(self, key):
            if key != '_fields' and key in self._fields:
                self._accessed_fields.add(key)
            return super().__getattribute__(key)
        def used_fields_as_dict(self) -> dict:
            return {k: v for k, v in self._asdict().items() if k in self._accessed_fields}

    return TrackedNamedTuple


def timeit(fn):
    def timed(*args, **kw):
        t1 = time.perf_counter()
        result = fn(*args, **kw)
        t2 = time.perf_counter()
        logging.info(f'Function `{fn.__name__}` completed; it took {t2 - t1:.1f} seconds.')
        return result

    return timed


def recursive_flatten_list(a: list) -> list:
    """Flatten list recursively.

    From https://stackoverflow.com/questions/12472338/flattening-a-list-recursively.
    """

    if not a:
        return a
    if isinstance(a[0], list):
        return recursive_flatten_list(a[0]) + recursive_flatten_list(a[1:])
    return a[:1] + recursive_flatten_list(a[1:])


def simple_flatten_list(a: list) -> list:
    """recursive_flatten_list can cause 'recursion depth exceeded' error, so we can also use this non-recursive version."""
    b = []
    for x in a:
        if isinstance(x, list):
            b += x
        else:
            b.append(x)
    return b


def round_to_significant_figures(x, n):
    return x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def gdf_memory_mb(gdf) -> float:
    """Get approximate size of geometry column in GeoDataFrame in memory, in MB."""

    return np.sum([sys.getsizeof(poly.wkb) for poly in gdf.geometry]) / 1e6


def most_common(a: pd.Series) -> any:
    return a.value_counts().index[0]


def get_weighted_agg_function(df: pd.DataFrame,
                              weight_name: str,
                              allow_ignore_weights: bool = False,
                              fill_value: float | None = None):
    """Get a lambda function that can be used to calculate weighted averages in Pandas groupby().agg() patterns.
    
    If allow_ignore_weights is True, weights are set to 1 if they sum to zero otherwise.

    If fill_value is given, fill_value is returned by function if weights sum to zero.
    
    If neither fill_value is given nor allow_ignore_weights is true, an Error is raised by function
    if weights sum to zero.

    TODO: also implement bounds (in get_weighted_agg_function_bounded below) into this function
    """
    if fill_value is None and not allow_ignore_weights:
        wm = lambda x: np.average(x, weights=df.loc[x.index, weight_name])
    else:
        def wm(x):
            w = df.loc[x.index, weight_name]
            if w.sum() == 0:
                if allow_ignore_weights:
                    return np.average(x)
                else:
                    return fill_value
            return np.average(x, weights=w)
    return wm


def get_weighted_agg_function_bounded(
    df: pd.DataFrame,
    weight_name: str,
    upper_bound: int | float,
    lower_bound: int | float = 0,
):
    """Same as get_weighted_agg_function, but with bounds. Returns nan if no entries match bounds."""

    def fn(x):
        f = (x >= lower_bound) & (x < upper_bound)
        w = df.loc[x.index, weight_name][f]
        if f.sum() == 0 or w.sum() == 0:
            return np.nan
        return np.average(x[f], weights=w)
    return fn


def _p_stars_str(value: float) -> str:
    if value < 0.001:
        return '***'
    elif value < 0.01:
        return '**'
    elif value < 0.1:
        return '*'
    return ''


def sm_results_table_as_df(sm_result) -> pd.DataFrame:
    """statsmodels result → tidy DataFrame with `coef` and `p` (p-value stars).

    Currently only used by legacy `src/uma_access/`; safe to delete when the
    uma_* tree is removed.
    """
    p_name = 'P>|t|' if 'P>|t|' in sm_result.summary2().tables[1].columns else 'P>|z|'
    tbl = sm_result.summary2().tables[1][['Coef.', p_name]]
    tbl = tbl.rename(columns={'Coef.': 'coef', p_name: 'p'})
    tbl['p'] = tbl['p'].apply(_p_stars_str)
    return tbl


def length_weighted_quantile(values, weights, q: float) -> float:
    """Weighted quantile via cumulative weights.

    For per-edge stats (`q` along the cumulative *length* / *weight* of
    edges, not edge count). Useful for "what speed does a random metre of
    travel see" rather than "what speed does a random edge have", which is
    often dominated by short residential edges.

    Args:
        values: 1-D array-like of per-edge values (e.g. speeds, slopes).
        weights: 1-D array-like of per-edge weights (e.g. edge lengths);
            must be the same length as `values` and sum to > 0.
        q: quantile in [0, 1].

    Returns:
        The value at the smallest cumulative weight-fraction ≥ `q`.
    """
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cum, q)])
