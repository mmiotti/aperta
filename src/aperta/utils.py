"""Small generic utilities used across aperta — nothing domain-specific.

Currently exposes just the `timeit` decorator (used to instrument long-running
pipeline phases).
"""

import logging
import time


def timeit(fn):
    """Decorator that logs each call's wall-clock duration at INFO level.

    Wraps `fn` so that every invocation prints `Function 'fn_name' completed;
    it took X.X seconds.` to the standard logger. Used to instrument
    long-running pipeline steps without scattering manual timing code.
    """

    def timed(*args, **kw):
        t1 = time.perf_counter()
        result = fn(*args, **kw)
        t2 = time.perf_counter()
        logging.info(f"Function `{fn.__name__}` completed; it took {t2 - t1:.1f} seconds.")
        return result

    return timed
