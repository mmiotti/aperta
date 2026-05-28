"""Aperta-specific exception types.

These are raised from across the library when an error condition is specific
to aperta (rather than a generic Python error like `ValueError`). Catching
these by type lets calling code distinguish aperta failures from upstream
library failures or programming bugs.
"""


class ContextError(Exception):
    """Raised when the optional Context layer is misconfigured or misused."""


class DataError(Exception):
    """Raised when input data is malformed, incomplete, or internally inconsistent."""


class ProcessingError(Exception):
    """Raised when a processing step fails for a reason specific to aperta's logic."""
