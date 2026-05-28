"""Sphinx configuration for the aperta documentation site."""

from importlib.metadata import version as _pkg_version

# -- Project information -----------------------------------------------------

project = "aperta"
author = "Marco Miotti"
copyright = "2026, Marco Miotti"

# Pull version from installed package metadata so docs and pyproject stay in sync.
try:
    release = _pkg_version("aperta")
except Exception:
    release = "unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # pulls docstrings from importable modules
    "sphinx.ext.autosummary",   # generates summary tables for autodoc entries
    "sphinx.ext.napoleon",      # parses Google-style docstrings
    "sphinx.ext.intersphinx",   # cross-link to numpy / pandas / networkx docs
    "sphinx.ext.viewcode",      # adds [source] links to each function
    "myst_parser",              # markdown support for narrative pages
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"aperta {release}"
html_static_path: list[str] = []

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"   # render type hints in the description, not signatures
autodoc_typehints_format = "short"  # `dict` not `typing.Dict`
autoclass_content = "class"         # docstring from __init__ + class merged

# -- Napoleon (Google-style docstrings) -------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_ivar = True  # render Attributes as :ivar — avoids duplicate-doc warning for dataclasses

# -- MyST (markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives
    "deflist",       # definition lists
    "smartquotes",   # typographic quotes
]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Misc --------------------------------------------------------------------

# Treat warnings as errors locally? Useful but noisy on first build; leave off
# until the build is clean.
# nitpicky = True
