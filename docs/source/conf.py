import os
import sys
import datetime

sys.path.insert(
    0, os.path.abspath("../../src/extremeweatherbench")
)  # Source code dir relative to this file

# -- Project information
utc_now = datetime.datetime.now(datetime.UTC).strftime("%H:%M %d %b %Y")
project = "ExtremeWeatherBench"
copyright = f"{datetime.datetime.now(datetime.UTC).strftime('%Y')}, Brightband.    ♻ Updated: {utc_now}"
author = "Taylor Mandelbaum"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output
html_static_path = ["_static"]
html_theme = "alabaster"
html_css_files = ["_static/custom.css"]
# -- Options for EPUB output
epub_show_urls = "footnote"

autosummary_generate = True
autodoc_inherit_docstrings = True
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
