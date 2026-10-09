# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------

project = 'Deep Cartograph'
copyright = '2026, Nostrum Biodiscovery'
author = 'Nostrum Biodiscovery'

# The full version, including alpha/beta/rc tags.
# Maintained by scripts/bump_version.py — do not edit by hand.
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon',
              'sphinx.ext.autosectionlabel',
              'sphinx_copybutton', 'myst_parser']

# Prefix autosection labels with the document name so repeated section titles
# (Usage, Options, Output) across tool pages don't collide.
autosectionlabel_prefix_document = True

# MyST (Markdown) extensions: colon_fence enables ::: admonitions, deflist
# enables definition lists.
myst_enable_extensions = ['colon_fence', 'deflist']

# Generate anchors for headings up to level 3, so tool pages can link to
# sections such as concepts.html#collective-variables.
myst_heading_anchors = 3

# Build outputs and local environments are not documentation sources.
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = f"Deep Cartograph {release}"
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_sphinx = False

# Nostrum logo in the sidebar (top of every page); theme-aware light/dark variants.
html_theme_options = {
    "light_logo": "nostrum_logo.png",
    "dark_logo": "nostrum_logo_white.png",
}

# -- Epilog -------------------------------------------------------------------
rst_epilog = """

----

*Last update:* |date|


.. |Product| replace:: Deep Cartograph
.. |date| date::
"""
