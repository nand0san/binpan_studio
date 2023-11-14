# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
import importlib.util

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'BinPan'
copyright = '2022, Fernando Alfonso'
author = 'Fernando Alfonso'

# The full version, including alpha/beta/rc tags
# current_path = os.getcwd()
# sys.path.append(current_path)
# parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# sys.path.append(parent)

root_path = Path("__file__").parent.absolute().parent.absolute()
secret_path = os.path.join(root_path, "secret.py")
# print(secret_path)
spec = importlib.util.spec_from_file_location("module.name", secret_path)
my_secret = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = my_secret
spec.loader.exec_module(my_secret)
version = "0.7.13"
release = version

# config = dotenv_values("version.env")
#
# release = config["BINPAN_VERSION"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Create neat summary tables
    'autodocsumm',
]
#     "sphinx.ext.napoleon",

autodoc_member_order = 'bysource'
autodoc_default_options = {'autosummary': True}
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints', 'secret.py', '.log']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# import sphinx_rtd_theme

# html_theme = 'sphinx_rtd_theme'
html_theme = 'shibuya'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    app.add_css_file('style.css')
