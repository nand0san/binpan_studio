import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'BinPan'
copyright = '2022, Fernando Alfonso'
author = 'Fernando Alfonso'

# Las credenciales las gestiona panzer (~/.panzer_creds); la doc no necesita cargar nada.
version = "0.10.0"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autodocsumm",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Backend de storage opcional (no va en requirements.txt): se mockea para que autodoc
# no falle al importarlo en el runner de CI. redis no se mockea: usa StrictRedis en una
# union de tipos evaluada en import y el mock rompe el operador `|`; queda un warning benigno.
autodoc_mock_imports = ['influxdb_client']

autodoc_member_order = 'bysource'
autodoc_default_options = {'autosummary': True}
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints', 'secret.py', '.log']

html_theme = 'shibuya'

html_static_path = ['_static']


def setup(app):
    app.add_css_file('style.css')
