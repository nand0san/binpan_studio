import sys
from pathlib import Path

# Permite que autodoc encuentre el paquete (un nivel arriba de docs/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importa la versión desde el paquete, no la hardcodes.
from binpan.version import __version__  # noqa: E402

project = 'BinPan'
copyright = '2022, Fernando Alfonso'
author = 'Fernando Alfonso'

version = __version__
release = __version__

# Extensiones mínimas. Sin sphinx.ext.napoleon: los docstrings son reST nativo
# compacto (estilo :param:), no Google ni NumPy.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autodocsumm",
]

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

add_module_names = False  # Evita prefijos largos del tipo "binpan.indicators.ema()".

html_theme = 'shibuya'
html_static_path = ['_static']
html_title = f'{project} {release}'
html_show_sourcelink = False

html_theme_options = {
    'github_url': 'https://github.com/nand0san/binpan_studio',
    'accent_color': 'iris',
}


def setup(app):
    app.add_css_file('style.css')
