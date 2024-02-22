import os
import sys
from pathlib import Path
import importlib.util

sys.path.insert(0, os.path.abspath('..'))

project = 'BinPan'
copyright = '2022, Fernando Alfonso'
author = 'Fernando Alfonso'

root_path = Path("__file__").parent.absolute().parent.absolute()
secret_path = os.path.join(root_path, "secret.py")
# print(secret_path)
spec = importlib.util.spec_from_file_location("module.name", secret_path)
my_secret = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = my_secret
spec.loader.exec_module(my_secret)
version = "0.8.13"
release = version

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

html_theme = 'shibuya'

html_static_path = ['_static']


def setup(app):
    app.add_css_file('style.css')
