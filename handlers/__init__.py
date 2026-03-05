"""
Handlers package.

Submodules are loaded lazily on first access (e.g. ``handlers.plotting``).
"""
import importlib


def __getattr__(name):
    try:
        return importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
