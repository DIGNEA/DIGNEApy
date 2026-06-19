"""Top-level package for digneapy."""

__author__ = """Alejandro Marrero"""
__email__ = "amarrerd@ull.edu.es"
__version__ = "0.3.1"


from digneapy import core, domains, generators, solvers

__all__ = ["core", "domains", "generators", "solvers"]


# Lazy import function
def __getattr__(attr_name):
    _submodules = {"visualize", "utils", "transformers"}

    import importlib
    import sys

    if attr_name in _submodules:
        full_name = f"digneapy.{attr_name}"
        submodule = importlib.import_module(full_name)
        sys.modules[full_name] = submodule
        return submodule

    else:
        raise ImportError(f"module digneapy has no attribute {attr_name}")
