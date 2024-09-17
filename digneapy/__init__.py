"""Top-level package for digneapy."""

__author__ = """Alejandro Marrero"""
__email__ = "amarrerd@ull.edu.es"
__version__ = "0.2.3"

from digneapy._core import (
    NS,
    Direction,
    Domain,
    IndType,
    Instance,
    Problem,
    Solution,
    Solver,
)
from digneapy.archives import Archive, GridArchive
from digneapy.operators import crossover, mutation, replacement, selection

__dignea_submodules = {"utils", "domains", "generators", "solvers"}


__all__ = list(
    __dignea_submodules
    | set(_core.__all__)
    | set(operators.__all__)
    | set(archives.__all__)
)


# Lazy import function
def __getattr__(name):
    if name == "transformers":
        import digneapy.transformers as transformers

        return transformers
    elif name == "domains":
        import digneapy.domains as domains

        return domains
    elif name == "generators":
        import digneapy.generators as generators

        return generators

    elif name == "solvers":
        import digneapy.solvers as solvers

        return solvers

    elif name == "utils":
        import digneapy.utils as utils

        return utils

    else:
        raise ImportError(f"module {__name__} has no attribute {name}")
