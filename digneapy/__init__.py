"""Top-level package for digneapy."""

__author__ = """Alejandro Marrero"""
__email__ = "amarrerd@ull.edu.es"
__version__ = "0.2.5"


from . import _core, archives, domains, operators
from ._core import (
    NS,
    Direction,
    Domain,
    IndType,
    Instance,
    P,
    Problem,
    Solution,
    Solver,
    SupportsSolve,
    descriptors,
    scores,
)
from ._core.descriptors import DESCRIPTORS, DescStrategy, descriptor
from ._core.scores import PerformanceFn, max_gap_target, runtime_score
from .archives import Archive, CVTArchive, GridArchive

__dignea_submodules = {"utils", "generators", "solvers", "visualize"}


__all__ = list(
    __dignea_submodules
    | set(_core.__all__)
    | set(operators.__all__)
    | set(archives.__all__)
    | set(descriptors.__all__)
    | set(scores.__all__)
    | set(domains.__all__)
)


# Lazy import function
def __getattr__(attr_name):
    import importlib
    import sys

    if attr_name in __dignea_submodules:
        full_name = f"digneapy.{attr_name}"
        submodule = importlib.import_module(full_name)
        sys.modules[full_name] = submodule
        return submodule

    else:
        raise ImportError(f"module {__name__} has no attribute {attr_name}")
