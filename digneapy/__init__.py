"""Top-level package for digneapy."""

__author__ = """Alejandro Marrero"""
__email__ = "amarrerd@ull.edu.es"
__version__ = "0.2.5"


from . import _core, archives, domains, operators
from ._core import (
    NS,
    DescriptorFn,
    DescriptorKey,
    DescriptorPipeline,
    Direction,
    Domain,
    IndType,
    Instance,
    Problem,
    Solution,
    Solver,
    descriptors,
    descriptors_registry,
    dominated_novelty_search,
    scores,
)
from ._core._metrics import Logbook, Statistics, qd_score, qd_score_auc
from ._core.scores import PerformanceFn, max_gap_target, runtime_score
from .archives import Archive, CVTArchive, GridArchive
from .generators import BaseGenerator, Dominated, Evolutionary, GenResult, MapElites
from .transformers import Transformer

__dignea_submodules = {"utils", "generators", "solvers", "visualize"}


__all__ = list(
    __dignea_submodules
    | set(_core.__all__)
    | set(operators.__all__)
    | set(archives.__all__)
    | set(scores.__all__)
    | set(domains.__all__)
)
