"""Top-level package for digneapy."""

__author__ = """Alejandro Marrero"""
__email__ = "amarrerd@ull.edu.es"
__version__ = "0.2.5"


from . import _core, archives, domains, operators
from ._core import (
    DescriptorFn,
    DescriptorKey,
    DescriptorPipeline,
    Domain,
    Instance,
    Problem,
    Solution,
    Solver,
    descriptors,
    descriptors_registry,
    scores,
)
from ._core._metrics import Logbook, Statistics, qd_score, qd_score_auc
from ._core.scores import PerformanceFn, maximise_perf_gap_easy, maximise_runtime_gap
from .archives import Archive, CVTArchive, GridArchive, UnstructuredArchive
from .generators import ES, BaseGenerator, Dominated, Evolutionary, GenResult, MapElites
from .transformers import Transformer
from .typing import Direction, IndType

__dignea_submodules = {"utils", "generators", "solvers", "visualize"}


__all__ = list(
    __dignea_submodules
    | set(_core.__all__)
    | set(operators.__all__)
    | set(archives.__all__)
    | set(scores.__all__)
    | set(domains.__all__)
)
