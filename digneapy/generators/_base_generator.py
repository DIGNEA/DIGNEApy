#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _base_generator.py
@Time    :   2026/03/25 11:22:20
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Mapping, Optional, Tuple

import numpy as np
import polars as pl

from digneapy._core.descriptors import DescriptorPipeline
from digneapy.generators._utils import extract_solvers_name

from .._core import (
    Domain,
    Instance,
    Logbook,
    PerformanceFn,
    Solver,
    Statistics,
    maximise_perf_gap_easy,
)
from ..archives import Archive


@dataclass
class GenerationResult:
    """Container for the outputs produced by a single run of a generator.

    This dataclass bundles together everything needed to inspect or reproduce
    a generation run: the solvers involved, the resulting instances, the
    evolutionary history, and a set of computed descriptive metrics.

    Metrics are computed automatically in :meth:`__post_init__` whenever the
    ``instances`` collection is non-empty, so callers do not need to invoke
    ``Statistics`` manually after constructing this object.

    Attributes:
        solvers (Sequence[str]): Names of the solvers used to evaluate the
            generated instances.
        instances (Archive | np.ndarray | Sequence[Instance]): The collection
            of instances produced by the generator. Can be an ``Archive``, a
            raw NumPy array, or any sequence of ``Instance`` objects.
        history (Logbook): Logbook recording the evolution of the generator's
            internal state (e.g. fitness, diversity) across generations.
        metrics (pl.DataFrame | Mapping, optional): Descriptive statistics
            computed over ``instances``. Automatically populated as a
            ``pl.DataFrame`` after initialisation if ``instances`` is
            non-empty; otherwise left as an empty ``pl.DataFrame``. Defaults
            to an empty ``pl.DataFrame``.
    """

    solvers: Sequence[str]
    instances: Archive | Sequence[Instance]
    history: Logbook
    metrics: pl.DataFrame | Mapping = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Compute descriptive statistics for the generated instances.

        If ``instances`` contains at least one element, ``metrics`` is
        populated by running ``Statistics()`` over the instances and
        requesting a ``pl.DataFrame`` representation. If ``instances`` is
        empty, ``metrics`` is left untouched and an empty ``pl.Series`` is
        assigned to ``self._metrics`` instead (note: this does not overwrite
        the public ``metrics`` field).
        """
        if len(self.instances) != 0:
            self.metrics = Statistics()(instances=self.instances)
        else:
            self._metrics = pl.Series()


class BaseGenerator(ABC):
    """Abstract base class for all Quality-Diversity generators.

    ``BaseGenerator`` factors out the logic that is common to every concrete
    Quality-Diversity (QD) instance generator in Digneapy, so that subclasses
    only need to implement the specific evolutionary loop in ``__call__``.

    Handles:
    - Common evaluation pipeline: running a portfolio of solvers over a
      population of generated instances and aggregating their scores.
    - Descriptor computation: delegated to a configurable
      ``DescriptorPipeline`` used to characterise instances for diversity
      purposes.
    - Result formatting: concrete generators are expected to package their
      output as a :class:`GenerationResult`.
    - Statistics tracking: via the internal ``Logbook``, which records the
      generator's progress across generations.

    Subclasses must implement :meth:`__call__`, which performs the actual
    generation process and returns a :class:`GenerationResult`.
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: np.uint32 | int,
        performance_function: PerformanceFn = maximise_perf_gap_easy,
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        generations: np.uint32 | int = np.uint32(1_000),
        repetitions: np.uint16 | int = np.uint16(1),
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates an instance of BaseGenerator with common attributes for generators

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Sequence[SupportSolve]): Sequence item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            performance_function (PerformanceFn, optional): Performance function to calculate
                    the performance score. Defaults to max_gap_target.
            describe_by (DESCRIPTORS, optional): _Descriptor used to calculate the diversity.
                    The options available are defined in the dictionary digneapy.DESCRIPTORS.
                    Defaults to "features".
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            repetitions (int, optional): Number times a solver in the portfolio must be run
                    over the same instance. Defaults to 1.

        Raises:
            ValueError: If ``pop_size``, ``repetitions``, or ``generations``
                cannot be converted to a positive integer.
            ValueError: If ``domain`` is not an instance of ``Domain``.
            ValueError: If ``portfolio`` is empty or contains any element that
                is not an instance of ``Solver``.
        """
        try:
            _pop_size = int(pop_size)
            _repetitions = int(repetitions)
            _generations = int(generations)
            if any(param <= 0 for param in (_pop_size, _repetitions, _generations)):
                raise ValueError()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"pop_size ({pop_size}), "
                f"repetitions ({repetitions}) and generations ({generations})"
                f"must be positive integers or np.uint. {exc}"
            ) from exc

        if not isinstance(domain, Domain):
            raise ValueError(f"BaseGenerator: Invalid domain {domain}.")

        if len(portfolio) == 0 or any(
            not isinstance(solver, Solver) for solver in portfolio
        ):
            raise ValueError(
                f"BaseGenerator: the portfolio is empty or contains invalid solvers. {portfolio}"
            )

        self._domain = domain
        self._portfolio = tuple(portfolio)
        self._pop_size = _pop_size
        self._population = []
        self._performance_fn = performance_function
        self._descriptor_pipe = descriptor_pipe
        self._generations = _generations
        self._repetitions = _repetitions
        self._logbook = Logbook()
        self._seed = seed

    @property
    def log(self) -> Logbook:
        """Return the ``Logbook`` tracking this generator's evolutionary history.

        Returns:
            Logbook: The logbook instance accumulating per-generation records
                (e.g. fitness statistics, diversity metrics) produced during
                a run of the generator.
        """
        return self._logbook

    @property
    def descriptor_pipeline(self):
        """Return the descriptor pipeline used to characterise instances.

        The pipeline is responsible for computing the descriptors used to
        assess diversity between instances (e.g. feature-based descriptors).

        Returns:
            DescriptorPipeline: The configured descriptor pipeline.
        """
        return self._descriptor_pipe

    def __str__(self):
        """Return a human-readable summary of the generator's configuration.

        Includes the population size, number of generations, domain name, and
        the names of the solvers in the portfolio.

        Returns:
            str: A formatted string describing the generator instance.
        """
        port_names = tuple(extract_solvers_name(self._portfolio))
        domain_name = self._domain.__name__ if self._domain is not None else "None"
        return f"{self.__class__.__name__}(pop_size={self._pop_size},gen={self._generations},domain={domain_name},portfolio={port_names!r})"

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the generator.

        Reuses :meth:`__str__` but swaps the surrounding parentheses for angle
        brackets, following the convention used elsewhere in the framework.

        Returns:
            str: A compact representation suitable for debugging/logging.
        """
        return self.__str__().replace("(", "<").replace(")", ">")

    def _evaluate_population(
        self,
        population: np.ndarray | Sequence[Instance],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the population of instances using the portfolio of solvers.

        For every instance in ``population``, the domain first converts it
        into a concrete ``Problem``. Each solver in the portfolio is then run
        ``self._repetitions`` times against that problem, and the best
        (highest-fitness) solution from each run is kept as that repetition's
        score. The mean score across repetitions is computed per
        instance/solver pair, and the configured performance function is
        applied to these mean scores to produce a single performance bias
        per instance.

        Note:
            This implementation assumes each solver call returns a sequence
            of candidate solutions, from which the best (by ``fitness``) is
            selected via ``max(..., key=attrgetter("fitness"))``. This is
            compatible with solvers that return a single solution (e.g.
            Pisinger-style algorithms or simple heuristics), since selecting
            the maximum over a one-element sequence is a no-op.

        Args:
            population (np.ndarray | List[Instance]): Collection of instances
                to evaluate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A two-element tuple containing:
                - performance_biases (np.ndarray): The per-instance
                  performance bias as computed by ``self._performance_fn``
                  over the mean solver scores.
                - solvers_scores (np.ndarray): The raw scores array of shape
                  ``(len(population), len(self._portfolio), self._repetitions)``,
                  containing the best fitness obtained by each solver on each
                  instance for every repetition.
        """
        solvers_scores = np.zeros(
            shape=(len(population), len(self._portfolio), self._repetitions)
        )
        problems_to_solve = self._domain.generate_problems_from_instances(population)
        for j, problem in enumerate(problems_to_solve):
            for i, solver in enumerate(self._portfolio):
                # There is no need to change anything in the evaluation code when using Pisinger solvers
                # because the algs. only return one solution per run (len(solutions) == 1)
                # The same happens with the simple KP heuristics. However, when using Pisinger solvers
                # the lower the running time the better they're considered to work an instance
                scores = np.zeros(self._repetitions)
                for rep in range(self._repetitions):
                    scores[rep] = max(
                        solver(problem), key=attrgetter("fitness")
                    ).fitness

                solvers_scores[j, i, :] = scores

        mean_solvers_scores = np.mean(solvers_scores, axis=2)
        performance_biases = self._performance_fn(mean_solvers_scores)
        return performance_biases, solvers_scores

    @abstractmethod
    def __call__(self, verbose: bool = False) -> GenerationResult:
        """Run the generator and produce a new set of instances.

        This is the main entry point of any concrete generator and must be
        implemented by subclasses. Implementations are expected to drive the
        evolutionary process for ``self._generations`` generations, evaluate
        candidate instances via :meth:`_evaluate_population`, update
        ``self._logbook`` with progress information, and package the final
        result as a :class:`GenerationResult`.

        Args:
            verbose (bool, optional): Whether the generator should print or
                log progress information while running. Defaults to False.

        Returns:
            GenerationResult: The instances produced by the generator,
                together with the solver names, evolutionary history, and
                computed metrics.
        """
        ...
