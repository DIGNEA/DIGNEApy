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
from dataclasses import dataclass
from operator import attrgetter
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._core import RNG, Domain, Instance, P, SupportsSolve
from .._core._metrics import Logbook, Statistics
from .._core.scores import PerformanceFn, max_gap_target


@dataclass
class GenResult:
    """Class to store the results of the generator
    Attributes:
        target (str): Name of the target solver used to evaluate the instances.
        instances (Sequence[Instance]): List of generated instances.
        history (Logbook): Logbook with the history of the generator.
        metrics (Optional[pd.Series], optional): Metrics of the instances. Defaults to None.
    """

    target: str
    instances: np.ndarray
    history: Logbook
    metrics: Optional[pd.Series] = None

    def __post_init__(self):
        if len(self.instances) != 0:
            self.metrics = Statistics()(self.instances, as_series=True)


class BaseGenerator(ABC, RNG):
    """Abstract base class for all Quality-Diversity generators.

    Handles:
    - Common evaluation pipeline
    - Descriptor computation
    - Result formatting
    - Statistics tracking

    Subclasses implement:
    - generate_population()
    - _generate_offspring()
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        batch_size: int,
        performance_function: PerformanceFn = max_gap_target,
        descriptor: str = "features",
        generations: int = 1_000,
        repetitions: int = 1,
        seed: int = 42,
    ):
        self._domain = domain
        self._portfolio = tuple(portfolio)
        self._batch_size = batch_size
        self._population = []
        self._performance_fn = performance_function
        self._descriptor_key = descriptor
        self._generations = generations
        self._repetitions = repetitions
        self._rng = np.random.default_rng(seed)
        self._logbook = Logbook()
        self.initialize_rng(seed=seed)

    @property
    def log(self) -> Logbook:
        return self._logbook

    def __str__(self):
        port_names = [s.__name__ for s in self._portfolio]
        domain_name = self._domain.name if self._domain is not None else "None"
        return f"Generator(pop_size={self._batch_size},gen={self._generations},domain={domain_name},portfolio={port_names!r})"

    def _evaluate_population(
        self,
        population: Sequence[Instance],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the population of instances using the portfolio of solvers.

        Args:
            population (Sequence[Instance]): Sequence of instances to evaluate
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
    def __call__(self, verbose: bool = False) -> GenResult:
        """Execute the algorithm. Override in subclasses."""
        pass
