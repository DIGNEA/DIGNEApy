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
from typing import List, Mapping, Optional, Tuple

import numpy as np
import polars as pl

from digneapy._core.descriptors import DescriptorPipeline
from digneapy.generators._utils import extract_solvers_name

from .._core import (
    Domain,
    Instance,
    Logbook,
    Solver,
    Statistics,
)
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import Archive


@dataclass
class GenResult:
    """Class to store the results of the generator
    Attributes:
        solvers (Sequence[str]): Name of the solvers used to evaluate the instances.
        instances (Sequence[Instance]): List of generated instances.
        history (Logbook): Logbook with the history of the generator.
        metrics (Optional[pd.Series], optional): Metrics of the instances. Defaults to None.
    """

    solvers: Sequence[str]
    instances: Archive | np.ndarray | Sequence[Instance]
    history: Logbook
    metrics: pl.DataFrame | Mapping = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        if len(self.instances) != 0:
            self.metrics = Statistics()(self.instances, as_dataframe=True)
        else:
            self._metrics = pl.Series()


class BaseGenerator(ABC):
    """Abstract base class for all Quality-Diversity generators.

    Handles:
    - Common evaluation pipeline
    - Descriptor computation
    - Result formatting
    - Statistics tracking
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: np.uint32,
        performance_function: PerformanceFn = max_gap_target,
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        generations: np.uint32 = np.uint32(1_000),
        repetitions: np.uint16 = np.uint16(1),
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates an instance of BaseGenerator with common attributes for generators

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Sequence[SupportSolve]): Sequence item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            describe_by (DESCRIPTORS, optional): _Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.DESCRIPTORS. Defaults to "features".
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
        """

        if any(
            type(param) not in (int,) or param < 0
            for param in (pop_size, repetitions, generations)
        ):
            raise ValueError(
                f"BaseGenerator: pop_size, repetitions and/or generations are negative. pop_size: {pop_size}, repetitions: {repetitions}, generations: {generations}"
            )
        if not isinstance(domain, Domain):
            raise ValueError(f"BaseGenerator: Invalid domain. Got {domain}.")
        if len(portfolio) == 0 or any(
            not isinstance(solver, Solver) for solver in portfolio
        ):
            raise ValueError(
                f"BaseGenerator: the portfolio is empty or contains invalid solvers. {portfolio}"
            )
        self._domain = domain
        self._portfolio = tuple(portfolio)
        self._pop_size = pop_size
        self._population = []
        self._performance_fn = performance_function
        self._descriptor_pipe = descriptor_pipe
        self._generations = generations
        self._repetitions = repetitions
        self._logbook = Logbook()

    @property
    def log(self) -> Logbook:
        return self._logbook

    @property
    def descriptor_pipeline(self):
        return self._descriptor_pipe

    def __str__(self):
        port_names = tuple(extract_solvers_name(self._portfolio))
        domain_name = self._domain.__name__ if self._domain is not None else "None"
        return f"{self.__class__.__name__}(pop_size={self._pop_size},gen={self._generations},domain={domain_name},portfolio={port_names!r})"

    def __repr__(self) -> str:
        return self.__str__().replace("(", "<").replace(")", ">")

    def _evaluate_population(
        self,
        population: np.ndarray | List[Instance],
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
    def __call__(self, verbose: bool = False) -> GenResult: ...
