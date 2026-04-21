#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   map_elites.py
@Time    :   2026/03/25 12:21:11
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Iterable, Optional

import numpy as np

from .._core import (
    Domain,
    Instance,
    P,
    SupportsSolve,
)
from .._core.descriptors import DESCRIPTORS, describe
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import CVTArchive, GridArchive
from ..operators import (
    Mutation,
    batch_uniform_one_mutation,
)
from ..transformers import SupportsTransform
from ._base_generator import BaseGenerator, GenResult


class MapElites(BaseGenerator):
    """Object to generate instances based on MAP-Elites algorithm."""

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int,
        archive: GridArchive | CVTArchive,
        mutation: Mutation,
        repetitions: int,
        describe_by: DESCRIPTORS = "instance",
        transformer: Optional[SupportsTransform] = None,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1_000,
        seed: int = 42,
    ):
        """Creates a MAP-Elites instance generator.
        The generator uses a set of solvers to evaluate the instances and MAP-Elites
        to guide the evolution of the instances.

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Iterable[SupportSolve]): Iterable item of callable objects that can evaluate a instance.
            pop_size (int): Number of instances in the population to evolve. Defaults to 100.
            archive (GridArchive | CVTArchive): Archive to store the instances. It can be a GridArchive or a CVTArchive.
            mutation (Mutation): Mutation operator
            repetitions (int):  Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            describe_by (str): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies.
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            generations (int): Number of generations to perform. Defaults to 1000.
            seed (int, optional): Seed for the RNG protocol. Defaults to 42.

        Raises:
            ValueError: If the archive is not a GridArchive or CVTArchive
        """
        super().__init__(
            domain,
            portfolio,
            pop_size,
            performance_function,
            describe_by,
            generations,
            repetitions,
            seed,
        )
        if not isinstance(archive, (GridArchive, CVTArchive)):
            raise ValueError(
                f"MapElitesGenerator expects an archive of class GridArchive or CVTArchive and got {archive.__class__.__name__}"
            )

        self._archive = archive
        self._mutation = mutation
        self._transformer = transformer

    @property
    def archive(self):
        return self._archive

    def __call__(self, verbose: bool = False) -> GenResult:
        instances = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(instances)
        descriptors, features = describe(
            population=instances,
            key=self._describe_by,
            scores=portfolio_scores,
            domain=self._domain,
            transformer=self._transformer,
        )

        # Here we do not care for p >= 0. We are starting the archive
        # Must be removed later on
        self._archive.extend(instances=instances, descriptors=descriptors)
        self._logbook.update(generation=0, population=instances, feedback=verbose)

        for generation in range(self._generations):
            indices = self._rng.choice(
                list(self._archive.filled_cells), size=self._pop_size
            )
            parents = np.asarray(self._archive[indices], copy=True)
            offspring = batch_uniform_one_mutation(
                parents, self._domain._lbs, ub=self._domain._ubs
            )
            perf_biases, portfolio_scores = self._evaluate_population(offspring)
            descriptors, features = describe(
                population=offspring,
                key=self._describe_by,
                scores=portfolio_scores,
                domain=self._domain,
                transformer=self._transformer,
            )

            offspring_population = [
                Instance(
                    variables=offspring[i],
                    fitness=perf_biases[i],
                    descriptor=descriptors[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                    features=features[i] if features is not None else None,
                )
                for i in range(self._pop_size)
            ]
            self._archive.extend(
                instances=offspring_population, descriptors=descriptors
            )

            # Record the stats and update the performed gens
            self._logbook.update(
                generation=generation + 1, population=self._archive, feedback=verbose
            )

        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        self._archive.purge_unfeasible()
        return GenResult(
            target=self._portfolio[0].__name__,
            instances=self._archive,
            history=self._logbook,
        )
