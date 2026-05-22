#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   evolutionary.py
@Time    :   2026/03/25 12:20:42
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from digneapy._core.descriptors import DescriptorPipeline

from .._core import (
    Domain,
    Instance,
    Solver,
)
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import Archive
from ..operators import (
    UCX,
    BinarySelection,
    Crossover,
    Generational,
    Mutation,
    Replacement,
    Selection,
    UMut,
)
from ._base_generator import BaseGenerator, GenResult


class Evolutionary(BaseGenerator):
    """Object to generate instances based on a Evolutionary Algorithn with set of diverse solutions"""

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: int,
        archive: Archive,
        solution_set: Optional[Archive] = None,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1000,
        repetitions: int = 1,
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = UCX(),
        mutation: Mutation = UMut(),
        selection: Selection = BinarySelection(),
        replacement: Replacement = Generational(),
        phi: float = 0.85,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search
        The generator uses a set of solvers to evaluate the instances and
        a novelty search algorithm to guide the evolution of the instances.

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Sequence[Solver]): Sequence item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            archive (Archive): Container to store diverse instances.
            solution_set (Optional[Archive], optional): Solution set to store the instances. Defaults to None.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            describe_by (DESCRIPTORS, optional): _Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.DESCRIPTORS. Defaults to "features".
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            crossover (Crossover, optional): Crossover operator. Defaults to uniform_crossover.
            mutation (Mutation, optional): Mutation operator. Defaults to uniform_one_mutation.
            selection (Selection, optional): Selection operator. Defaults to binary_tournament_selection.
            replacement (Replacement, optional): Replacement operator. Defaults to generational_replacement.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            phi (float, optional): Phi balance value for the weighted fitness function. Defaults to 0.85.
            seed (int, optional): Seed for the RNG protocol. Defaults to 42.

        Raises:
            ValueError: Raises error if phi is not a floating point value or it is not in the range [0.0-1.0]
            KeyError: Raises error if the descriptor strategy is not available in the DESCRIPTORS dictionary
        """
        super().__init__(
            domain,
            portfolio,
            pop_size,
            performance_function,
            descriptor_pipe,
            generations=generations,
            repetitions=repetitions,
        )

        if any(param < 0.0 or param > 1.0 for param in (cxrate, mutrate, phi)):
            msg = f"cxrate, mutrate and phi must be a float number in the range [0.0-1.0]. Got: cxrate={cxrate}, mutrate={mutrate}, phi={phi}."
            raise ValueError(msg)

        if not isinstance(archive, Archive):
            raise TypeError("archive must be a subclass of Archive")

        if solution_set is not None and not isinstance(solution_set, Archive):
            raise TypeError("solution_set must be a subclass of Archive")

        self.phi = float(phi)
        self._archive = archive
        self._solution_set = solution_set
        self.offspring_size = self._pop_size
        self.cxrate = float(cxrate)
        self.mutrate = float(mutrate)
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.replacement = replacement
        self.seed = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )

        self._rng = np.random.default_rng(self.seed)

    def __call__(self, verbose: bool = False) -> GenResult:

        self._population = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(self._population)
        descriptors = self._descriptor_pipe(
            population=self._population, scores=portfolio_scores, domain=self._domain
        )
        for pgen in range(self._generations):
            offspring = self.generate(self._pop_size)
            perf_biases, portfolio_scores = self._evaluate_population(offspring)
            descriptors = self._descriptor_pipe(
                population=offspring,
                scores=portfolio_scores,
                domain=self._domain,
            )
            novelty_scores = self._archive(descriptors=descriptors)
            offspring_fitness = self._compute_fitness(perf_biases, novelty_scores)

            # Update to include this
            # 1. Novelty Scores --> novelty_scores
            # 2. Performance bias --> perf_biases
            # 3. Fitness --> oiffspring_fitness
            # 4. Descriptor --> descriptors

            offspring = [
                Instance(
                    variables=offspring[i],
                    fitness=offspring_fitness[i],
                    descriptor=descriptors[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                    s=novelty_scores[i],
                    # Todo: Consider remove features as explicit attribute features=features[i] if features is not None else None,
                )
                for i in range(len(offspring))
            ]
            # Only the feasible instances are considered to be included
            # in the archive and the solution set.
            feasible_indeces = np.where(perf_biases > 0)[0]
            self._archive.extend(
                instances=[offspring[i] for i in feasible_indeces],
                descriptors=descriptors[feasible_indeces],
                novelty_scores=novelty_scores[feasible_indeces],
            )

            if self._solution_set:
                novelty_solution_set = self._solution_set(descriptors=descriptors)
                self._solution_set.extend(
                    instances=[offspring[i] for i in feasible_indeces],
                    descriptors=descriptors[feasible_indeces],
                    novelty_scores=novelty_solution_set[feasible_indeces],
                )

            # However the whole offspring population is used in the replacement operator
            self._population = self.replacement(self._population, offspring)
            # Record the stats and update the performed gens
            self._logbook.update(
                generation=pgen, population=self._population, feedback=verbose
            )

        if verbose:  # pragma: no cover
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        _instances = (
            self._solution_set if self._solution_set is not None else self._archive
        )
        return GenResult(
            target=self._portfolio[0].__name__,
            instances=_instances,
            history=self._logbook,
        )

    def generate(self, pop_size: int) -> np.ndarray:
        """Generates a offspring population of size |offspring_size| from the current population

        Args:
            offspring_size (int): offspring size. Defaults to pop_size.

        Returns:
            Sequence[Instance]  Returns a sequence with the instances definitions, the offspring population.
        """
        offspring = [None] * pop_size  # np.empty(offspring_size, dtype=Instance)
        for i in range(pop_size):
            p_1 = self.selection(self._population)
            p_2 = self.selection(self._population)
            child = self.__reproduce(p_1, p_2)
            offspring[i] = child

        return np.asarray(offspring, copy=True)

    def __reproduce(self, parent_1: Instance, parent_2: Instance) -> Instance:
        """Generates a new offspring instance from two parent instances

        Args:
            parent_1 (Instance): First Parent
            parent_1 (Instance): Second Parent

        Returns:
            Instance: New offspring
        """
        offspring = parent_1.clone()
        if self._rng.random() < self.cxrate:
            offspring = self.crossover(offspring, parent_2)
            return self.mutation(offspring, self._domain.lbs, self._domain.ubs)
        else:
            return self.mutation(offspring, self._domain.lbs, self._domain.ubs)

    def _compute_fitness(
        self, performance_biases: np.ndarray, novelty_scores: np.ndarray
    ) -> np.ndarray:
        """Calculates the fitness of each instance in the population

        Args:
            performance_biases (np.ndarray): Performance biases or scores of each instance
            novelty_scores (np.ndarray): Novelty scores of each instance

        Returns:
            fitness of each instance (np.ndarray)
        """
        phi_r = 1.0 - self.phi
        fitness = np.zeros(len(performance_biases))
        fitness = (performance_biases * self.phi) + (novelty_scores * phi_r)
        return fitness
