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

from typing import Iterable, Optional

import numpy as np

from .._core import (
    NS,
    Domain,
    Instance,
    P,
    SupportsSolve,
    dominated_novelty_search,
)
from .._core.descriptors import DESCRIPTORS, describe
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import Archive
from ..operators import (
    Crossover,
    Mutation,
    Replacement,
    Selection,
    binary_tournament_selection,
    generational_replacement,
    uniform_crossover,
    uniform_one_mutation,
)
from ..transformers import SupportsTransform
from ._base_generator import BaseGenerator, GenResult


class Evolutionary(BaseGenerator):
    """Object to generate instances based on a Evolutionary Algorithn with set of diverse solutions"""

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int,
        novelty_approach: NS,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1000,
        repetitions: int = 1,
        solution_set: Optional[Archive] = None,
        describe_by: DESCRIPTORS = "features",
        transformer: Optional[SupportsTransform] = None,
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = uniform_crossover,
        mutation: Mutation = uniform_one_mutation,
        selection: Selection = binary_tournament_selection,
        replacement: Replacement = generational_replacement,
        phi: float = 0.85,
        seed: int = 42,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search
        The generator uses a set of solvers to evaluate the instances and
        a novelty search algorithm to guide the evolution of the instances.

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Iterable[SupportSolve]): Iterable item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            novelty_approach (NS): Novelty Search strategy to produce diverse instances.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            solution_set (Optional[Archive], optional): Solution set to store the instances. Defaults to None.
            describe_by (DESCRIPTORS, optional): _Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.DESCRIPTORS. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
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
            describe_by,
            generations,
            repetitions,
            seed,
        )

        try:
            phi = float(phi)
        except ValueError:
            raise ValueError("Phi must be a float number in the range [0.0-1.0].")

        if phi < 0.0 or phi > 1.0:
            msg = f"Phi must be a float number in the range [0.0-1.0]. Got: {phi}."
            raise ValueError(msg)

        self.phi = phi
        self._novelty_search = novelty_approach
        self._solution_set = None  # By default there's not solution set
        if solution_set is not None:
            self._ns_solution_set = NS(archive=solution_set, k=1)

        self._transformer = transformer
        self.offspring_size = self._pop_size
        self.cxrate = cxrate
        self.mutrate = mutrate
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.replacement = replacement

    def __call__(self, verbose: bool = False) -> GenResult:
        if self._domain is None:
            raise ValueError("You must specify a domain to run the generator.")
        if len(self._portfolio) == 0:
            raise ValueError(
                "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
            )
        if self._novelty_search is None:
            raise ValueError("Novelty Search cannot be None in Evolutionary Generator")

        self._population = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(self._population)
        descriptors, features = describe(
            population=self._population,
            key=self._describe_by,
            scores=portfolio_scores,
            domain=self._domain,
            transformer=self._transformer,
        )
        for pgen in range(self._generations):
            offspring = self.generate(self._pop_size)
            perf_biases, portfolio_scores = self._evaluate_population(offspring)
            descriptors, features = describe(
                population=offspring,
                key=self._describe_by,
                scores=portfolio_scores,
                domain=self._domain,
                transformer=self._transformer,
            )

            novelty_scores = self._novelty_search(instances_descriptors=descriptors)
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
                    features=features[i] if features is not None else None,
                )
                for i in range(len(offspring))
            ]
            # Only the feasible instances are considered to be included
            # in the archive and the solution set.
            feasible_indeces = np.where(perf_biases > 0)[0]
            self._novelty_search.archive.extend(
                instances=[offspring[i] for i in feasible_indeces],
                descriptors=descriptors[feasible_indeces],
                novelty_scores=novelty_scores[feasible_indeces],
            )

            if self._ns_solution_set:
                novelty_solution_set = self._ns_solution_set(
                    instances_descriptors=descriptors
                )
                self._ns_solution_set.archive.extend(
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

        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        _instances = (
            self._ns_solution_set.archive
            if self._ns_solution_set is not None
            else self._novelty_search.archive
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
            return self.mutation(offspring, self._domain.bounds)
        else:
            return self.mutation(offspring, self._domain.bounds)

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


class Dominated(Evolutionary):
    """
    Object to generate instances based on a Evolutionary Algorithn
    with a Dominated Novelty Search approach
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int = 128,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1000,
        repetitions: int = 1,
        k: int = 15,
        describe_by: DESCRIPTORS = "features",
        transformer: Optional[SupportsTransform] = None,
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = uniform_crossover,
        mutation: Mutation = uniform_one_mutation,
        selection: Selection = binary_tournament_selection,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Iterable[SupportSolve]): Iterable item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 128.
            offspring_size (int, optional): Number of instances in the offspring population. Defaults to 128.
            generations (int, optional): Number of total generations to perform. Defaults to 1000.
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor_strategy (str, optional): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            crossover (Crossover, optional): Crossover operator. Defaults to uniform_crossover.
            mutation (Mutation, optional): Mutation operator. Defaults to uniform_one_mutation.
            selection (Selection, optional): Selection operator. Defaults to binary_tournament_selection.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
        """
        super().__init__(
            domain=domain,
            portfolio=portfolio,
            pop_size=pop_size,
            novelty_approach=None,
            performance_function=performance_function,
            generations=generations,
            repetitions=repetitions,
            transformer=transformer,
            cxrate=cxrate,
            mutrate=mutrate,
            crossover=crossover,
            mutation=mutation,
            selection=selection,
            describe_by=describe_by,
        )
        self.k = k
        self.offspring_size = pop_size

    def __call__(self, verbose: bool = False) -> GenResult:
        if self._domain is None:
            raise ValueError("You must specify a domain to run the generator.")
        if len(self._portfolio) == 0:
            raise ValueError(
                "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
            )
        self._population = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(self._population)
        descriptors, features = describe(
            population=self._population,
            key=self._describe_by,
            scores=portfolio_scores,
            domain=self._domain,
            transformer=self._transformer,
        )

        if features is not None:
            combined_features = np.empty(
                shape=(self._pop_size * 2, features.shape[1]), dtype=np.float32
            )
        for pgen in range(self._generations):
            offspring = self.generate(self._pop_size)
            off_perf_biases, off_portfolio_scores = self._evaluate_population(offspring)
            off_descriptors, off_features = describe(
                population=offspring,
                key=self._describe_by,
                scores=off_portfolio_scores,
                domain=self._domain,
                transformer=self._transformer,
            )
            combined_descriptors = np.concatenate(
                (descriptors, off_descriptors), axis=0
            )
            combined_performances = np.concatenate(
                (perf_biases, off_perf_biases), axis=0
            )
            combined_port_scores = np.concatenate(
                (portfolio_scores, portfolio_scores), axis=0
            )
            genotypes = np.concatenate(
                (np.asarray(self._population, copy=True), offspring), axis=0
            )
            if features is not None:
                combined_features = np.concatenate((features, off_features), axis=0)

            (
                sorted_descriptors,
                sorted_performances,
                sorted_competition_fitness,
                sorted_indexing,
            ) = dominated_novelty_search(
                descriptors=combined_descriptors,
                performances=combined_performances,
                k=self.k,
                force_feasible_only=True,
            )

            # Keep the top N for the next generation
            sorted_indexing = sorted_indexing[: self._pop_size]
            fitness = sorted_competition_fitness[: self._pop_size]
            descriptors = sorted_descriptors[: self._pop_size]
            perf_biases = sorted_performances[: self._pop_size]
            # Track from the combined arrays based on the indexing
            portfolio_scores = combined_port_scores[sorted_indexing]
            genotypes = genotypes[sorted_indexing]
            if features is not None:
                features = combined_features[sorted_indexing]
            # Both population and offspring are used in the replacement
            # Record the stats and update the performed gens
            self._population = [
                Instance(
                    variables=genotypes[i],
                    fitness=fitness[i],
                    descriptor=descriptors[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                    features=features[i] if features is not None else (),
                )
                for i in range(self._pop_size)
            ]

            self._logbook.update(
                generation=pgen, population=self._population, feedback=verbose
            )

        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        return GenResult(
            target=self._portfolio[0].__name__,
            instances=self._population,
            history=self._logbook,
        )
