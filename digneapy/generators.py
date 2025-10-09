#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   generators.py
@Time    :   2023/10/30 14:20:21
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["GenResult", "EAGenerator", "MapElitesGenerator", "DEAGenerator"]

import random
from collections.abc import Iterable
from dataclasses import dataclass
from operator import attrgetter
from typing import Optional, Protocol, Tuple, Sequence
from functools import partial
import numpy as np
import pandas as pd

from ._core import (
    NS,
    dominated_novelty_search,
    RNG,
    Domain,
    Instance,
    P,
    SupportsSolve,
)
from ._core._metrics import Logbook, Statistics
from ._core.descriptors import DESCRIPTORS
from ._core.scores import PerformanceFn, max_gap_target
from .archives import Archive, CVTArchive, GridArchive
from .operators import (
    Crossover,
    Mutation,
    Replacement,
    Selection,
    binary_tournament_selection,
    generational_replacement,
    uniform_crossover,
    uniform_one_mutation,
)
from .transformers import SupportsTransform


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


class Generator(Protocol):
    """Protocol to type check all generators of instances types in digneapy"""

    def __call__(self, *args, **kwargs) -> GenResult: ...


class EAGenerator(Generator, RNG):
    """Object to generate instances based on a Evolutionary Algorithn with set of diverse solutions"""

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        novelty_approach: NS,
        pop_size: int = 100,
        generations: int = 1000,
        solution_set: Optional[Archive] = None,
        descriptor_strategy: str = "features",
        transformer: Optional[SupportsTransform] = None,
        repetitions: int = 1,
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = uniform_crossover,
        mutation: Mutation = uniform_one_mutation,
        selection: Selection = binary_tournament_selection,
        replacement: Replacement = generational_replacement,
        performance_function: PerformanceFn = max_gap_target,
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
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            solution_set (Optional[Archive], optional): Solution set to store the instances. Defaults to None.
            descriptor_strategy (str, optional): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
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

        try:
            self._desc_key = descriptor_strategy
            self._descriptor_strategy = DESCRIPTORS[self._desc_key]
        except KeyError:
            self._desc_key = "instance"
            self._descriptor_strategy = DESCRIPTORS[self._desc_key]
            print(
                f"Descriptor: {descriptor_strategy} not available. Using the full instance as default descriptor"
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
        self.pop_size = pop_size
        self.offspring_size = pop_size
        self.generations = generations
        self.domain = domain
        self.portfolio = tuple(portfolio) if portfolio else ()
        self.population = []
        self.repetitions = repetitions
        self.cxrate = cxrate
        self.mutrate = mutrate

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.replacement = replacement
        self.performance_function = performance_function

        self._logbook = Logbook()
        self.initialize_rng(seed=seed)

    @property
    def log(self) -> Logbook:
        return self._logbook

    def __str__(self):
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"EAGenerator(pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r},{self._novelty_search.__str__()})"

    def __repr__(self) -> str:
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"EAGenerator<pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r},{self._novelty_search.__repr__()}>"

    def __call__(self, verbose: bool = False) -> GenResult:
        if self.domain is None:
            raise ValueError("You must specify a domain to run the generator.")
        if len(self.portfolio) == 0:
            raise ValueError(
                "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
            )
        self.population = self.domain.generate_instances(n=self.pop_size)
        perf_biases, portfolio_scores = self.__evaluate_population(self.population)
        descriptors, features = self.__update_descriptors(
            self.population, portfolio_scores=portfolio_scores
        )
        for pgen in range(self.generations):
            offspring = self.__generate_offspring(self.pop_size)
            perf_biases, portfolio_scores = self.__evaluate_population(offspring)
            descriptors, features = self.__update_descriptors(
                offspring, portfolio_scores=portfolio_scores
            )

            novelty_scores = self._novelty_search(instances_descriptors=descriptors)
            offspring_fitness = self.__compute_fitness(perf_biases, novelty_scores)

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
                # TODO: Here I should only compute the feasible individuals
                novelty_solution_set = self._ns_solution_set(
                    instances_descriptors=descriptors
                )
                self._ns_solution_set.archive.extend(
                    instances=[offspring[i] for i in feasible_indeces],
                    descriptors=descriptors[feasible_indeces],
                    novelty_scores=novelty_solution_set[feasible_indeces],
                )

            # However the whole offspring population is used in the replacement operator
            self.population = self.replacement(self.population, offspring)
            # Record the stats and update the performed gens
            self._logbook.update(
                generation=pgen, population=self.population, feedback=verbose
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
            target=self.portfolio[0].__name__,
            instances=_instances,
            history=self._logbook,
        )

    def __generate_offspring(self, offspring_size: int) -> np.ndarray:
        """Generates a offspring population of size |offspring_size| from the current population

        Args:
            offspring_size (int): offspring size. Defaults to pop_size.

        Returns:
            Sequence[Instance]  Returns a sequence with the instances definitions, the offspring population.
        """
        offspring = [None] * offspring_size  # np.empty(offspring_size, dtype=Instance)
        for i in range(offspring_size):
            p_1 = self.selection(self.population)
            p_2 = self.selection(self.population)
            child = self.__reproduce(p_1, p_2)
            offspring[i] = child

        return np.array(offspring)

    def __update_descriptors(
        self,
        population: np.ndarray,
        portfolio_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Updates the descriptors of the population of instances

        Args:
            population (Sequence[Instance]): Population of instances to update the descriptors.
        """
        descriptors = np.empty(len(population))
        features = None
        if self._desc_key == "features":
            descriptors = self.domain.extract_features(population)
            features = descriptors.copy()
        elif self._desc_key == "performance":
            descriptors = np.mean(portfolio_scores, axis=2)
        else:
            descriptors = self._descriptor_strategy(population)

        if self._transformer is not None:
            # Transform the descriptors if necessary
            descriptors = self._transformer(descriptors)

        return (descriptors, features)

    def __evaluate_population(
        self, population: Sequence[Instance]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the population of instances using the portfolio of solvers.

        Args:
            population (Sequence[Instance]): Sequence of instances to evaluate
        """
        solvers_scores = np.zeros(
            shape=(len(population), len(self.portfolio), self.repetitions)
        )
        problems_to_solve = self.domain.generate_problems_from_instances(population)
        for j, problem in enumerate(problems_to_solve):
            for i, solver in enumerate(self.portfolio):
                # There is no need to change anything in the evaluation code when using Pisinger solvers
                # because the algs. only return one solution per run (len(solutions) == 1)
                # The same happens with the simple KP heuristics. However, when using Pisinger solvers
                # the lower the running time the better they're considered to work an instance
                scores = np.zeros(self.repetitions)
                for rep in range(self.repetitions):
                    scores[rep] = max(
                        solver(problem), key=attrgetter("fitness")
                    ).fitness

                solvers_scores[j, i, :] = scores

        mean_solvers_scores = np.mean(solvers_scores, axis=2)
        performance_biases = self.performance_function(mean_solvers_scores)
        return performance_biases, solvers_scores

    def __compute_fitness(
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
        fitness = (fitness * self.phi) + (novelty_scores * phi_r)
        return fitness

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
            return self.mutation(offspring, self.domain.bounds)
        else:
            return self.mutation(offspring, self.domain.bounds)


class MapElitesGenerator(Generator, RNG):
    """Object to generate instances based on MAP-Elites algorithm."""

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        initial_pop_size: int,
        generations: int,
        archive: GridArchive | CVTArchive,
        mutation: Mutation,
        repetitions: int,
        descriptor: str,
        performance_function: PerformanceFn = max_gap_target,
        seed: int = 42,
    ):
        """Creates a MAP-Elites instance generator.
        The generator uses a set of solvers to evaluate the instances and MAP-Elites
        to guide the evolution of the instances.

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Iterable[SupportSolve]): Iterable item of callable objects that can evaluate a instance.
            initial_pop_size (int): Number of instances in the population to evolve. Defaults to 100.
            generations (int): Number of generations to perform. Defaults to 1000.
            archive (GridArchive | CVTArchive): Archive to store the instances. It can be a GridArchive or a CVTArchive.
            mutation (Mutation): Mutation operator
            repetitions (int):  Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            descriptor (str): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            seed (int, optional): Seed for the RNG protocol. Defaults to 42.

        Raises:
            ValueError: If the archive is not a GridArchive or CVTArchive
        """
        if not isinstance(archive, (GridArchive, CVTArchive)):
            raise ValueError(
                f"MapElitesGenerator expects an archive of class GridArchive or CVTArchive and got {archive.__class__.__name__}"
            )
        self._domain = domain
        self._portfolio = list(portfolio)
        self._init_pop_size = initial_pop_size
        self._generations = generations
        self._archive = archive
        self._mutation = mutation
        self._repetitions = repetitions
        self._performance_fn = performance_function

        if descriptor not in DESCRIPTORS:
            msg = f"descriptor {descriptor} not available in {self.__class__.__name__}.__init__. Set to features by default"
            print(msg)
            descriptor = "features"

        self._descriptor = descriptor
        match descriptor:
            case "features":
                self._descriptor_strategy = self._domain.extract_features
            case _:
                self._descriptor_strategy = DESCRIPTORS[descriptor]

        self._logbook = Logbook()
        self.initialize_rng(seed=seed)

    @property
    def archive(self):
        return self._archive

    @property
    def log(self) -> Logbook:
        return self._logbook

    def __str__(self) -> str:
        port_names = [s.__name__ for s in self._portfolio]
        domain_name = self._domain.name if self._domain is not None else "None"
        return f"MapElites(descriptor={self._descriptor},pop_size={self._init_pop_size},gen={self._generations},domain={domain_name},portfolio={port_names!r})"

    def __repr__(self) -> str:
        port_names = [s.__name__ for s in self._portfolio]
        domain_name = self._domain.name if self._domain is not None else "None"
        return f"MapElites<descriptor={self._descriptor},pop_size={self._init_pop_size},gen={self._generations},domain={domain_name},portfolio={port_names!r}>"

    def _populate_archive(self):
        instances = self._domain.generate_instances(n=self._init_pop_size)
        perf_biases, portfolio_scores = self.__evaluate_population(instances)
        # Here we do not care for p >= 0. We are starting the archive
        # Must be removed later on
        self._archive.extend(instances)
        return instances

    def __evaluate_population(
        self, population: Sequence[Instance]
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

    def __call__(self, verbose: bool = False) -> GenResult:
        instances = self._populate_archive()
        self._logbook.update(generation=0, population=instances, feedback=verbose)
        for generation in range(self._generations):
            parents = [
                p.clone()
                for p in random.choices(self._archive.instances, k=self._init_pop_size)
            ]
            offspring = list(
                map(
                    lambda ind: self._mutation(ind, self._domain.bounds),
                    parents,
                )
            )
            perf_biases, portfolio_scores = self.__evaluate_population(offspring)
            feasible_indices = np.where(perf_biases >= 0)[0]
            feasible_offspring = [
                Instance(
                    variables=offspring[i],
                    fitness=perf_biases[i],
                    descriptor=offspring[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                )
                for i in feasible_indices
            ]
            print(feasible_indices)
            print(feasible_offspring)
            # Only the feasible instances are considered to be included
            # in the archive and the solution set.
            self._archive.extend(feasible_offspring)

            # Record the stats and update the performed gens
            self._logbook.update(
                generation=generation + 1, population=self._archive, feedback=verbose
            )

        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        unfeasible_instances = list(filter(lambda i: i.p < 0, self._archive))
        self._archive.remove(unfeasible_instances)
        return GenResult(
            target=self._portfolio[0].__name__,
            instances=self._archive,
            history=self._logbook,
        )


class DEAGenerator(EAGenerator):
    """
    Object to generate instances based on a Evolutionary Algorithn
    with a Dominated Novelty Search approach
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int = 128,
        offspring_size: int = 128,
        generations: int = 1000,
        k: int = 15,
        descriptor_strategy: str = "features",
        transformer: Optional[SupportsTransform] = None,
        repetitions: int = 1,
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = uniform_crossover,
        mutation: Mutation = uniform_one_mutation,
        selection: Selection = binary_tournament_selection,
        performance_function: PerformanceFn = max_gap_target,
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
            generations=generations,
            descriptor_strategy=descriptor_strategy,
            transformer=transformer,
            repetitions=repetitions,
            cxrate=cxrate,
            mutrate=mutrate,
            crossover=crossover,
            mutation=mutation,
            selection=selection,
            performance_function=performance_function,
        )
        self.offspring_size = offspring_size

    def __str__(self):
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"DEAGenerator(pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r})"

    def __repr__(self) -> str:
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"DEAGenerator<pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r}>"

    def __call__(self, verbose: bool = False) -> GenResult:
        if self.domain is None:
            raise ValueError("You must specify a domain to run the generator.")
        if len(self.portfolio) == 0:
            raise ValueError(
                "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
            )
        self.population = self.domain.generate_instances(self.pop_size)
        perf_biases, portfolio_scores = self.__evaluate_population(self.population)
        descriptors, features = self.__update_descriptors(
            self.population, portfolio_scores=portfolio_scores
        )
        for pgen in range(self.generations):
            offspring = self.__generate_offspring(self.pop_size)
            perf_biases, portfolio_scores = self.__evaluate_population(offspring)
            descriptors, features = self.__update_descriptors(
                offspring, portfolio_scores=portfolio_scores
            )
            combined_population = list(self.population) + list(offspring)
            descriptors, performances, sorted_indexing = dominated_novelty_search(
                combined_population
            )
            # Both population and offspring are used in the replacement
            self.population = [
                Instance(
                    variables=combined_population[i],
                    fitness=perf_biases[i],
                    descriptor=offspring[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                )
                for i in sorted_indexing[: self.pop_size]
            ]
            # Record the stats and update the performed gens
            self._logbook.update(
                generation=pgen, population=self.population, feedback=verbose
            )

        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        return GenResult(
            target=self.portfolio[0].__name__,
            instances=self.population,
            history=self._logbook,
        )
