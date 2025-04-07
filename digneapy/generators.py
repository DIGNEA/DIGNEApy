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

from collections.abc import Iterable, Sequence
from operator import attrgetter
from typing import Optional, Tuple, Protocol
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd

from ._core import NS, Domain, DominatedNS, Instance, P, SupportsSolve, RNG
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
    instances: Sequence[Instance]
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
            The target solver is the first solver in the portfolio.

        Args: TODO: Update the references
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            evaluations (int, optional): Number of total evaluations to perform. Defaults to 1000.
            archive (Archive): Archive to store the instances to guide the evolution.
            solution_set (Archive): Solution set to store the instances. Defaults to None.
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor_strategy (str, optional): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            dist_metric (str, optional): Defines the distance metric used by NearestNeighbor in the archives. Defaults to Euclidean.
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Iterable[SupportSolve]): Iterable item of callable objects that can evaluate a instance.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            phi (float, optional): Phi balance value for the weighted fitness function. Defaults to 0.85.

        Raises:
            ValueError: Raises error if phi is not a floating point value or it is not in the range [0.0-1.0]
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
        self._ns_solution_set = None  # By default there's not solution set
        if solution_set is not None:
            self._ns_solution_set = NS(archive=solution_set, k=1)

        self._transformer = transformer
        self.pop_size = pop_size
        self.offspring_size = pop_size
        self.generations = generations
        self.domain = domain
        self.portfolio = tuple(portfolio) if portfolio else ()
        self.population: list[Instance] = []
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
        self.population = [
            self.domain.generate_instance() for _ in range(self.pop_size)
        ]
        self.population = self._evaluate_population(self.population)
        self.population = self._update_descriptors(self.population)

        for pgen in range(self.generations):
            offspring: list[Instance] = self._generate_offspring(self.pop_size)
            offspring = self._evaluate_population(offspring)
            offspring = self._update_descriptors(offspring)
            offspring, _ = self._novelty_search(offspring)
            offspring = self._compute_fitness(population=offspring)

            # Only the feasible instances are considered to be included
            # in the archive and the solution set.
            # feasible_off_archive = list(filter(lambda i: i.p >= 0, offspring))
            self._novelty_search.archive.extend(i for i in offspring if i.p >= 0)

            if self._ns_solution_set:
                offspring, _ = self._ns_solution_set(offspring)
                self._ns_solution_set.archive.extend(i for i in offspring if i.p >= 0)

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

    def _generate_offspring(self, offspring_size: int) -> list[Instance]:
        """Generates a offspring population of size |offspring_size| from the current population

        Args:
            offspring_size (int): offspring size. Defaults to pop_size.

        Returns:
            list[Instance]: Returns a list of instances, the offspring population.
        """
        offspring = np.empty(offspring_size, dtype=object)
        for i in range(offspring_size):
            p_1 = self.selection(self.population)
            p_2 = self.selection(self.population)
            off = self._reproduce(p_1, p_2)
            offspring[i] = off
        return offspring

    def _update_descriptors(self, population: list[Instance]):
        """Updates the descriptors of the population of instances

        Args:
            population (list[Instance]): Population of instances to update the descriptors.
        """
        _desc_arr = np.empty(len(population), dtype=object)
        if self._desc_key == "features":
            for i in range(len(population)):
                descriptor = np.array(self.domain.extract_features(population[i]))
                population[i].features = descriptor
                _desc_arr[i] = descriptor
        else:
            _desc_arr = self._descriptor_strategy(population)

        if self._transformer is not None:
            # Transform the descriptors if necessary
            _desc_arr = self._transformer(_desc_arr)
        for i in range(len(population)):
            population[i].descriptor = _desc_arr[i]

        return population

    def _evaluate_population(self, population: Iterable[Instance]):
        """Evaluates the population of instances using the portfolio of solvers.

        Args:
            population (Iterable[Instance]): Population to evaluate
        """
        for individual in population:
            solvers_scores = np.zeros((len(self.portfolio), self.repetitions))
            problem = self.domain.from_instance(individual)

            for i, solver in enumerate(self.portfolio):
                scores = np.zeros(self.repetitions)
                for r in range(self.repetitions):
                    solutions = solver(problem)
                    # There is no need to change anything in the evaluation code when using Pisinger solvers
                    # because the algs. only return one solution per run (len(solutions) == 1)
                    # The same happens with the simple KP heuristics. However, when using Pisinger solvers
                    # the lower the running time the better they're considered to work an instance

                    best_solution = max(solutions, key=attrgetter("fitness"))
                    scores[r] = best_solution.fitness

                solvers_scores[i] = scores

            individual.portfolio_scores = solvers_scores
            avg_p_solver = np.mean(solvers_scores, axis=1)
            individual.p = self.performance_function(avg_p_solver)

        return population

    def _compute_fitness(self, population: Iterable[Instance]):
        """Calculates the fitness of each instance in the population

        Args:
            population (Iterable[Instance], optional): Population of instances to set the fitness. Defaults to None.
        """
        phi_r = 1.0 - self.phi
        for individual in population:
            individual.fitness = individual.p * self.phi + individual.s * phi_r
        return population

    def _reproduce(self, p_1, p_2) -> Instance:
        """Generates a new offspring instance from two parent instances

        Args:
            p_1 (Instance): First Parent
            p_2 (Instance): Second Parent

        Returns:
            Instance: New offspring
        """
        if self._rng.random() < self.cxrate:
            off = self.crossover(p_1, p_2)
            return self.mutation(off, self.domain.bounds)
        else:
            off = p_1.clone()
            return self.mutation(off, self.domain.bounds)


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
        instances = [
            self._domain.generate_instance() for _ in range(self._init_pop_size)
        ]
        instances = self._evaluate_population(instances)
        # Here we do not care for p >= 0. We are starting the archive
        # Must be removed later on
        self._archive.extend(instances)

    def _evaluate_population(self, population: Iterable[Instance]):
        """Evaluates the population of instances using the portfolio of solvers.

        Args:
            population (Iterable[Instance]): Population to evaluate
        """
        for individual in population:
            solvers_scores = np.zeros((len(self._portfolio), self._repetitions))
            problem = self._domain.from_instance(individual)
            for i, solver in enumerate(self._portfolio):
                scores = np.zeros(self._repetitions)
                for r in range(self._repetitions):
                    solutions = solver(problem)
                    best_solution = max(solutions, key=attrgetter("fitness"))
                    scores[r] = best_solution.fitness

                solvers_scores[i] = scores

            individual.portfolio_scores = solvers_scores
            individual.p = self._performance_fn(np.mean(solvers_scores, axis=1))
            individual.fitness = individual.p
            individual.features = self._descriptor_strategy(individual)
            individual.descriptor = individual.features

        return population

    def __call__(self, verbose: bool = False) -> GenResult:
        self._populate_archive()
        self._logbook.update(generation=0, population=self._archive, feedback=verbose)
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

            offspring = self._evaluate_population(offspring)
            # Only the feasible instances are considered to be included
            # in the archive and the solution set.

            self._archive.extend([i for i in offspring if i.p >= 0])

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
            evaluations (int, optional): Number of total evaluations to perform. Defaults to 1000.
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor_strategy (str, optional): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.

        """
        super().__init__(
            domain=domain,
            portfolio=portfolio,
            pop_size=pop_size,
            novelty_approach=DominatedNS(k=k),
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
        self.population = [
            self.domain.generate_instance() for _ in range(self.pop_size)
        ]
        self.population = self._evaluate_population(self.population)
        self.population = self._update_descriptors(self.population)
        for pgen in range(self.generations):
            offspring: list[Instance] = self._generate_offspring(self.offspring_size)
            offspring = self._evaluate_population(offspring)
            offspring = self._update_descriptors(offspring)
            combined_population = list(self.population) + list(offspring)
            combined_population, _ = self._novelty_search(combined_population)
            # Both population and offspring are used in the replacement
            self.population = list(combined_population[: self.pop_size])
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
