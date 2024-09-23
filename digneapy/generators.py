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

__all__ = ["EAGenerator", "MapElitesGenerator"]

import copy
import random
from collections.abc import Iterable
from operator import attrgetter
from typing import Optional

import numpy as np
from deap import tools

from ._core import (
    NS,
    Domain,
    Instance,
    P,
    SupportsSolve,
)
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


class EAGenerator(NS):
    """Object to generate instances based on a Evolutionary Algorithn with a NS archive"""

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int = 100,
        generations: int = 1000,
        archive: Optional[Archive] = None,
        s_set: Optional[Archive] = None,
        k: int = 15,
        descriptor: str = "features",
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
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search

        Args:
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            evaluations (int, optional): Number of total evaluations to perform. Defaults to 1000.
            archive (Archive, optional): Archive to store the instances to guide the evolution. Defaults to Archive(threshold=0.001)..
            s_set (Archive, optional): Solution set to store the instances. Defaults to Archive(threshold=0.001).
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor (str, optional): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Iterable[SupportSolve]): Iterable item of callable objects that can evaluate a instance.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            phi (float, optional): Phi balance value for the weighted fitness function. Defaults to 0.85.
            The target solver is the first solver in the portfolio. Defaults to True.

        Raises:
            ValueError: Raises error if phi is not in the range [0.0-1.0]
        """
        super().__init__(archive, s_set, k, descriptor, transformer)
        self.pop_size = pop_size
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

        self._stats_s = tools.Statistics(key=attrgetter("s"))
        self._stats_p = tools.Statistics(key=attrgetter("p"))
        self._stats = tools.MultiStatistics(s=self._stats_s, p=self._stats_p)
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)

        self._logbook = tools.Logbook()
        self._logbook.header = "gen", "s", "p"
        self._logbook.chapters["s"].header = "min", "avg", "std", "max"
        self._logbook.chapters["p"].header = "min", "avg", "std", "max"

        try:
            phi = float(phi)
        except ValueError:
            raise ValueError("Phi must be a float number in the range [0.0-1.0].")

        if phi < 0.0 or phi > 1.0:
            msg = f"Phi must be a float number in the range [0.0-1.0]. Got: {phi}."
            raise ValueError(msg)
        self.phi = phi

    @property
    def log(self) -> tools.Logbook:
        return self._logbook

    def __str__(self):
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"EAGenerator(pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r},{super().__str__()})"

    def __repr__(self) -> str:
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"EAGenerator<pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r},{super().__repr__()}>"

    def __call__(self, verbose: bool = False):
        return self._run(verbose)

    def _evaluate_population(self, population: Iterable[Instance]):
        """Evaluates the population of instances using the portfolio of solvers.

        Args:
            population (Iterable[Instance]): Population to evaluate
        """
        for individual in population:
            avg_p_solver = np.zeros(len(self.portfolio))
            solvers_scores = []
            problem = self.domain.from_instance(individual)
            for i, solver in enumerate(self.portfolio):
                scores = []
                for _ in range(self.repetitions):
                    solutions = solver(problem)
                    # There is no need to change anything in the evaluation code when using Pisinger solvers
                    # because the algs. only return one solution per run (len(solutions) == 1)
                    # The same happens with the simple KP heuristics. However, when using Pisinger solvers
                    # the lower the running time the better they're considered to work an instance
                    solutions = sorted(
                        solutions, key=attrgetter("fitness"), reverse=True
                    )
                    scores.append(solutions[0].fitness)
                solvers_scores.append(scores)
                avg_p_solver[i] = np.mean(scores)

            individual.portfolio_scores = tuple(solvers_scores)
            individual.p = self.performance_function(avg_p_solver)

    def _compute_fitness(self, population: Iterable[Instance]):
        """Calculates the fitness of each instance in the population

        Args:
            population (Iterable[Instance], optional): Population of instances to set the fitness. Defaults to None.
        """
        phi_r = 1.0 - self.phi
        for individual in population:
            individual.fitness = individual.p * self.phi + individual.s * phi_r

    def _reproduce(self, p_1, p_2) -> Instance:
        """Generates a new offspring instance from two parent instances

        Args:
            p_1 (Instance): First Parent
            p_2 (Instance): Second Parent

        Returns:
            Instance: New offspring
        """
        off = copy.deepcopy(p_1)
        if np.random.rand() < self.cxrate:
            off = self.crossover(p_1, p_2)
        off = self.mutation(p_1, self.domain.bounds)
        return off

    def _run(self, verbose: bool = False):
        if self.domain is None:
            raise ValueError("You must specify a domain to run the generator.")
        if len(self.portfolio) == 0:
            raise ValueError(
                "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
            )
        self.population = [
            self.domain.generate_instance() for _ in range(self.pop_size)
        ]
        self._evaluate_population(self.population)
        performed_gens = 0
        while performed_gens < self.generations:
            offspring: list[Instance] = []
            for _ in range(self.pop_size):
                p_1 = self.selection(self.population)
                p_2 = self.selection(self.population)
                off = self._reproduce(p_1, p_2)
                if self._describe_by == "features":
                    off.features = self.domain.extract_features(off)
                offspring.append(off)

            self._evaluate_population(offspring)
            self.sparseness(offspring)
            self._compute_fitness(population=offspring)
            # Only the feasible instances are considered to be included
            # in the archive and the solution set.
            feasible_offspring = list(filter(lambda i: i.p >= 0, offspring))
            self.archive.extend(feasible_offspring)
            self.sparseness_solution_set(offspring)
            self.solution_set.extend(feasible_offspring)
            # However the whole offspring population is used in the replacement operator
            self.population = self.replacement(self.population, offspring)

            # Record the stats and update the performed gens
            record = self._stats.compile(self.population)
            self._logbook.record(gen=performed_gens, **record)
            performed_gens += 1

            if verbose:
                status = f"\rGeneration {performed_gens}/{self.generations} completed"
                print(status, flush=True, end="")
        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        return (self.archive, self.solution_set)


class MapElitesGenerator:
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

        self._stats_fitness = tools.Statistics(key=attrgetter("fitness"))
        self._stats_fitness.register("avg", np.mean)
        self._stats_fitness.register("std", np.std)
        self._stats_fitness.register("min", np.min)
        self._stats_fitness.register("max", np.max)
        self._logbook = tools.Logbook()
        self._logbook.header = ("gen", "min", "avg", "std", "max")

    @property
    def archive(self):
        return self._archive

    @property
    def log(self) -> tools.Logbook:
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
            avg_p_solver = np.zeros(len(self._portfolio))
            solvers_scores = []
            problem = self._domain.from_instance(individual)
            for i, solver in enumerate(self._portfolio):
                scores = []
                for _ in range(self._repetitions):
                    solutions = solver(problem)
                    # There is no need to change anything in the evaluation code when using Pisinger solvers
                    # because the algs. only return one solution per run (len(solutions) == 1)
                    # The same happens with the simple KP heuristics. However, when using Pisinger solvers
                    # the lower the running time the better they're considered to work an instance
                    solutions = sorted(
                        solutions, key=attrgetter("fitness"), reverse=True
                    )
                    scores.append(solutions[0].fitness)
                solvers_scores.append(scores)
                avg_p_solver[i] = np.mean(scores)

            individual.portfolio_scores = tuple(solvers_scores)
            individual.p = self._performance_fn(avg_p_solver)
            individual.fitness = individual.p
            ind_features = tuple(self._descriptor_strategy(individual))
            individual.features = ind_features
            individual.descriptor = ind_features
        return population

    def _run(self, verbose: bool = False):
        self._populate_archive()

        record = self._stats_fitness.compile(self._archive)
        self._logbook.record(gen=0, **record)

        for generation in range(self._generations):
            parents = [
                copy.deepcopy(p)
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
            feasible_offspring = list(filter(lambda i: i.p >= 0, offspring))

            self._archive.extend(feasible_offspring)

            # Record the stats and update the performed gens
            record = self._stats_fitness.compile(self._archive)
            self._logbook.record(gen=generation + 1, **record)

            if verbose:
                status = f"\rGeneration {generation}/{self._generations} completed"
                print(status, flush=True, end="")

        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        unfeasible_instances = list(filter(lambda i: i.p < 0, self._archive))

        self._archive.remove(unfeasible_instances)
        return self._archive

    def __call__(self, verbose: bool = False):
        return self._run(verbose)
