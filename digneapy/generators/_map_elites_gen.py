#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _map_elites.py
@Time    :   2024/06/17 10:12:09
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import copy
import random
from collections.abc import Iterable
from operator import attrgetter

import numpy as np
from deap import tools

from digneapy.archives import GridArchive
from digneapy.core import Domain, Instance
from digneapy.core.problem import P
from digneapy.core.solver import SupportsSolve
from digneapy.generators._perf_metrics import PerformanceFn, default_performance_metric
from digneapy.operators import mutation
from digneapy.qd import descriptor_strategies


class MElitGen:
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        initial_pop_size: int,
        generations: int,
        archive: GridArchive,
        mutation: mutation.Mutation,
        repetitions: int,
        descriptor: str,
        performance_function: PerformanceFn = default_performance_metric,
    ):
        self._domain = domain
        self._portfolio = list(portfolio)
        self._init_pop_size = initial_pop_size
        self._generations = generations
        self._archive = archive
        self._mutation = mutation
        self._repetitions = repetitions
        self._performance_fn = performance_function

        if descriptor not in descriptor_strategies:
            msg = f"descriptor {descriptor} not available in {self.__class__.__name__}.__init__. Set to features by default"
            print(msg)
            descriptor = "features"

        self._descriptor = descriptor
        match descriptor:
            case "features":
                self._descriptor_strategy = self._domain.extract_features
            case _:
                self._descriptor_strategy = descriptor_strategies[descriptor]

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

            self._archive.extend(offspring)

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
        return self._archive

    def __call__(self, verbose: bool = False):
        return self._run(verbose)
