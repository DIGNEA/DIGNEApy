#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   generator.py
@Time    :   2023/10/30 14:20:21
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""
from collections.abc import Sequence, Iterable
from digneapy.generators.perf_metrics import default_performance_metric
from digneapy.qd import NS
from digneapy.archives import Archive
from digneapy.core import Instance, Domain, Solver
from digneapy.operators import crossover, mutation, selection, replacement
from typing import List, Callable, Optional
from operator import attrgetter
import numpy as np
import copy
from deap import tools


PerformanceFn = Callable[[Sequence[float]], float]


class EIG(NS):
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[Solver],
        pop_size: int = 100,
        generations: int = 1000,
        archive: Optional[Archive] = None,
        s_set: Optional[Archive] = None,
        k: int = 15,
        descriptor: str = "features",
        transformer: Optional[Callable[[Sequence | Iterable], np.ndarray]] = None,
        repetitions: int = 1,
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: crossover.Crossover = crossover.uniform_crossover,
        mutation: mutation.Mutation = mutation.uniform_one_mutation,
        selection: selection.Selection = selection.binary_tournament_selection,
        replacement: replacement.Replacement = replacement.generational,
        performance_function: PerformanceFn = default_performance_metric,
        phi: float = 0.85,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search

        Args:
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            evaluations (int, optional): Number of total evaluations to perform. Defaults to 1000.
            archive (Archive, optional): Archive to store the instances to guide the evolution. Defaults to Archive(threshold=0.001)..
            s_set (Archive, optional): Solution set to store the instances. Defaults to Archive(threshold=0.001).
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor (str, optional): Descriptor used to calculate the diversity. The options are features, performance or instance. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Tuple[Solver]): Tuple of callable objects that can evaluate a instance.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            phi (float, optional): Phi balance value for the weighted fitness function. Defaults to 0.85.
            The target solver is the first solver in the portfolio. Defaults to True.

        Raises:
            AttributeError: Raises error if phi is not in the range [0.0-1.0]
        """
        super().__init__(archive, s_set, k, descriptor, transformer)
        self.pop_size = pop_size
        self.generations = generations
        self.domain = domain
        self.portfolio = tuple(portfolio) if portfolio else ()
        self.population: List[Instance] = []
        self.repetitions = repetitions
        self.cxrate = cxrate
        self.mutrate = mutrate

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.replacement = replacement
        self.performance_function = performance_function

        self._stats_s = tools.Statistics(key=lambda ind: ind.s)
        self._stats_p = tools.Statistics(key=lambda ind: ind.p)
        self._stats = tools.MultiStatistics(s=self._stats_s, p=self._stats_p)
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)

        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "s", "p"
        self.logbook.chapters["s"].header = "min", "avg", "std", "max"
        self.logbook.chapters["p"].header = "min", "avg", "std", "max"

        try:
            phi = float(phi)
        except ValueError:
            raise AttributeError(f"Phi must be a float number in the range [0.0-1.0].")

        if phi < 0.0 or phi > 1.0:
            msg = f"Phi must be a float number in the range [0.0-1.0]. Got: {phi}."
            raise AttributeError(msg)
        self.phi = phi

    def __str__(self):
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"EIG(pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r},{super().__str__()})"

    def __repr__(self) -> str:
        port_names = [s.__name__ for s in self.portfolio]
        domain_name = self.domain.name if self.domain is not None else "None"
        return f"EIG<pop_size={self.pop_size},gen={self.generations},domain={domain_name},portfolio={port_names!r},{super().__repr__()}>"

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

            individual.portfolio_scores = list(solvers_scores)
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

    def _update_archive(self, instances: Iterable[Instance]):
        """Updates the Novelty Search Archive with all the instances that has a 's' greater than t_a and 'p' > 0"""
        if not instances:
            return
        filter_fn = lambda x: x.s >= self.archive.threshold and x.p >= 0.0
        self.archive.extend(instances, filter_fn=filter_fn)

    def _update_solution_set(self, instances: Iterable[Instance]):
        """Updates the Novelty Search Solution set with all the instances that has a 's' greater than t_ss and 'p' > 0"""
        if not instances:
            return
        filter_fn = lambda x: x.s >= self.archive.threshold and x.p >= 0.0
        self.solution_set.extend(instances, filter_fn)

    def _run(self, verbose: bool = False):
        if self.domain is None:
            raise AttributeError("You must specify a domain to run the generator.")
        if len(self.portfolio) == 0:
            raise AttributeError(
                "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
            )
        self.population = [
            self.domain.generate_instance() for _ in range(self.pop_size)
        ]
        # Filter function to update the archives
        # It must consider the performance score
        filter_fn = lambda x: x.s >= self.archive.threshold and x.p >= 0.0

        self._evaluate_population(self.population)
        performed_gens = 0
        while performed_gens < self.generations:
            offspring = []
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

            self.archive.extend(offspring, filter_fn)

            self.sparseness_solution_set(offspring)
            self.solution_set.extend(offspring, filter_fn)

            self.population = self.replacement(self.population, offspring)

            # Record the stats and update the performed gens
            record = self._stats.compile(self.population)
            self.logbook.record(gen=performed_gens, **record)
            performed_gens += 1

            if verbose:
                status = f"\rGeneration {performed_gens}/{self.generations} completed"
                print(status, flush=True, end="")
        if verbose:
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        return (self.archive, self.solution_set)
