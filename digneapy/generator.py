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
from sklearn.neighbors import NearestNeighbors
from .novelty_search import NoveltySearch
from .core import Instance, Domain, Solver
from .operators import crossover, mutation, selection, replacement
from typing import List, Tuple, Iterable, Callable
from operator import attrgetter
import functools
import numpy as np
import copy


PerformanceFn = Callable[[Iterable[float]], float]


def _default_performance_metric(scores: Iterable[float]) -> float:
    """Default performace metric for the instances.
    It tries to maximise the gap between the target solver
    and the other solvers in the portfolio.

    Args:
        scores (Iterable[float]): Scores of each solver over an instance. It is expected
        that the first value is the score of the target.

    Returns:
        float: Performance value for an instance. Instance.p attribute.
    """
    return scores[0] - max(scores[1:])


class EIG(NoveltySearch):
    def __init__(
        self,
        pop_size: int = 100,
        evaluations: int = 10000,
        t_a: float = 0.001,
        t_ss: float = 0.001,
        k: int = 15,
        descriptor="features",
        domain: Domain = None,
        portfolio: Tuple[Solver] = None,
        repetitions: int = 1,
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: crossover.Crossover = crossover.uniform_crossover,
        mutation: mutation.Mutation = mutation.uniform_one_mutation,
        selection: selection.Selection = selection.binary_tournament_selection,
        performance_function: PerformanceFn = _default_performance_metric,
        phi: float = 0.85,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search

        Args:
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            evaluations (int, optional): Number of total evaluations to perform. Defaults to 10000.
            t_a (float, optional): Archive threshold. Defaults to 0.001.
            t_ss (float, optional): Solution set threshold. Defaults to 0.001.
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor (str, optional): Descriptor used to calculate the diversity. The options are features or performance. Defaults to "features".
            domain (Domain, optional): Domain for which the instances are generated for. Defaults to None.
            portfolio (Tuple[Solver], optional): Tuple of callable objects that can evaluate a instance. Defaults to None.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            phi (float, optional): Phi balance value for the weighted fitness function. Defaults to 0.85.
            The target solver is the first solver in the portfolio. Defaults to True.

        Raises:
            AttributeError: Raises error if phi is not in the range [0.0-1.0]
        """
        super().__init__(t_a, t_ss, k, descriptor)
        self.pop_size = pop_size
        self.max_evaluations = evaluations
        self.domain = domain
        self.portfolio = tuple(portfolio) if portfolio else ()
        self.population = []
        self.repetitions = repetitions
        self.cxrate = cxrate
        self.mutrate = mutrate

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.performance_function = performance_function

        if phi < 0.0 or phi > 1.0:
            msg = f"phi must be in range [0.0-1.0]. Got: {phi}."
            raise AttributeError(msg)
        self.phi = phi

    def __str__(self):
        port_names = [s.__name__ for s in self.portfolio]
        return f"EIG(pop_size={self.pop_size},evaluations={self.max_evaluations},domain={self.domain.name},portfolio={port_names!r},{super().__str__()})"

    def __repr__(self) -> str:
        port_names = [s.__name__ for s in self.portfolio]
        return f"EIG<pop_size={self.pop_size},evaluations={self.max_evaluations},domain={self.domain.name},portfolio={port_names!r},{super().__repr__()}>"

    def __call__(self):
        return self._run()

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
                    solutions = sorted(
                        solutions, key=attrgetter("fitness"), reverse=True
                    )
                    scores.append(solutions[0].fitness)
                solvers_scores.append(scores)
                avg_p_solver[i] = np.mean(scores)

            individual.portfolio_scores = list(solvers_scores)
            # avg_p_solver[0] - max(avg_p_solver[1:])
            individual.p = self.performance_function(avg_p_solver)

    def _compute_fitness(self, population: Iterable[Instance] = None):
        """Calculates the fitness of each instance in the population

        Args:
            population (Iterable[Instance], optional): Population of instances to set the fitness. Defaults to None.
        """
        phi_r = 1.0 - self.phi
        for individual in population:
            individual.fitness = individual.p * self.phi + individual.s * phi_r

    def _reproduce(self, p_1: Instance, p_2: Instance) -> Instance:
        """Generates a new offspring instance from two parent instances

        Args:
            p_1 (Instance): First Parent
            p_2 (Instance): Second Parent

        Returns:
            Instance: New offspring
        """
        off = copy.copy(p_1)
        if np.random.rand() < self.cxrate:
            off = self.crossover(p_1, p_2)
        off = self.mutation(p_1, self.domain.bounds)
        return off

    def _update_archive(self, instances: List[Instance]):
        """Updates the Novelty Search Archive with all the instances that has a 's' greater than t_a"""
        if not instances:
            return
        self._archive.extend(filter(lambda x: x.s >= self.t_a and x.p > 0.0, instances))

    def _update_solution_set(self, instances: List[Instance], verbose: bool = False):
        """Updates the Novelty Search Archive with all the instances that has a 's' greater than t_ss when K is set to 1"""

        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{self.__class__.__name__} trying to update the solution set with an empty instance list"
            raise AttributeError(msg)

        if self._k >= len(instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness_solution_set with k({self._k}) > len(instances)({len(instances)})"
            raise AttributeError(msg)

        _descriptors_arr = super()._combined_archive_and_population(
            self.solution_set, instances
        )

        neighbourhood = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
        neighbourhood.fit(_descriptors_arr)
        for instance, descriptor in zip(
            instances, _descriptors_arr[0 : len(instances)]
        ):
            dist, ind = neighbourhood.kneighbors([descriptor])
            dist, ind = dist[0][1:], ind[0][1:]
            s = (1.0 / self._k) * sum(dist)
            if s >= self._t_ss and instance.p > 0.0:
                self._solution_set.append(instance)

    def _run(self):
        self.population = [
            self.domain.generate_instance() for _ in range(self.pop_size)
        ]
        self._evaluate_population(self.population)
        performed_evals = 0
        while performed_evals < self.max_evaluations:
            offspring = []
            for _ in range(self.pop_size):
                p_1 = self.selection(self.population)
                p_2 = self.selection(self.population)
                off = self._reproduce(p_1, p_2)
                off.features = self.domain.extract_features(off)
                offspring.append(off)

            self._evaluate_population(offspring)
            self.sparseness(offspring)
            self._compute_fitness(population=offspring)
            self._update_archive(offspring)
            self._update_solution_set(offspring)
            self.population = replacement.generational(self.population, offspring)

            performed_evals += self.pop_size

        return (self.archive, self.solution_set)
