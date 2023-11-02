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
from typing import List, Tuple
from operator import attrgetter
import itertools
import numpy as np
import copy


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
        phi: float = 0.85,
        force_bias: bool = True,
    ):
        """_summary_

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
            force_bias (bool, optional): Force that the instances are biased to the performance of the target solver.
            The target solver is the first solver in the portfolio. Defaults to True.

        Raises:
            AttributeError: _description_
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

        if phi < 0.0 or phi > 1.0:
            msg = f"phi must be in range [0.0-1.0]. Got: {phi}."
            raise AttributeError(msg)
        self.phi = phi
        self.force_bias = force_bias

    def __str__(self):
        port_names = [s.__name__ for s in self.portfolio]
        return f"EIG(pop_size={self.pop_size},evaluations={self.max_evaluations},domain={self.domain.name},portfolio={port_names!r},{super().__str__()})"

    def __repr__(self) -> str:
        port_names = [s.__name__ for s in self.portfolio]
        return f"EIG<pop_size={self.pop_size},evaluations={self.max_evaluations},domain={self.domain.name},portfolio={port_names!r},{super().__repr__()}>"

    def __call__(self):
        return self.run()

    def evaluate_population(self, population: List[Instance]):
        for individual in population:
            avg_p_solver = np.zeros(len(self.portfolio))
            solvers_scores = []
            problem = self.domain.from_instance(individual)
            for i, solver in enumerate(self.portfolio):
                scores = [solver(problem)[0] for _ in range(self.repetitions)]
                solvers_scores.append(scores)
                avg_p_solver[i] = np.mean(scores)

            individual.portfolio_scores = list(solvers_scores)
            individual.p = avg_p_solver[0] - max(avg_p_solver[1:])
            print(individual)

    def _compute_fitness(self, population: List[Instance] = None):
        phi_r = 1.0 - self.phi
        for individual in population:
            individual.fitness = individual.p * self.phi + individual.s * phi_r

    def _reproduce(self, p_1: Instance, p_2: Instance) -> Instance:
        off = copy.copy(p_1)
        if np.random.rand() < self.cxrate:
            cross_point = np.random.randint(low=0, high=len(off))
            off[:cross_point] = p_2[:cross_point]
            off[cross_point:] = p_1[cross_point:]
        # Mutation uniform one
        mut_point = np.random.randint(low=0, high=len(off))
        new_value = np.random.uniform(
            low=self.domain.lower_i(mut_point), high=self.domain.upper_i(mut_point)
        )
        off[mut_point] = new_value
        return off

    def __replacement(
        self, current_population: List[Instance], offspring: List[Instance]
    ) -> List[Instance]:
        all_individuals = list(itertools.chain(current_population, offspring))
        best_f = sorted(
            all_individuals,
            key=attrgetter("fitness"),
            reverse=True,
        )
        new_population = [best_f[0]] + offspring[1:]
        return new_population

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

    def run(self):
        self.population = [
            self.domain.generate_instance() for _ in range(self.pop_size)
        ]
        self.evaluate_population(self.population)
        performed_evals = 0
        while performed_evals < self.max_evaluations:
            offspring = []
            for _ in range(self.pop_size):
                p_1 = self.population[np.random.randint(0, self.pop_size)]
                p_2 = self.population[np.random.randint(0, self.pop_size)]
                off = self._reproduce(p_1, p_2)
                off.features = self.domain.extract_features(off)
                offspring.append(off)

            self.evaluate_population(offspring)
            self.sparseness(offspring)
            self._compute_fitness(population=offspring)
            self._update_archive(offspring)
            self._update_solution_set(offspring)

            self.population = copy.copy(offspring)
            performed_evals += self.pop_size

            # if self.force_bias:
            #     biased_offspring = list(filter(lambda x: x.p > 0.0, offspring))
            #     # If we don't have any feasible offspring we can do two things
            #     # In the first generation we just include the best offspring found yet
            #     # In any other generation, we simply avoid this step
            #     if not biased_offspring:
            #         if performed_evals == 0:
            #             best_offs_yet = sorted(
            #                 offspring, key=attrgetter("fitness"), reverse=True
            #             )
            #         # Updates the archive and solution set
            #         self.update_archive([best_offs_yet[0]])
            #     else:
            #         # Updates the archive and solution set
            #         self.update_archive(biased_offspring)
            #         self.update_solution_set(biased_offspring)
            # else:
