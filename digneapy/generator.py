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

from .novelty_search import NoveltySearch
from .core import Instance, Domain, Solver
from typing import List
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
        portfolio: List[Solver] = None,
        repetitions: int = 1,
        cxrate: float = 0.5,
        mutrate: float = 0.8,
    ):
        super().__init__(t_a, t_ss, k, descriptor)
        self.pop_size = pop_size
        self.max_evaluations = evaluations
        self.domain = domain
        self.portfolio = list(portfolio) if portfolio else []
        self.population = []
        self.repetitions = repetitions
        self.cxrate = cxrate
        self.mutrate = mutrate

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
            p = 0
            avg_p_solver = np.zeros(len(self.portfolio))
            solvers_scores = []
            problem = self.domain.from_instance(individual)
            for i, solver in enumerate(self.portfolio):
                scores = [solver(problem)[0] for _ in range(self.repetitions)]
                solvers_scores.append(scores)
                avg_p_solver[i] = np.mean(scores)

            p = avg_p_solver[0] - max(avg_p_solver[1:])
            individual.portfolio_scores = list(solvers_scores)
            individual.p = p

    def _compute_fitness(self, population: List[Instance]):
        for individual in population:
            individual.fitness = individual.p * 0.85 + individual.s * 0.15

    def _reproduce(self, p_1: Instance, p_2: Instance) -> Instance:
        off = copy.copy(p_1)
        if np.random.rand() < self.cxrate:
            cross_point = np.random.randint(low=0, high=len(off))
            off[:cross_point] = p_2[:cross_point]
            off[cross_point:] = p_1[cross_point:]
        # Mutation uniform one
        mut_point = np.random.randint(low=0, high=len(off))
        new_value = np.random.rand(
            self.domain.lower_i(mut_point), self.domain.upper_i(mut_point)
        )
        off[mut_point] = new_value
        return off

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
            self._compute_fitness(offspring)
            # Updates the archive and solution set
            self.update_archive(offspring)
            self.update_solution_set(offspring)
            self.population = copy.copy(offspring)

            performed_evals += self.pop_size
