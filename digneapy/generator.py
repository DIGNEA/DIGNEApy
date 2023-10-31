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

from .novelty_search import NoveltySearch, Archive
from .core import Instance, Domain, Solver
from typing import List
import numpy as np


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
    ):
        super().__init__(t_a, t_ss, k, descriptor)
        self.pop_size = pop_size
        self.max_evaluations = evaluations
        self.domain = domain
        self.portfolio = list(portfolio) if portfolio else []
        self.population = []
        self.repetitions = repetitions

    def __str__(self):
        return f"EIG(pop_size={self.pop_size},evaluations={self.max_evaluations},domain={self.domain.name},portfolio={self.portfolio!r},{super().__str__()})"

    def __repr__(self) -> str:
        return f"EIG<pop_size={self.pop_size},evaluations={self.max_evaluations},domain={self.domain.name},portfolio={self.portfolio!r},{super().__repr__()}>"

    def evaluate_population(self, population: List[Instance]):
        for individual in population:
            p = 0
            avg_p_solver = np.zeros(len(self.portfolio))
            solvers_scores = []
            problem = self.domain.from_instance(individual)
            for i, solver in enumerate(self.portfolio):
                scores = [solver.run(problem)[0] for _ in range(self.repetitions)]
                solvers_scores.append(scores)
                avg_p_solver[i] = np.mean(scores)

            p = avg_p_solver[0] - max(avg_p_solver[1:])
            individual.portfolio_scores = list(solvers_scores)
            individual.p = p

    def run(self):
        self.population = [
            self.domain.generate_instance() for _ in range(self.pop_size)
        ]
        self.evaluate_population(self.population)
        for ind in self.population:
            ind.features = self.domain.extract_features(ind)
            print(f"{ind!r}")
            print(ind)
        s = self.sparseness(self.population)
