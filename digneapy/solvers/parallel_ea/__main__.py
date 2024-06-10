#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __main__.py
@Time    :   2024/06/10 12:19:27
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from digneapy.domains.knapsack import Knapsack
from digneapy.core import Solution, Solver
import numpy as np
from parallel_ea import _ParEACpp


class ParEAKP(_ParEACpp):
    """Parallel Evolutionary Algorithm for Knapsack Problems
    It uses Uniform One Mutation and Uniform Mutation as mating operators.
    The replacement is based on a Greedy strategy. The parent and offspring
    populations are evaluated pairwise and at each position i, the best individual
    between parent_i and offspring_i survives for the next_population_i.
    """

    def __init__(
        self,
        pop_size: int = 32,
        generations: int = 1000,
        mutpb: float = 0.2,
        cxpb: float = 0.7,
        cores: int = 1,
    ):
        """Creates an instance of the ParEAKP solver

        Args:
            pop_size (int, optional): Population size. Defaults to 32.
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            mutpb (float, optional): Probability of mutation. Defaults to 0.2.
            cxpb (float, optional): Probability of crossover between two individuals. Defaults to 0.7.
            cores (int, optional): Number of cores to use. Defaults to 1.
        """
        super().__init__(pop_size, generations, mutpb, cxpb, cores)
        self._pop_size = pop_size
        self._generations = generations
        self._mutpb = mutpb
        self._cxpb = cxpb
        self._n_cores = cores
        self.__name__ = (
            f"ParEAKP_PS_{self._pop_size}_CXPB_{self._cxpb}_MUTPB_{self._mutpb}"
        )

    def __call__(self, problem: Knapsack, *args, **kwargs) -> list[Solution]:
        """Runs the algorithm to solve the KP problem

        Args:
            kp (Knapsack, optional): Instance of a KP. Defaults to None.

        Raises:
            AttributeError: If no instance is given

        Returns:
            List[Solution]: Best solution found by the algorithm
        """
        if problem is None:
            msg = "Knapsack Problem is None in ParEAKP.__call__(). Expected a Knapsack instance."
            raise AttributeError(msg)
        x, fitness = self.run(
            len(problem), problem.weights, problem.profits, problem.capacity
        )
        return [Solution(chromosome=x, objectives=(fitness,), fitness=fitness)]
