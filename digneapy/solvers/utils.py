#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   special_cases.py
@Time    :   2025/10/14 14:41:38
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from typing import Callable
from digneapy.domains import Knapsack
import numpy as np
import random

__all__ = ["shuffle_and_run_for_knapsack"]


def shuffle_and_run_for_knapsack(
    solver,
    n_repetitions: int = 10,
    reduce_fn: Callable = np.mean,
    verbose: bool = False,
):
    def solve(problem: Knapsack):
        results = np.zeros(n_repetitions, dtype=np.float64)
        fitness = solver(problem)[0].fitness
        results[0] = fitness
        variables = np.asarray(problem, copy=True, dtype=int)
        items = [(w, p) for w, p in zip(variables[1::2], variables[2::2])]
        for r in range(1, n_repetitions):
            shuffled_items = np.asarray(random.sample(items, k=len(items)), dtype=int)
            shuffled_problem = Knapsack(
                weights=shuffled_items[:, 0],
                profits=shuffled_items[:, 1],
                capacity=variables[0],
            )
            results[r] = solver(shuffled_problem)[0].fitness
        fitness = reduce_fn(results)
        if verbose:
            print(
                f"Shuffle {n_repetitions} times and reduce with {reduce_fn.__name__}",
                end="",
            )
            print(f", results: {results}, final fitness: {fitness}")
        return fitness

    return solve
