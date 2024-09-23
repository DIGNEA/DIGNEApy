#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   heuristics.py
@Time    :   2023/10/30 11:59:03
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["default_kp", "map_kp", "miw_kp", "mpw_kp"]

import numpy as np

from digneapy._core import Solution
from digneapy.domains.kp import Knapsack


def default_kp(problem: Knapsack, *args, **kwargs) -> list[Solution]:
    if problem is None:
        msg = "No problem found in args of default_kp heuristic"
        raise ValueError(msg)

    inside = 0
    profit = 0
    chromosome = np.zeros(len(problem))
    for idx in range(len(problem)):
        if problem.weights[idx] + inside <= problem.capacity:
            inside += problem.weights[idx]
            profit += problem.profits[idx]
            chromosome[idx] = 1
    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


def map_kp(problem: Knapsack, *args, **kwargs) -> list[Solution]:
    if problem is None:
        msg = "No problem found in args of map_kp heuristic"
        raise ValueError(msg)

    indices = np.argsort(problem.profits)[::-1]

    inside = 0
    profit = 0
    chromosome = np.zeros(len(problem))
    for idx in indices:
        if problem.weights[idx] + inside <= problem.capacity:
            inside += problem.weights[idx]
            profit += problem.profits[idx]
            chromosome[idx] = 1
    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


def miw_kp(problem: Knapsack, *args, **kwargs) -> list[Solution]:
    if problem is None:
        msg = "No problem found in args of miw_kp heuristic"
        raise ValueError(msg)

    indices = np.argsort(problem.weights)

    inside = 0
    profit = 0
    chromosome = np.zeros(len(problem))
    for idx in indices:
        if problem.weights[idx] + inside <= problem.capacity:
            inside += problem.weights[idx]
            profit += problem.profits[idx]
            chromosome[idx] = 1
        else:
            break

    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


def mpw_kp(problem: Knapsack, *args, **kwargs) -> list[Solution]:
    if problem is None:
        msg = "No problem found in args of mpw_kp heuristic"
        raise ValueError(msg)

    profits_per_weights = [(p / w) for p, w in zip(problem.profits, problem.weights)]
    indices = np.argsort(profits_per_weights)[::-1]
    inside = 0
    profit = 0
    chromosome = np.zeros(len(problem))
    for idx in indices:
        if problem.weights[idx] + inside <= problem.capacity:
            inside += problem.weights[idx]
            profit += problem.profits[idx]
            chromosome[idx] = 1

    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]
