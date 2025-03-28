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

    p = problem.profits
    w = problem.weights
    q = problem.capacity
    N = len(problem)
    chromosome = np.zeros(N, dtype=np.int8)
    packed = 0
    profit = 0
    for i in range(N):
        if packed + w[i] <= q:
            packed += w[i]
            profit += p[i]
            chromosome[i] = 1
    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


def map_kp(problem: Knapsack, *args, **kwargs) -> list[Solution]:
    if problem is None:
        msg = "No problem found in args of map_kp heuristic"
        raise ValueError(msg)

    w = problem.weights
    p = problem.profits
    q = problem.capacity
    chromosome = np.zeros(len(problem), dtype=np.int8)
    packed = 0
    profit = 0
    indices = np.argsort(-p)

    for i in indices:
        if packed + w[i] <= q:
            chromosome[i] = 1
            packed += w[i]
            profit += p[i]
        if packed >= q:
            break

    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


def miw_kp(problem: Knapsack, *args, **kwargs) -> list[Solution]:
    if problem is None:
        msg = "No problem found in args of map_kp heuristic"
        raise ValueError(msg)

    indices = np.argsort(problem.weights)
    chromosome = np.zeros(len(problem), dtype=np.int8)
    weights_cumsum = np.cumsum(problem.weights[indices])
    selected = indices[weights_cumsum <= problem.capacity]
    chromosome[selected] = 1
    profit = np.sum(problem.profits[selected])
    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


def mpw_kp(problem: Knapsack, *args, **kwargs) -> list[Solution]:
    if problem is None:
        msg = "No problem found in args of mpw_kp heuristic"
        raise ValueError(msg)
    w = problem.weights
    p = problem.profits
    q = problem.capacity
    N = len(w)
    indices = np.argsort([(p[i] / w[i]) for i in range(N)])[::-1]

    packed = 0
    profit = 0
    chromosome = np.zeros(len(problem), dtype=np.int8)
    for idx in indices:
        if packed + w[idx] <= q:
            packed += w[idx]
            profit += p[idx]
            chromosome[idx] = 1

    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]
