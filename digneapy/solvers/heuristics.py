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


from ..core import Solver
from digneapy.domains.knapsack import Knapsack
import numpy as np


class DefaultKP(Solver):
    def __init__(self):
        super().__init__("Default_KP")

    def run(self, problem: Knapsack = None, **kwargs):
        if problem is None:
            msg = f"No problem found in args of {self.__class__.__name__} run method"
            raise AttributeError(msg)

        inside = 0
        profit = 0
        chromosome = np.zeros(len(problem))
        idx = 0
        while idx < len(problem):
            if problem.weights[idx] + inside <= problem.capacity:
                inside += problem.weights[idx]
                profit += problem.profits[idx]
                chromosome[idx] = 1
            idx += 1
        return (profit, chromosome)


class MaP(Solver):
    def __init__(self):
        super().__init__("MaP")

    def run(self, problem: Knapsack = None, **kwargs):
        if problem is None:
            msg = f"No problem found in args of {self.__class__.__name__} run method"
            raise AttributeError(msg)

        indices = np.argsort(problem.profits)[::-1]

        inside = 0
        profit = 0
        chromosome = np.zeros(len(problem))
        for idx in indices:
            if problem.weights[idx] + inside <= problem.capacity:
                inside += problem.weights[idx]
                profit += problem.profits[idx]
                chromosome[idx] = 1
        return (profit, chromosome)


class MiW(Solver):
    def __init__(self):
        super().__init__("MiW")

    def run(self, problem: Knapsack = None, **kwargs):
        if problem is None:
            msg = f"No problem found in args of {self.__class__.__name__} run method"
            raise AttributeError(msg)

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

        return (profit, chromosome)


class MPW(Solver):
    def __init__(self):
        super().__init__("MPW")

    def run(self, problem: Knapsack = None, **kwargs):
        if problem is None:
            msg = f"No problem found in args of {self.__class__.__name__} run method"
            raise AttributeError(msg)

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

        return (profit, chromosome)
