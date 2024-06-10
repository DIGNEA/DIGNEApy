#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   core.py
@Time    :   2024/05/29 11:16:49
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from pisinger_cpp import minknap_cpp, combo_cpp, expknap_cpp
from digneapy.domains.knapsack import Knapsack
from digneapy.core import Solution
from typing import List
import numpy as np


def minknap(problem: Knapsack = None, only_time: bool = True) -> List[Solution]:
    if problem is None:
        msg = "No problem found in args of minknap heuristic"
        raise AttributeError(msg)

    x = np.zeros(len(problem))
    time, best = minknap_cpp(
        len(problem), problem.profits, problem.weights, x, problem.capacity
    )
    f = time if only_time else best
    return [Solution(chromosome=x, objectives=(f,), fitness=f)]


def expknap(problem: Knapsack = None, only_time: bool = True) -> List[Solution]:
    if problem is None:
        msg = "No problem found in args of expknap heuristic"
        raise AttributeError(msg)

    x = np.zeros(len(problem))
    time, best = expknap_cpp(
        len(problem), problem.profits, problem.weights, x, problem.capacity
    )
    f = time if only_time else best
    return [Solution(chromosome=x, objectives=(f,), fitness=f)]


def combo(problem: Knapsack = None, only_time: bool = True) -> List[Solution]:
    if problem is None:
        msg = "No problem found in args of combo heuristic"
        raise AttributeError(msg)

    x = np.zeros(len(problem))
    time, best = combo_cpp(
        len(problem), problem.profits, problem.weights, x, problem.capacity
    )
    f = time if only_time else best
    return [Solution(chromosome=x, objectives=(f,), fitness=f)]
