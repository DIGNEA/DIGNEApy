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


from digneapy.domains.knapsack import Knapsack
from typing import Tuple, List
import numpy as np


def default_kp(problem: Knapsack = None) -> Tuple[float, List[int]]:
    if problem is None:
        msg = "No problem found in args of default_kp heuristic"
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


def map_kp(problem: Knapsack = None) -> Tuple[float, List[int]]:
    if problem is None:
        msg = "No problem found in args of map_kp heuristic"
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


def miw_kp(problem: Knapsack = None) -> Tuple[float, List[int]]:
    if problem is None:
        msg = "No problem found in args of miw_kp heuristic"
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


def mpw_kp(problem: Knapsack = None) -> Tuple[float, List[int]]:
    if problem is None:
        msg = "No problem found in args of mpw_kp heuristic"
        raise AttributeError(msg)

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
        else:
            break

    return (profit, chromosome)
