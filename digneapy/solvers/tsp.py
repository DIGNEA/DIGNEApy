#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   tsp.py
@Time    :   2025/03/05 09:32:55
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["two_opt", "nneighbour", "three_opt"]


import numpy as np

from digneapy._core import Solution
from digneapy.domains.tsp import TSP


def two_opt(problem: TSP, *args, **kwargs) -> list[Solution]:
    """2-Opt Heuristic for the Travelling Salesman Problem

    Args:
        problem (TSP): problem to solve

    Raises:
        ValueError: If problem is None

    Returns:
        list[Solution]: Collection of solutions
    """
    if problem is None:
        raise ValueError("No problem found in two_opt heuristic")
    tour = [0] + list(range(1, problem.dimension)) + [0]

    distances = problem._distances
    improve = True
    while improve:
        improve = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1:
                    continue
                current = (
                    distances[tour[i - 1]][tour[j - 1]] + distances[tour[i]][tour[j]]
                )
                newer = (
                    distances[tour[i - 1]][tour[i]] + distances[tour[j - 1]][tour[j]]
                )

                if newer < current:
                    tour[i:j] = tour[j - 1 : i - 1 : -1]
                    improve = True

    _fitness = problem.evaluate(tour)[0]
    return [Solution(chromosome=tour, objectives=(_fitness,), fitness=_fitness)]


def nneighbour(problem: TSP, *args, **kwargs) -> list[Solution]:
    """Nearest-Neighbour Heuristic for the Travelling Salesman Problem

    Args:
        problem (TSP): Problem to solve

    Raises:
        ValueError: If problem is None

    Returns:
        list[Solution]: Collection of solutions to the problem.
    """
    if problem is None:
        raise ValueError("No problem found in nneighbour heuristic")

    distances = problem._distances
    current_node = 0
    visited_nodes: set[int] = set([current_node])
    tour = [0]
    length = 0.0
    while len(visited_nodes) != problem.dimension:
        next_node = 0
        min_distance = np.inf

        for j in range(problem.dimension):
            if j not in visited_nodes and distances[current_node][j] < min_distance:
                min_distance = distances[current_node][j]
                next_node = j

        visited_nodes.add(next_node)
        tour.append(next_node)
        length += min_distance
        current_node = next_node
    tour.append(0)
    length += distances[current_node][0]
    length = 1.0 / length
    return [Solution(chromosome=tour, objectives=(length,), fitness=length)]


def three_opt(problem: TSP, *args, **kwargs) -> list[Solution]:
    """3-Opt Heuristic for the Travelling Salesman Problem

    Args:
        problem (TSP): Problem to solve

    Raises:
        ValueError: If problem is None

    Returns:
        list[Solution]: Collection of solutions to the problem
    """
    if problem is None:
        raise ValueError("No problem found in three_opt heuristic")
    tour = [0] + list(range(1, problem.dimension)) + [0]
    distances = problem._distances
    N = problem.dimension
    improve = True
    while improve:
        improve = False
        for i in range(1, N - 2):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N):
                    new_tour = tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:]

                    current = (
                        distances[tour[i - 1]][tour[i]]
                        + distances[tour[j - 1]][tour[j]]
                        + distances[tour[k - 1]][tour[k]]
                    )
                    newer = (
                        distances[new_tour[-2]][new_tour[-1]]
                        + distances[new_tour[0]][new_tour[1]]
                    )

                    if newer < current:
                        tour = new_tour
                        improve = True

    _fitness = problem.evaluate(tour)[0]
    return [Solution(chromosome=tour, objectives=(_fitness,), fitness=_fitness)]
