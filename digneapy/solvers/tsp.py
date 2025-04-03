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

__all__ = ["nneighbour", "greedy", "three_opt"]

from collections import Counter

import numpy as np

from digneapy import Solution
from digneapy.domains.tsp import TSP


def greedy(problem: TSP, *args, **kwargs) -> list[Solution]:
    """The Greedy heuristic gradually constructs a tour by
    repeatedly selecting the shortest edge and adding it to the tour as long as
    it doesn’t create a cycle with less than N edges, or increases the degree of
    any node to more than 2. We must not add the same edge twice of course.

       Args:
           problem (TSP): Problem to solve

       Raises:
           ValueError: If problem is None

       Returns:
           list[Solution]: Collection of solutions for the given problem
    """
    if problem is None:
        raise ValueError("No problem found in two_opt heuristic")
    N = problem.dimension
    distances = problem._distances
    counter = Counter()
    selected: set[tuple[int, int]] = set()

    ordered_edges = sorted(
        [(distances[i][j], i, j) for i in range(N) for j in range(i + 1, N)]
    )

    length = 0.0
    for dist, i, j in ordered_edges:
        if (i, j) in selected or (j, i) in selected:
            continue
        if counter[i] >= 2 or counter[j] >= 2:
            continue
        selected.add((i, j))
        counter[i] += 1
        counter[j] += 1
        length += dist
        if len(selected) == N:
            break

    _fitness = 1.0 / length

    return [
        Solution(
            chromosome=list(range(N + 1)), objectives=(_fitness,), fitness=_fitness
        )
    ]


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
    tour = np.zeros(problem.dimension + 1)
    length = np.float32(0)
    idx = 1
    while len(visited_nodes) != problem.dimension:
        next_node = 0
        min_distance = np.finfo(np.float32).max

        for j in range(problem.dimension):
            if j not in visited_nodes and distances[current_node][j] < min_distance:
                min_distance = distances[current_node][j]
                next_node = j

        visited_nodes.add(next_node)
        tour[idx] = next_node
        idx += 1
        length += min_distance
        current_node = next_node

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
    distances = problem._distances
    N = problem.dimension
    tour = np.arange(start=0, stop=N + 1, step=1, dtype=int)
    tour[-1] = 0
    improve = True
    while improve:
        improve = False
        for i in range(1, N - 2):
            for j in range(i + 2, N - 1):
                for k in range(j + 2, N):
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

    fitness = problem.evaluate(tour)[0]
    return [Solution(chromosome=tour, objectives=(fitness,), fitness=fitness)]
