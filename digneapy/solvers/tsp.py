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

__all__ = ["two_opt", "nearest_neighbour", "shortest_edge"]
from typing import List, LiteralString

import numpy as np

from digneapy.core import Solution
from digneapy.domains.tsp import TSP

from ._c_tsp_solvers import ctwo_opt


def reconstruct_tour(
    adjacency: List[List[int]], start_node: int, n_nodes: int
) -> np.ndarray:
    """Reconstructs a TSP tour from a adjacency list

    Args:
        adjacency (List[List[int]]): Adjacency list for each node in the instance.
        start_node (int): Starting point for the tour. This value is used as the
            ending node as well.
        n_nodes (int): Number of nodes in the instance.

    Returns:
        np.ndarray: Tour of n_nodes + 1 nodes created from the adjacency list
    """
    tour = [start_node]
    visited = [False] * n_nodes
    visited[start_node] = True
    current = start_node
    for _ in range(n_nodes - 1):
        next_node = next(n for n in adjacency[current] if not visited[n])
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    return np.asarray(tour, dtype=np.uint32)


def two_opt(
    problem: TSP, start_node: int = 0, init: LiteralString = "random"
) -> list[Solution]:
    """Solve a TSP instance using the 2-opt local search heuristic.

    Constructs an initial tour and then improves it iteratively by repeatedly
    swapping pairs of edges whenever doing so reduces the total tour length.
    The process continues until no improving 2-opt swap can be found, at which
    point the solution is a local optimum with respect to 2-opt neighbourhood.
    The core improvement loop is delegated to the Cython-accelerated
    ``ctwo_opt`` implementation.

    Args:
        problem (TSP): The TSP instance to solve.
        start_node (int, optional): Index of the city used as the departure
            and return point of the tour. Defaults to 0.
        init (LiteralString, optional): Strategy used to build the initial
            tour before the 2-opt improvement phase begins. Accepted values
            are:

            - ``"random"``: generates a random feasible tour via
              ``problem.create_solution(random=True, start_node=start_node)``.
            - ``"nearest_neighbour"``: builds a greedy tour by always
              travelling to the closest unvisited city next, via
              ``nearest_neighbour(problem, start_node)``.

            Defaults to ``"random"``.

    Raises:
        ValueError: If ``init`` is not ``"random"`` or ``"nearest_neighbour"``.
        RuntimeError: If any exception is raised during tour construction or
            the 2-opt improvement phase, wrapping the original exception as
            its cause.

    Returns:
        list[Solution]: The list of solutions returned by ``ctwo_opt``,
            typically containing a single locally optimal tour.
    """
    if init not in ("random", "nearest_neighbour"):
        raise ValueError(
            f"init in two_opt must be random or nearest_neighbour. Got: {init}"
        )
    try:
        if init == "random":
            initial_solution = problem.create_solution(
                random=True, start_node=start_node
            )
        else:
            initial_solution = nearest_neighbour(
                problem=problem, start_node=start_node
            )[0]
        return ctwo_opt(initial_solution, problem)
    except Exception as e:
        raise RuntimeError(
            f"Cannot solve TSP in two_opt because an exception was raised. Cause {e}"
        )


def shortest_edge(problem: TSP, start_node: int = 0) -> list[Solution]:
    """The Shortest Edge Greedy heuristic for TSP

    It gradually constructs a tour by repeatedly selecting the
    shortest edge and adding it to the tour as long as it doesn’t
    create a cycle with less than N edges, or increases the degree of
    any node to more than 2.
    We must not add the same edge twice of course.

       Args:
           problem (TSP): Problem to solve

       Raises:
           ValueError: If problem is None

       Returns:
           list[Solution]: Collection of solutions for the given problem
    """
    try:
        N = problem.dimension
        if not (0 <= start_node < N):
            raise ValueError(f"start_node must be in [0, {N}), got {start_node}")

        distances = problem._distances
        parent = list(range(N))
        rank = [0] * N

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # path compression
                parent[x], x = root, parent[x]
            return root

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        ordered_edges = sorted([
            (distances[i][j], i, j) for i in range(N) for j in range(i + 1, N)
        ])

        length = np.float64(0.0)
        selected_edges = 0
        degree = [0] * N
        adjacency: list[list[int]] = [[] for _ in range(N)]
        for dist, i, j in ordered_edges:
            if degree[i] >= 2 or degree[j] >= 2:
                continue

            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                if selected_edges != N - 1:
                    continue
            union(i, j)
            degree[i] += 1
            degree[j] += 1

            adjacency[i].append(j)
            adjacency[j].append(i)
            length += dist
            selected_edges += 1

            if selected_edges == N:
                break

        tour = reconstruct_tour(adjacency=adjacency, start_node=start_node, n_nodes=N)
        _fitness = 1.0 / length
        return [
            Solution(
                variables=tour,
                objectives=(_fitness,),
                fitness=_fitness,
                constraints=(0,),
            )
        ]
    except Exception as e:
        raise RuntimeError(
            f"Cannot solve TSP in shortest_edge because an exception was raised. Cause {e}"
        )


def nearest_neighbour(problem: TSP, start_node: int = 0) -> list[Solution]:
    """Nearest-Neighbour Heuristic for the Travelling Salesman Problem

    Args:
        problem (TSP): Problem to solve

    Raises:
        ValueError: If problem is None

    Returns:
        list[Solution]: Collection of solutions to the problem.
    """
    try:
        distances = problem._distances
        current_node = start_node
        visited_nodes: set[int] = set([current_node])
        tour = np.zeros(problem.dimension, dtype=np.uint32)

        length = np.float64(0.0)
        # The start node of the tour
        tour[0] = start_node
        index = 1
        N = problem.dimension
        while len(visited_nodes) != N:
            next_node = -1
            min_distance = np.inf

            for j in range(N):
                if j not in visited_nodes and distances[current_node][j] < min_distance:
                    min_distance = distances[current_node][j]
                    next_node = j

            visited_nodes.add(next_node)
            length += min_distance
            tour[index] = next_node
            index += 1
            current_node = next_node

        # Closing edge: wrap to tour[start_node]
        length += distances[current_node][start_node]
        fitness = 1.0 / length

        return [
            Solution(
                variables=tour,
                objectives=(fitness,),
                fitness=fitness,
                constraints=(0,),
            )
        ]
    except Exception as e:
        raise RuntimeError(
            f"Cannot solve TSP in nneighbour because an exception was raised. Cause {e}"
        )
