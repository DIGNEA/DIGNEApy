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

__all__ = ["two_opt", "three_opt"]


cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()  # add this line

from collections import Counter
from digneapy._core import Solution

ctypedef long int li
ctypedef unsigned short ushort

@cython.boundscheck(False)  # Deactivate bounds checking
cdef float evaluate(cnp.ndarray individual, cnp.ndarray distances):
    cdef float distance = 0.0
    cdef float fitness = 0.0
    cdef ushort i
    cdef ushort N = len(individual)
    counter = Counter(individual)

    if any(counter[c] != 1 for c in counter if c != 0) or (individual[0] != 0 or individual[-1] != 0):
        return 2.938736e-39 # --> 1.0 / np.float.max 
    else:
        for i in range(N - 2):
            distance += distances[individual[i]][individual[i + 1]]

        fitness = 1.0 / (distance)
        return fitness

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef list two_opt(object problem):
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

    cdef ushort N = problem.dimension
    cdef cnp.ndarray[ushort] tour = np.arange(start=0, stop=N+1, step=1, dtype=np.uint16)
    tour[N] = 0
    cdef ushort tour_length = len(tour)
    cdef cnp.ndarray distances = problem._distances
    cdef bint improve = True
    cdef ushort i, j
    cdef double current, newer
    cdef float fitness
    while improve:
        improve = False
        for i in range(1, tour_length - 2):
            for j in range(i + 2, tour_length):
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
    fitness = evaluate(tour, distances)
    return [Solution(variables=tour, objectives=(fitness,), fitness=fitness)]


@cython.boundscheck(False)
cpdef list three_opt(object problem):
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
    cdef ushort N = problem.dimension 
    cdef cnp.ndarray[ushort] tour = np.arange(start=0, stop=N+1, step=1, dtype=np.uint16)
    cdef cnp.ndarray distances = problem._distances
    cdef ushort i, j
    cdef bint improve = True
    tour[-1] = 0
    while improve:
        improve = False
        for i in range(1, N - 2):
            for j in range(i + 2, N - 1):
                for k in range(j + 2, N):
                    new_tour = np.concatenate((
                        tour[:i],
                        tour[j:k][::-1],
                        tour[i:j],
                        tour[k:],
                    ))

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
    return [
        Solution(variables=tour, objectives=(fitness,), fitness=np.float64(fitness))
    ]
