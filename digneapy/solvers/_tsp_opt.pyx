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

__all__ = ["two_opt"]


cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()  # add this line

from collections import Counter
from digneapy._core import Solution

ctypedef long int li

@cython.boundscheck(False)  # Deactivate bounds checking
cdef float evaluate(cnp.ndarray individual, cnp.ndarray distances):
    cdef float distance = 0.0
    cdef float fitness = 0.0
    cdef li i
    cdef li N = len(individual)
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

    cdef li N = problem.dimension
    cdef cnp.ndarray[li] tour = np.arange(start=0, stop=N+1, step=1)
    tour[N] = 0
    cdef li tour_length = len(tour)
    cdef cnp.ndarray distances = problem._distances
    cdef bint improve = True
    cdef li i, j
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
    return [Solution(chromosome=tour, objectives=(fitness,), fitness=fitness)]
