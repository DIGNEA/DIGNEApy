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



cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()  # add this line

from digneapy.core import Solution

ctypedef long int li
ctypedef unsigned short ushort





@cython.boundscheck(False)  # Deactivate bounds checking
cpdef list ctwo_opt(object initial_solution, object problem):
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

    cdef cnp.uint16_t N = problem.dimension
    cdef cnp.ndarray[cnp.float64_t, ndim=2] distances = problem._distances
    cdef list tour = initial_solution.variables.tolist()
    cdef bint improve = True
    cdef cnp.uint16_t i, j
    cdef cnp.float64_t length, delta, old_edges, new_edges
    cdef cnp.float64_t cycled, duplicated  # adjust dtype if evaluate returns ints
    cdef int a, b, c, d
    # Initial solution is the solution to beat
    length = 1.0 / initial_solution.fitness
    # Initial solution is the solution to beat
    while improve:
        improve = False
        for i in range(N - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, N):
                c, d = tour[j - 1], tour[j]
                # Closing edge: wrap d around to tour[0] when j == N
                d = tour[0] if j == N else tour[j]
                # Edges removed by the swap: (a,b) and (c,d)
                # Edges added by the swap:   (a,c) and (b,d)
                old_edges = distances[a][b] + distances[c][d]
                new_edges = distances[a][c] + distances[b][d]
                delta = old_edges - new_edges
                if delta > 0:
                    tour[i + 1 : j] = tour[i + 1 : j][::-1]
                    length -= delta
                    improve = True
                    b = tour[i+1]
    fitness = 1.0 / length 
    return [Solution(variables=tour, 
                objectives=(fitness,), 
                fitness=fitness, 
                constraints=(0,))]

