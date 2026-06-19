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

__all__ = ["ctwo_opt"]


cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()  # add this line

from digneapy.core import Solution

ctypedef long int li
ctypedef unsigned short ushort

cpdef ctwo_opt(object problem):
    if problem is None:
        raise RuntimeError("Cannot solve TSP in 2-Opt because problem is None.")

    cdef int number_of_nodes = problem.dimension
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] tour = np.arange(problem.dimension + 1, dtype=np.uint32)
    tour[number_of_nodes] = 0

    cdef double[:,:] distances = problem._distances
    cdef bool improved = True
    cdef double min_change, change
    cdef int i, j, min_i, min_j
    min_change = 0.0
    while improved:
        improved = False
        min_change = 0.0

        # Find the best move
        for i in range(number_of_nodes - 2):
            for j in range(i + 2, number_of_nodes - 1):
                change = distances[i, j] + distances[i + 1, j + 1]
                change -= distances[i, i + 1] - distances[j, j + 1]
                if change < min_change:
                    min_change = change
                    min_i, min_j = i, j
                    improved = True
        # Update tour with best move
        if min_change < 0:
            tour[min_i + 1 : min_j + 1] = tour[min_i + 1 : min_j + 1][::-1]

    fitness, cycled, duplicated = problem.evaluate(tour)
    return [
            Solution(
                variables=tour,
                objectives=(fitness,),
                fitness=fitness,
                constraints=(cycled, duplicated),
            )
    ]

