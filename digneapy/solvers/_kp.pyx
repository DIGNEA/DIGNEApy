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

__all__ = ["default_kp", "map_kp", "miw_kp", "mpw_kp"]


cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()  # add this line


from digneapy._core import Solution
from digneapy.domains.kp import Knapsack


ctypedef long int li
ctypedef long long int lli

cpdef list default_kp(problem: Knapsack):

    cdef li q, N, packed, profit, i
    cdef li [:] p, w, chromosome 

    q = problem.capacity
    N = len(problem)
    packed = 0
    profit = 0
    p = problem.profits
    w = problem.weights
    chromosome = np.zeros(N, dtype=int)
    
    for i in range(N):
        if packed + w[i] <= q:
            packed += w[i]
            profit += p[i]
            chromosome[i] = 1
    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


cpdef list map_kp(problem: Knapsack):
    
    cdef li q, N, packed, profit, i
    cdef li [:] p, w, chromosome, indices

    q = problem.capacity
    N = len(problem)
    packed = 0
    profit = 0
    p = problem.profits
    w = problem.weights
    chromosome = np.zeros(N, dtype=int)
    indices = np.argsort(p)[::-1]

    for i in indices:
        if packed + w[i] <= q:
            chromosome[i] = 1
            packed += w[i]
            profit += p[i]
        if packed >= q:
            break

    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


cpdef list miw_kp(problem: Knapsack):
    
    cdef li profit, i

    indices = np.argsort(problem.weights)
    chromosome = np.zeros(len(problem), dtype=int)
    weights_cumsum = np.cumsum(problem.weights[indices])
    selected = indices[weights_cumsum <= problem.capacity]
    chromosome[selected] = 1
    profit = np.sum(problem.profits[selected])
    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]


cpdef list mpw_kp(problem: Knapsack):
    
    cdef li q, N, packed, profit, i
    cdef li [:] p, w, chromosome, indices
    q = problem.capacity
    N = len(problem)
    packed = 0
    profit = 0
    p = problem.profits
    w = problem.weights
    chromosome = np.zeros(N, dtype=int)
    indices = np.argsort([(p[i] / w[i]) for i in range(N)])[::-1]

    for idx in indices:
        if packed + w[idx] <= q:
            packed += w[idx]
            profit += p[idx]
            chromosome[idx] = 1

    return [Solution(chromosome=chromosome, objectives=(profit,), fitness=profit)]
