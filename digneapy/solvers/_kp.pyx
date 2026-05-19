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
cnp.import_array() 


from digneapy._core import Solution
from digneapy.domains.kp import Knapsack


ctypedef unsigned short ushort
ctypedef unsigned int uint # 16b [0, 65,535]
ctypedef unsigned int uli # 32b [0, 4,294,967,295]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cpdef list default_kp(problem: Knapsack):
    if problem is None:
        msg = "No problem found in args of default_kp heuristic"
        raise ValueError(msg)

    cdef uli q, packed, profit
    cdef uli [:] p, w
    cdef uint N, i
    cdef unsigned short [:] variables
    

    q = problem.capacity
    N = len(problem)
    packed = 0
    profit = 0
    p = problem.profits
    w = problem.weights
    variables = np.zeros(N, dtype=np.uint16)
    
    for i in range(N):
        if packed + w[i] <= q:
            packed += w[i]
            profit += p[i]
            variables[i] = 1
    return [Solution(variables=variables, objectives=(profit,), fitness=np.float64(profit))]


cpdef list map_kp(problem: Knapsack):
    if problem is None:
        msg = "No problem found in args of map_kp heuristic"
        raise ValueError(msg)    
    cdef uint q, packed
    cdef uint N, i
    cdef uli profit
    cdef uint [:] p, w
    cdef unsigned short [:] variables
    cdef long [:] indices

    q = problem.capacity
    N = len(problem)
    packed = 0
    profit = 0
    p = problem.profits
    w = problem.weights
    variables = np.zeros(N, dtype=np.uint16)
    indices = np.argsort(p)[::-1]

    for i in indices:
        if packed + w[i] <= q:
            variables[i] = 1
            packed += w[i]
            profit += p[i]
        if packed >= q:
            break

    return [Solution(variables=variables, objectives=(profit,), fitness=np.float64(profit))]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cpdef list miw_kp(problem: Knapsack):
    if problem is None:
        msg = "No problem found in args of miw_kp heuristic"
        raise ValueError(msg)    

    cdef uli profit
    cdef uint N, q
    cdef cnp.ndarray[uint, ndim=1] variables
    cdef cnp.ndarray[long, ndim=1] indices  
    cdef cnp.ndarray[uint, ndim=1] np_w = np.asarray(problem.weights)
    cdef cnp.ndarray[uint, ndim=1] np_p = np.asarray(problem.profits)

    q = problem.capacity
    N = len(problem)

    indices = np.argsort(np_w)
    variables = np.zeros(N, dtype=np.uint32)
    weights_cumsum = np.cumsum(np_w[indices])
    selected = indices[weights_cumsum <= q]
    variables[selected] = 1
    profit = np.sum(np_p[selected])
    return [Solution(variables=variables, objectives=(profit,), fitness=np.float64(profit))]


cpdef list mpw_kp(problem: Knapsack):
    if problem is None:
        raise ValueError("No problem found in args of mpw_kp heuristic")    
 
    cdef uint q, packed
    cdef uint N, i
    cdef uli profit
    cdef uint [:] p, w
    cdef unsigned short [:] variables
    cdef long [:] indices

    q = problem.capacity
    N = len(problem)
    packed = 0
    profit = 0
    p = problem.profits
    w = problem.weights
    variables = np.zeros(N, dtype=np.uint16)
    indices = np.argsort([(p[i] / w[i]) for i in range(N)])[::-1]

    for i in indices:
        if packed + w[i] <= q:
            packed += w[i]
            profit += p[i]
            variables[i] = 1

    return [Solution(variables=variables, objectives=(profit,), fitness=profit)]
