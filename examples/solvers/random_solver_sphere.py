#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   random_solver_sphere.py
@Time    :   2026/05/20 15:42:54
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from digneapy.domains import Sphere
from digneapy.solvers import random_solver

if __name__ == "__main__":
    N = 5
    ITERATIONS = 100
    problem = Sphere(N)
    seed = np.random.SeedSequence(4213)
    seed_iteration = seed.spawn(ITERATIONS + 1)
    for i in range(1, ITERATIONS + 1):
        solution = random_solver(problem, dtype=np.float64, seed=seed_iteration[i])
        print(f"It {i} -> {solution})")
