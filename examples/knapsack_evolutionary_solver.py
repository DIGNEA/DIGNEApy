#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_evolutionary_solver.py
@Time    :   2024/4/11 11:41:13
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from digneapy.solvers import EA, MAXIMISE
import numpy as np
from digneapy.domains import knapsack


def main():
    N = 100
    p = list(np.random.randint(1, 100 + 1, size=N))
    w = list(np.random.randint(1, 100 + 1, size=N))
    q = np.random.randint(0, high=250)
    kp = knapsack.Knapsack(profits=p, weights=w, capacity=q)
    print(f"Instance to solve: {kp}")
    solver = EA(
        dir=MAXIMISE,
        dim=N,
        pop_size=32,
        cxpb=0.8,
        mutpb=(1.0 / 100.0),
        generations=1000,
    )
    solutions = solver(kp)
    print(solutions)


if __name__ == "__main__":
    main()
