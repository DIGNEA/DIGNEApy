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

import numpy as np

from digneapy import Direction
from digneapy.domains import Knapsack
from digneapy.solvers.evolutionary import EA
from digneapy.utils import clock


@clock
def clocked_solving(solver, kp):
    return solver(kp)


def main():
    N = 100
    p = list(np.random.randint(1, 100 + 1, size=N))
    w = list(np.random.randint(1, 100 + 1, size=N))
    q = np.random.randint(0, high=250)
    kp = Knapsack(profits=p, weights=w, capacity=q)
    solver = EA(
        direction=Direction.MAXIMISE,
        dim=N,
        pop_size=32,
        cxpb=0.8,
        mutpb=(1.0 / 100.0),
        generations=1000,
        min_g=0,
        max_g=1,
    )
    solutions = clocked_solving(solver, kp)


if __name__ == "__main__":
    main()
