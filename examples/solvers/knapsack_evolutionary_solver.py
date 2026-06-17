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
def clocked_solving(solver: EA, kp: Knapsack):
    return solver(kp)


def main():
    number_of_items = 100
    rng = np.random.default_rng()
    profits = rng.integers(low=1, high=1000, size=number_of_items, dtype=np.uint32)
    weights = rng.integers(low=1, high=1000, size=number_of_items, dtype=np.uint32)
    capacity = rng.integers(low=1, high=1e5, size=1, dtype=np.uint64)[0]
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)
    solver = EA(
        direction=Direction.MAXIMISE,
        dim=number_of_items,
        pop_size=32,
        cxpb=0.8,
        mutpb=(1.0 / number_of_items),
        generations=1000,
        min_g=0,
        max_g=1,
    )
    _ = clocked_solving(solver, knapsack)


if __name__ == "__main__":
    main()
