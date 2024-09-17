#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   testing_mpw.py
@Time    :   2024/09/12 11:39:13
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from digneapy.domains import kp
from digneapy.solvers.kp import mpw_kp


def generate_instance():
    p = list(range(1, 101))
    w = list(range(1, 101))
    q = 50
    return kp.Knapsack(p, w, q)


def test_mpw_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = mpw_kp(default_instance)[0]
    expected_p = 50
    expected_chromosome = [0.0] * 49 + [1.0] + [0.0] * 50
    print(solution)
    print(solution.chromosome)
    assert len(solution) == len(default_instance)
    assert solution.fitness == expected_p
    assert solution.objectives == (expected_p,)


if __name__ == "__main__":
    kp = generate_instance()
    test_mpw_kp_heuristic(kp)
