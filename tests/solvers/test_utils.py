#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_utils.py
@Time    :   2025/10/30 11:50:50
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest

from digneapy._core import Solution
from digneapy.domains import kp as knapsack
from digneapy.solvers import default_kp, map_kp
from digneapy.solvers import shuffle_and_run_for_knapsack


@pytest.fixture
def knapsack_instance():
    N = 100
    p = np.arange(N)
    w = np.arange(N)
    Q = np.random.default_rng().integers(low=10_000, high=100_000, size=1)[0]
    return knapsack.Knapsack(p, w, Q)


def test_shuffle_for_def_solver(knapsack_instance):
    R = 10
    fitness = default_kp(knapsack_instance)[0].fitness
    solver = shuffle_and_run_for_knapsack(default_kp, n_repetitions=R)
    shuffled_fitness = solver(knapsack_instance)[0].fitness
    assert fitness >= 0
    assert shuffled_fitness >= 0
    assert shuffled_fitness >= fitness


def test_shuffle_for_map_does_not_alter(knapsack_instance):
    R = 10
    fitness = map_kp(knapsack_instance)[0].fitness
    solver = shuffle_and_run_for_knapsack(map_kp, n_repetitions=R)
    shuffled_fitness = solver(knapsack_instance)[0].fitness
    assert fitness >= 0
    assert shuffled_fitness >= 0
    assert shuffled_fitness == fitness
