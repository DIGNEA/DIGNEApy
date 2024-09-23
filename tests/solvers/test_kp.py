#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_kp.py
@Time    :   2024/09/18 14:00:00
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest
from deap import benchmarks

from digneapy import Direction
from digneapy._core import Solution
from digneapy.domains import kp as knapsack
from digneapy.solvers.kp import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.solvers.pisinger import combo, expknap, minknap


@pytest.fixture
def default_instance():
    p = list(range(1, 101))
    w = list(range(1, 101))
    q = 50
    return knapsack.Knapsack(p, w, q)


@pytest.fixture
def default_large_knap():
    c = np.random.randint(1e3, 1e5)
    w = np.random.randint(1000, 5000, size=1000, dtype=np.int32)
    p = np.random.randint(1000, 5000, size=1000, dtype=np.int32)
    kp = knapsack.Knapsack(profits=p, weights=w, capacity=c)
    return kp


def test_default_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = default_kp(default_instance)[0]
    expected_p = sum(default_instance.profits[:9])
    expected_chromosome = [1.0] * 9 + [0.0] * 91
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert solution.chromosome == expected_chromosome

    with pytest.raises(Exception):
        default_kp(None)


def test_map_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = map_kp(default_instance)[0]
    expected_p = 50
    expected_chromosome = [0] * 49 + [1] + [0] * 50
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert solution.chromosome == expected_chromosome

    with pytest.raises(ValueError):
        map_kp(None)


def test_miw_kp_heuristic(default_instance):
    """
    MiW should works exactly as Default
    """
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = miw_kp(default_instance)[0]
    expected_p = sum(default_instance.profits[:9])
    expected_chromosome = [1.0] * 9 + [0.0] * 91
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert solution.chromosome == expected_chromosome

    with pytest.raises(ValueError):
        miw_kp(None)


def test_mpw_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = mpw_kp(default_instance)[0]
    expected_p = 50
    expected_chromosome = [0.0] * 49 + [1.0] + [0.0] * 50
    assert solution.chromosome == expected_chromosome
    assert len(solution) == len(default_instance)
    assert solution.fitness == expected_p
    assert solution.objectives == (expected_p,)

    with pytest.raises(ValueError):
        mpw_kp(None)


def test_combo(default_large_knap):
    solutions = combo(default_large_knap)
    assert solutions[0].fitness <= 1.0  # Here compares time
    solutions = combo(default_large_knap, only_time=False)
    assert len(solutions) == 1
    assert all(isinstance(i, Solution) for i in solutions)
    assert solutions[0].fitness >= 0.0
    assert len(solutions[0]) == 1000


def test_minknap(default_large_knap):
    solutions = minknap(default_large_knap)
    assert solutions[0].fitness <= 1.0  # Here compares time
    solutions = minknap(default_large_knap, only_time=False)
    assert len(solutions) == 1
    assert all(isinstance(i, Solution) for i in solutions)
    assert solutions[0].fitness >= 0.0
    assert len(solutions[0]) == 1000


def test_expknap(default_large_knap):
    solutions = expknap(default_large_knap)
    assert solutions[0].fitness <= 15.0  # Here compares time (15.0s max time allowed)
    solutions = expknap(default_large_knap, only_time=False)
    assert len(solutions) == 1
    assert all(isinstance(i, Solution) for i in solutions)
    assert solutions[0].fitness >= 0.0
    assert len(solutions[0]) == 1000


def test_pisinger_are_exact(default_large_knap):
    r_exknap = expknap(default_large_knap, only_time=False)
    r_combo = combo(default_large_knap, only_time=False)
    r_minknap = minknap(default_large_knap, only_time=False)
    all_solutions = [*r_exknap, *r_combo, *r_minknap]
    expected = r_combo[0].fitness
    assert len(all_solutions) == 3
    assert all(isinstance(i, Solution) for i in all_solutions)
    assert all(i.fitness == expected for i in all_solutions)


def test_pisingers_raises():
    """
    Raises an exception because the the problem is None
    """
    with pytest.raises(ValueError):
        expknap(None)
    with pytest.raises(ValueError):
        combo(None)
    with pytest.raises(ValueError):
        minknap(None)
