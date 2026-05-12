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

from digneapy.domains import kp as knapsack
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp

# from digneapy.solvers.pisinger import combo, expknap, minknap


@pytest.fixture
def default_instance():
    p = np.asarray(list(range(1, 101)))
    w = np.asarray(list(range(1, 101)))
    q = 50
    return knapsack.Knapsack(p, w, q)


@pytest.fixture
def default_large_knap():
    rng = np.random.default_rng(seed=42)
    c = rng.integers(1e3, 1e5)
    w = rng.integers(1000, 5000, size=1000)
    p = rng.integers(1000, 5000, size=1000)
    kp = knapsack.Knapsack(profits=p, weights=w, capacity=c)

    return kp


def test_default_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = default_kp(default_instance)[0]
    expected_p = sum(default_instance.profits[:9])
    expected_variables = [1.0] * 9 + [0.0] * 91
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert np.array_equal(solution.variables, np.asarray(expected_variables))

    with pytest.raises(Exception):
        default_kp(None)


def test_map_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = map_kp(default_instance)[0]
    expected_p = 50
    expected_variables = [0] * 49 + [1] + [0] * 50
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert np.array_equal(solution.variables, np.asarray(expected_variables))

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
    expected_variables = [1.0] * 9 + [0.0] * 91
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert np.array_equal(solution.variables, np.asarray(expected_variables))

    with pytest.raises(ValueError):
        miw_kp(None)


def test_mpw_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = mpw_kp(default_instance)[0]
    expected_p = 50
    assert len(solution) == len(default_instance)
    assert solution.fitness == expected_p
    assert solution.objectives == (expected_p,)

    with pytest.raises(ValueError):
        mpw_kp(None)
