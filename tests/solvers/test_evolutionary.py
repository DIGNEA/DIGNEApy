#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_heuristics.py
@Time    :   2024/04/15 09:13:27
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest
from deap import benchmarks

from digneapy.core import Solution
from digneapy.domains import Knapsack
from digneapy.solvers.evolutionary import EA, Direction


@pytest.fixture
def default_large_knap():
    rng = np.random.default_rng()
    c = rng.integers(1e3, 1e5, dtype=np.uint64)
    w = rng.integers(1000, 5000, size=1000, dtype=np.int32)
    p = rng.integers(1000, 5000, size=1000, dtype=np.int32)
    kp = Knapsack(profits=p, weights=w, capacity=c)
    return kp


def test_evolutionary_algorith_with_incremental_knapsack():
    number_of_items = 100
    profits = np.arange(1, number_of_items + 1, dtype=np.uint32)
    weights = np.arange(1, number_of_items + 1, dtype=np.uint32)
    capacity = 50
    knapsack = Knapsack(capacity=capacity, profits=profits, weights=weights)

    generations = np.uint32(100)
    pop_size = np.uint32(10)
    ea = EA(
        direction=Direction.MAXIMISE,
        dim=number_of_items,
        min_g=0,
        max_g=1,
        generations=generations,
        pop_size=pop_size,
    )
    population = ea(knapsack)
    assert len(population) == 11
    assert len(ea._logbook) == generations + 1
    assert len(ea._best_found) == number_of_items
    assert len(ea._population) == pop_size

    assert all(isinstance(i, Solution) for i in population)
    assert isinstance(ea._best_found, Solution)
    assert ea._best_found.fitness <= 50  # 50 Is the optimal
    assert ea.__name__ == "EA_PS_10_CXPB_0.6_MUTPB_0.3"
    # There are multiple options to reach the maximum fitness
    # So we dont compare the chromosomes


def test_ea_solves_sphere():
    generations = np.uint32(100)
    pop_size = np.uint32(10)

    ea = EA(
        direction=Direction.MINIMISE,
        dim=30,
        min_g=0,
        max_g=1,
        generations=generations,
        pop_size=pop_size,
    )
    population = ea(benchmarks.sphere)
    assert len(population) == pop_size + 1
    assert len(ea._logbook) == generations + 1
    assert len(ea._best_found) == 30
    assert len(ea._population) == pop_size
    assert all(isinstance(i, Solution) for i in ea._population)
    assert isinstance(ea._best_found, Solution)


def test_ea_supports_multiprocess():
    generations = np.uint32(100)
    pop_size = np.uint32(10)

    ea = EA(
        direction=Direction.MINIMISE,
        dim=30,
        min_g=0,
        max_g=1,
        generations=generations,
        pop_size=pop_size,
        n_cores=np.uint8(4),
    )
    population = ea(benchmarks.sphere)
    assert len(population) == pop_size + 1
    assert len(ea._logbook) == generations + 1
    assert len(ea._best_found) == 30
    assert len(ea._population) == pop_size
    assert all(isinstance(i, Solution) for i in ea._population)
    assert isinstance(ea._best_found, Solution)
    assert ea._n_cores == 4


def test_evolutionary_algorithm_raises_if_not_problem():
    """
    Raises an exception because we did not
    set any problem to evaluate
    """
    with pytest.raises(Exception):
        generations = np.uint32(100)
        pop_size = np.uint32(10)
        ea = EA(
            direction=Direction.MAXIMISE,
            dim=100,
            min_g=0,
            max_g=1,
            generations=generations,
            pop_size=pop_size,
        )
        ea(None)


def test_evolutionary_algorithm_raises_if_wrong_direction():
    """
    Raises an exception because the direction is not allowed
    """
    with pytest.raises(Exception):
        dimension = 10
        generations = np.uint32(100)
        pop_size = np.uint32(10)
        _ = EA(
            direction="any_other_given_direction",
            dim=dimension,
            min_g=0,
            max_g=1,
            generations=generations,
            pop_size=pop_size,
        )
