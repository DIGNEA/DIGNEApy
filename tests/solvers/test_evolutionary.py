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

from digneapy import Direction
from digneapy._core import Solution
from digneapy.domains import kp as knapsack
from digneapy.solvers.evolutionary import EA


@pytest.fixture
def default_instance():
    p = list(range(1, 101))
    w = list(range(1, 101))
    q = 50
    return knapsack.Knapsack(p, w, q)


@pytest.fixture
def default_large_knap():
    rng = np.random.default_rng(seed=42)
    c = rng.integers(1e3, 1e5)
    w = rng.integers(1000, 5000, size=1000, dtype=np.int32)
    p = rng.integers(1000, 5000, size=1000, dtype=np.int32)
    kp = knapsack.Knapsack(profits=p, weights=w, capacity=c)
    return kp


def test_ea_with_def_kp(default_instance):
    generations = 100
    pop_size = 10

    ea = EA(
        direction=Direction.MAXIMISE,
        dim=len(default_instance),
        min_g=0,
        max_g=1,
        generations=generations,
        pop_size=pop_size,
    )
    population = ea(default_instance)
    assert len(population) == 11
    assert len(ea._logbook) == generations + 1
    assert len(ea._best_found) == len(default_instance)
    assert len(ea._population) == pop_size

    assert all(isinstance(i, Solution) for i in population)
    assert isinstance(ea._best_found, Solution)
    assert ea._best_found.fitness <= 50  # 50 Is the optimal
    assert ea.__name__ == "EA_PS_10_CXPB_0.6_MUTPB_0.3"
    assert ea._name == "EA_PS_10_CXPB_0.6_MUTPB_0.3"
    # There are multiple options to reach the maximum fitness
    # So we dont compare the chromosomes


def test_ea_solves_sphere():
    generations = 100
    pop_size = 10

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
    generations = 100
    pop_size = 10

    ea = EA(
        direction=Direction.MINIMISE,
        dim=30,
        min_g=0,
        max_g=1,
        generations=generations,
        pop_size=pop_size,
        n_cores=4,
    )
    population = ea(benchmarks.sphere)
    assert len(population) == pop_size + 1
    assert len(ea._logbook) == generations + 1
    assert len(ea._best_found) == 30
    assert len(ea._population) == pop_size
    assert all(isinstance(i, Solution) for i in ea._population)
    assert isinstance(ea._best_found, Solution)
    assert ea._n_cores == 4


def test_ea_raises_problem():
    """
    Raises an exception because we did not
    set any problem to evaluate
    """
    with pytest.raises(Exception):
        generations = 100
        pop_size = 10
        ea = EA(
            direction=Direction.MAXIMISE,
            dim=100,
            min_g=0,
            max_g=1,
            generations=generations,
            pop_size=pop_size,
        )
        ea(None)


def test_ea_raises_direction(default_instance):
    """
    Raises an exception because the direction is not allowed
    """
    with pytest.raises(Exception):
        generations = 100
        pop_size = 10
        _ = EA(
            direction="ANY",
            dim=len(default_instance),
            min_g=0,
            max_g=1,
            generations=generations,
            pop_size=pop_size,
        )
