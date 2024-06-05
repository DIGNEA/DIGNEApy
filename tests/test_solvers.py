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

import pytest
import numpy as np
from digneapy.domains import knapsack
from digneapy.solvers import heuristics
from digneapy.solvers.evolutionary import EA
from digneapy.core import Solution
from deap import benchmarks
from digneapy import solvers


@pytest.fixture
def default_instance():
    p = list(range(1, 101))
    w = list(range(1, 101))
    q = 50
    return knapsack.Knapsack(p, w, q)


def test_default_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = heuristics.default_kp(default_instance)[0]
    expected_p = sum(default_instance.profits[:9])
    expected_chromosome = [1.0] * 9 + [0.0] * 91
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert solution.chromosome == expected_chromosome

    with pytest.raises(Exception):
        heuristics.default_kp(None)


def test_map_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = heuristics.map_kp(default_instance)[0]
    expected_p = 50
    expected_chromosome = [0] * 49 + [1] + [0] * 50
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert solution.chromosome == expected_chromosome

    with pytest.raises(Exception):
        heuristics.map_kp(None)


def test_miw_kp_heuristic(default_instance):
    """
    MiW should works exactly as Default
    """
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = heuristics.miw_kp(default_instance)[0]
    expected_p = sum(default_instance.profits[:9])
    expected_chromosome = [1.0] * 9 + [0.0] * 91
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert solution.chromosome == expected_chromosome

    with pytest.raises(Exception):
        heuristics.miw_kp(None)


def test_mpw_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = heuristics.mpw_kp(default_instance)[0]
    expected_p = 47
    expected_chromosome = [1.0] * 4 + [0.0] * 32 + [1.0] + [0.0] * 63
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution.fitness == expected_p
    assert solution.chromosome == expected_chromosome

    with pytest.raises(Exception):
        heuristics.mpw_kp(None)


def test_ea_with_def_kp(default_instance):
    generations = 100
    pop_size = 10

    ea = EA(
        dir=solvers.MAXIMISE,
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

    assert all(type(i) == Solution for i in population)
    assert type(ea._best_found) == Solution
    assert ea._best_found.fitness <= 50  # 50 Is the optimal
    assert ea.__name__ == "EA_PS_10_CXPB_0.6_MUTPB_0.3"
    assert ea._name == "EA_PS_10_CXPB_0.6_MUTPB_0.3"
    # There are multiple options to reach the maximum fitness
    # So we dont compare the chromosomes


def test_parallel_ea_with_def_kp(default_instance):
    generations = 100
    pop_size = 10

    ea = EA(
        dir=solvers.MAXIMISE,
        dim=len(default_instance),
        min_g=0,
        max_g=1,
        generations=generations,
        pop_size=pop_size,
        n_cores=2,
    )
    population = ea(default_instance)
    assert len(population) == 11
    assert len(ea._logbook) == generations + 1
    assert len(ea._best_found) == len(default_instance)
    assert len(ea._population) == pop_size

    assert all(type(i) == Solution for i in population)
    assert type(ea._best_found) == Solution
    assert ea._best_found.fitness <= 50
    assert ea.__name__ == "EA_PS_10_CXPB_0.6_MUTPB_0.3"
    assert ea._name == "EA_PS_10_CXPB_0.6_MUTPB_0.3"
    # There are multiple options to reach the maximum fitness
    # So we dont compare the chromosomes


def test_ea_solves_sphere():
    generations = 100
    pop_size = 10

    ea = EA(
        dir=solvers.MINIMISE,
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
    assert all(type(i) == Solution for i in ea._population)
    assert type(ea._best_found) == Solution


def test_ea_raises_problem():
    """
    Raises an exception because we did not
    set any problem to evaluate
    """
    with pytest.raises(Exception):
        generations = 100
        pop_size = 10
        ea = EA(
            dir=solvers.MAXIMISE,
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
        ea = EA(
            dir="ANY",
            dim=len(default_instance),
            min_g=0,
            max_g=1,
            generations=generations,
            pop_size=pop_size,
        )


# Do not test Parallel EA --> Takes to much time on most computers
# def test_parallel_cpp_ea(default_instance):
#     max_cores = min(2, os.cpu_count() + 1)
#     solver = ParEAKP(cores=max_cores, generations=100)
#     solutions = solver(default_instance)
#     assert len(solutions) == 1
#     assert all(type(i) == Solution for i in solutions)
#     assert solutions[0].fitness <= 50.0
#     assert len(solutions[0]) == len(default_instance)
#     assert len(solver._pop_size) == 32
#     assert len(solver._generations) == 1000
#     assert solver.__name__ == "ParEAKP_PS_32_CXPB_0.7_MUTPB_0.2"
