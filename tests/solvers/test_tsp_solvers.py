#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_tsp.py
@Time    :   2025/03/05 14:22:19
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy._core import Solution
from digneapy.domains.tsp import TSP
from digneapy.solvers import greedy, nneighbour, two_opt


@pytest.fixture
def sample_tsp_problem():
    number_of_nodes = 50
    lower_bound = -10.0
    upper_bounds = 100.0
    coordinates = np.random.default_rng().uniform(
        low=lower_bound,
        high=upper_bounds,
        size=(number_of_nodes, 2),
    )
    return TSP(number_of_nodes=number_of_nodes, coords=coordinates)


def test_two_opt_solves_sample(sample_tsp_problem):
    solutions = two_opt(sample_tsp_problem)
    assert len(solutions) == 1
    solution: Solution = solutions[0]
    assert len(solution) == len(sample_tsp_problem) + 1
    assert_equal(solution.constraints, (0, 0))
    assert solution.fitness >= 0.0


def test_two_opt_raises_sample():
    with pytest.raises(RuntimeError):
        two_opt(problem=None)


def test_two_opt_is_deterministic(sample_tsp_problem):
    n_repetition = 10
    solutions = [two_opt(sample_tsp_problem)[0] for _ in range(n_repetition)]
    assert len(solutions) == n_repetition
    expected_solution = solutions[0]
    assert all(x == expected_solution for x in solutions[1:])


def test_nneighbour_solves_sample(sample_tsp_problem):
    solutions = nneighbour(sample_tsp_problem)
    assert len(solutions) == 1
    solution: Solution = solutions[0]
    assert len(solution) == len(sample_tsp_problem) + 1
    assert_equal(solution.constraints, (0, 0))
    assert solution.fitness >= 0.0


def test_nneighbour_raises_sample():
    with pytest.raises(RuntimeError):
        nneighbour(None)


def test_nneighbour_is_deterministic(sample_tsp_problem):
    n_repetition = 10
    solutions = [nneighbour(sample_tsp_problem)[0] for _ in range(n_repetition)]

    assert len(solutions) == n_repetition
    expected_solution = solutions[0]
    assert all(x == expected_solution for x in solutions[1:])


def test_greedy_solves_sample(sample_tsp_problem):
    solutions = greedy(sample_tsp_problem)
    assert len(solutions) == 1
    solution: Solution = solutions[0]
    assert len(solution) == len(sample_tsp_problem) + 1
    assert_equal(solution.constraints, (0, 0))
    assert solution.fitness >= 0.0


def test_greedy_raises_sample():
    with pytest.raises(RuntimeError):
        greedy(None)


def test_greedy_is_deterministic(sample_tsp_problem):
    n_repetition = 10
    solutions = [greedy(sample_tsp_problem)[0] for _ in range(n_repetition)]
    assert len(solutions) == n_repetition
    expected_solution = solutions[0]
    assert all(x == expected_solution for x in solutions[1:])
