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

from digneapy.domains.tsp import TSP
from digneapy.solvers import greedy, nneighbour, three_opt, two_opt


@pytest.fixture
def default_tsp_instance():
    rng = np.random.default_rng(seed=42)

    N = 20
    _coords = rng.integers(
        low=(0),
        high=(1000),
        size=(N, 2),
        dtype=int,
    )
    return TSP(nodes=N, coords=_coords)


def test_two_opt_solves_sample(default_tsp_instance):
    solutions = two_opt(default_tsp_instance)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(default_tsp_instance) + 1
    assert not np.isclose(solutions[0].fitness, 0.0)


def test_two_opt_raises_sample():
    with pytest.raises(ValueError):
        two_opt(None)


def test_two_opt_is_deterministic(default_tsp_instance):
    solutions = [two_opt(default_tsp_instance)[0].fitness for _ in range(10)]
    assert len(solutions) == 10
    assert all(x == solutions[0] for x in solutions)


def test_three_opt_solves_sample():
    rng = np.random.default_rng(seed=42)
    tsp = TSP(
        nodes=5,
        coords=rng.integers(
            low=(0),
            high=(1000),
            size=(5, 2),
            dtype=int,
        ),
    )
    solutions = three_opt(tsp)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(tsp) + 1
    assert not np.isclose(solutions[0].fitness, 0.0)


def test_three_opt_raises_sample():
    with pytest.raises(ValueError):
        three_opt(None)


@pytest.mark.skip(reason="To costly")
def test_three_opt_is_deterministic(default_tsp_instance):
    solutions = [three_opt(default_tsp_instance)[0].fitness for _ in range(2)]
    assert len(solutions) == 2
    assert all(x == solutions[0] for x in solutions)


def test_nneighbour_solves_sample(default_tsp_instance):
    solutions = nneighbour(default_tsp_instance)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(default_tsp_instance) + 1
    assert not np.isclose(solutions[0].fitness, 0.0)


def test_nneighbour_raises_sample():
    with pytest.raises(ValueError):
        nneighbour(None)


def test_nneighbour_is_deterministic(default_tsp_instance):
    solutions = [nneighbour(default_tsp_instance)[0].fitness for _ in range(10)]
    assert len(solutions) == 10
    assert all(x == solutions[0] for x in solutions)


def test_greedy_solves_sample(default_tsp_instance):
    solutions = greedy(default_tsp_instance)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(default_tsp_instance) + 1
    assert not np.isclose(solutions[0].fitness, 0.0)


def test_greedy_raises_sample():
    with pytest.raises(ValueError):
        greedy(None)


def test_greedy_is_deterministic(default_tsp_instance):
    solutions = [greedy(default_tsp_instance)[0].fitness for _ in range(10)]
    assert len(solutions) == 10
    assert all(x == solutions[0] for x in solutions)


def test_tsp_heuristics_provide_different_solutions(default_tsp_instance):
    two_opt_solutions = [two_opt(default_tsp_instance)[0].fitness for _ in range(10)]
    nneighbour_solutions = [
        nneighbour(default_tsp_instance)[0].fitness for _ in range(10)
    ]
    greedy_solutions = [greedy(default_tsp_instance)[0].fitness for _ in range(10)]
    all_solutions = [two_opt_solutions, nneighbour_solutions, greedy_solutions]
    all_fs = [
        two_opt_solutions[0],
        nneighbour_solutions[0],
    ]
    assert all(len(x) == len(two_opt_solutions) for x in all_solutions)
    assert len(all_fs) == len(set(all_fs))
