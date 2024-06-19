#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_bpp_heuristics.py
@Time    :   2024/06/18 12:09:33
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy.domains import BPP
from digneapy.solvers import best_fit, first_fit, next_fit, worst_fit


@pytest.fixture
def default_bpp_instance():
    items = [
        275,
        386,
        129,
        835,
        900,
        664,
        230,
        850,
        456,
        893,
        35,
        227,
        641,
        162,
        431,
        321,
        682,
        113,
        390,
        565,
        904,
        696,
        671,
        819,
        974,
        231,
        586,
        227,
        270,
        506,
        612,
        842,
        95,
        254,
        496,
        62,
        915,
        221,
        371,
        171,
        922,
        839,
        373,
        216,
        855,
        662,
        485,
        602,
        745,
        338,
    ]
    capacity = 400
    return BPP(items=items, capacity=capacity)


def test_best_fit_solves_sample(default_bpp_instance):
    solutions = best_fit(default_bpp_instance)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(default_bpp_instance)
    assert solutions[0].fitness != 0.0


def test_best_fit_raises_sample(default_bpp_instance):
    with pytest.raises(ValueError):
        best_fit(None)


def test_best_fit_is_deterministic(default_bpp_instance):
    solutions = [best_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    assert len(solutions) == 10
    assert all(x == solutions[0] for x in solutions)


def test_first_fit_solves_sample(default_bpp_instance):
    solutions = first_fit(default_bpp_instance)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(default_bpp_instance)
    assert solutions[0].fitness != 0.0


def test_first_fit_raises_sample(default_bpp_instance):
    with pytest.raises(ValueError):
        first_fit(None)


def test_first_fit_is_deterministic(default_bpp_instance):
    solutions = [first_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    assert len(solutions) == 10
    assert all(x == solutions[0] for x in solutions)


def test_next_fit_solves_sample(default_bpp_instance):
    solutions = next_fit(default_bpp_instance)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(default_bpp_instance)
    assert solutions[0].fitness != 0.0


def test_next_fit_raises_sample(default_bpp_instance):
    with pytest.raises(ValueError):
        next_fit(None)


def test_next_fit_is_deterministic(default_bpp_instance):
    solutions = [next_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    assert len(solutions) == 10
    assert all(x == solutions[0] for x in solutions)


def test_worst_fit_solves_sample(default_bpp_instance):
    solutions = worst_fit(default_bpp_instance)
    assert len(solutions) == 1
    assert len(solutions[0]) == len(default_bpp_instance)
    assert solutions[0].fitness != 0.0


def test_worst_fit_raises_sample(default_bpp_instance):
    with pytest.raises(ValueError):
        worst_fit(None)


def test_worst_fit_is_deterministic(default_bpp_instance):
    solutions = [worst_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    assert len(solutions) == 10
    assert all(x == solutions[0] for x in solutions)


def test_bpp_heuristic_provide_different_solutions(default_bpp_instance):
    best_solutions = [best_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    first_solutions = [first_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    next_solutions = [next_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    worst_solutions = [worst_fit(default_bpp_instance)[0].fitness for _ in range(10)]
    all_solutions = [best_solutions, first_solutions, next_solutions, worst_solutions]
    all_fs = [
        best_solutions[0],
        first_solutions[0],
        next_solutions[0],
        worst_solutions[0],
    ]
    assert all(len(x) == len(best_solutions) for x in all_solutions)
    assert len(all_fs) == len(set(all_fs))
