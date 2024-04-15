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
import copy
import numpy as np
from digneapy.domains import knapsack
from digneapy.solvers import heuristics
from digneapy.core import Solution


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
    assert solution._fitness == expected_p
    assert solution.chromosome == expected_chromosome


def test_map_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = heuristics.map_kp(default_instance)[0]
    expected_p = 50
    expected_chromosome = [0] * 49 + [1] + [0] * 50
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution._fitness == expected_p
    assert solution.chromosome == expected_chromosome


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
    assert solution._fitness == expected_p
    assert solution.chromosome == expected_chromosome


def test_mpw_kp_heuristic(default_instance):
    assert default_instance.capacity == 50
    assert len(default_instance) == 100
    solution = heuristics.mpw_kp(default_instance)[0]
    expected_p = 47
    expected_chromosome = [1.0] * 4 + [0.0] * 32 + [1.0] + [0.0] * 63
    assert len(solution) == len(default_instance)
    assert solution.objectives[0] == expected_p
    assert solution._fitness == expected_p
    assert solution.chromosome == expected_chromosome
