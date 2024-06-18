#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_solution.py
@Time    :   2024/06/18 11:38:51
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import copy

import numpy as np
import pytest

from digneapy.core import Solution


@pytest.fixture
def default_solution():
    return Solution(
        chromosome=np.random.randint(low=0, high=100, size=100),
        objectives=(
            np.random.uniform(low=0, high=10),
            np.random.uniform(low=0, high=10),
        ),
        constraints=(
            np.random.uniform(low=0, high=10),
            np.random.uniform(low=0, high=10),
        ),
        fitness=np.random.random(),
    )


def test_default_solution_attrs(default_solution):
    assert default_solution
    assert len(default_solution) == 100
    assert all(0 <= i <= 100 for i in default_solution)
    cloned = copy.deepcopy(default_solution)
    assert cloned == default_solution
    assert not cloned > default_solution
    cloned.fitness = 10000
    assert cloned > default_solution
    # ValueError
    cloned.chromosome[0] = 100.0
    assert cloned != default_solution

    chr_slice = default_solution[:15]
    assert len(chr_slice) == 15
    assert chr_slice == default_solution[:15]
    # Check access is correct
    cloned.chromosome = list(range(100))
    for i in range(len(cloned)):
        assert cloned[i] == i

    other_solution = Solution(
        chromosome=list(range(200)),
        objectives=(1.0, 1.0),
        fitness=100.0,
        constraints=(0.0, 0.0),
    )
    assert len(other_solution) == 200
    assert len(other_solution.chromosome) == 200
    # Str comparison
    assert (
        other_solution.__str__()
        == "Solution(dim=200,f=100.0,objs=(1.0, 1.0),const=(0.0, 0.0))"
    )
    # Equal comparison
    assert default_solution.__eq__(list()) == NotImplemented
    assert default_solution.__gt__(list()) == NotImplemented
    empty_s = Solution()
    assert len(empty_s) == 0
