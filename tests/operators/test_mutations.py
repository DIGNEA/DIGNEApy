#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_mutations.py
@Time    :   2024/06/18 11:41:51
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import copy

import numpy as np
import pytest

from digneapy.core import Instance, Solution
from digneapy.operators import mutation


@pytest.fixture
def default_instance():
    return Instance()


@pytest.fixture
def initialised_instances():
    N = 100
    chr_1 = np.random.randint(low=0, high=100, size=N)
    chr_2 = np.random.randint(low=0, high=100, size=N)
    instance_1 = Instance(chr_1)
    instance_2 = Instance(chr_2)
    return (instance_1, instance_2)


@pytest.fixture
def default_solution():
    return Solution()


@pytest.fixture
def initialised_solutions():
    N = 100
    chr_1 = np.random.randint(low=0, high=100, size=N)
    chr_2 = np.random.randint(low=0, high=100, size=N)
    solution_1 = Solution(chromosome=chr_1)
    solution_2 = Solution(chromosome=chr_2)
    return (solution_1, solution_2)


def test_uniform_one_mutation_instances(initialised_instances):
    bounds = [(0, 100) for _ in range(100)]
    instance, _ = initialised_instances
    original = copy.deepcopy(instance)

    new_instance = mutation.uniform_one_mutation(instance, bounds)
    assert new_instance != original
    assert sum(1 for i, j in zip(original, new_instance) if i != j) == 1


def test_uniform_one_mutation_solutions(initialised_solutions):
    bounds = [(0, 100) for _ in range(100)]
    solution, _ = initialised_solutions
    original = copy.deepcopy(solution)

    new_solution = mutation.uniform_one_mutation(solution, bounds)
    assert new_solution != original
    assert sum(1 for i, j in zip(original, new_solution) if i != j) == 1


def test_uniform_one_mutation_raises():
    bounds = [(0, 100) for _ in range(100)]
    with pytest.raises(Exception):
        mutation.uniform_one_mutation(list(), bounds)


def test_uniform_one_mutation_raises_bounds():
    bounds = [(0, 1, 2) for _ in range(100)]
    with pytest.raises(Exception):
        mutation.uniform_one_mutation(list(range(100)), bounds)
