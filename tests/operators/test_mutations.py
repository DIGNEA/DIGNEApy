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

from digneapy import Instance, Solution
from digneapy.operators import uniform_one_mutation, batch_uniform_one_mutation


@pytest.fixture
def default_instance():
    return Instance()


@pytest.fixture
def initialised_instances():
    N = 100
    rng = np.random.default_rng(42)
    chr_1 = rng.integers(low=0, high=100, size=N)
    chr_2 = rng.integers(low=0, high=100, size=N)
    instance_1 = Instance(chr_1)
    instance_2 = Instance(chr_2)
    return (instance_1, instance_2)


@pytest.fixture
def default_solution():
    return Solution()


@pytest.fixture
def initialised_solutions():
    N = 100
    rng = np.random.default_rng(42)
    chr_1 = rng.integers(low=0, high=100, size=N)
    chr_2 = rng.integers(low=0, high=100, size=N)
    solution_1 = Solution(variables=chr_1)
    solution_2 = Solution(variables=chr_2)
    return (solution_1, solution_2)


def test_uniform_one_mutation_instances(initialised_instances):
    bounds = [(0, 100) for _ in range(100)]
    instance, _ = initialised_instances
    original = copy.deepcopy(instance)

    new_instance = uniform_one_mutation(instance, bounds)
    assert new_instance != original
    assert sum(1 for i, j in zip(original, new_instance) if i != j) == 1


def test_uniform_one_mutation_solutions(initialised_solutions):
    bounds = [(0, 100) for _ in range(100)]
    solution, _ = initialised_solutions
    original = copy.deepcopy(solution)

    new_solution = uniform_one_mutation(solution, bounds)
    assert new_solution != original
    assert sum(1 for i, j in zip(original, new_solution) if i != j) == 1


@pytest.mark.parametrize("n_solutions", (10, 50, 100))
@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("dimension", (50, 100, 200))
def test_batch_uniform_one_mutation_solutions(n_solutions, lb, ub, dimension):
    rng = np.random.default_rng(7342389472389423)
    solutions = np.asarray(
        [
            Solution(variables=rng.integers(low=lb, high=ub, size=dimension))
            for _ in range(n_solutions)
        ]
    )
    lbs = np.full(shape=dimension, fill_value=lb)
    ubs = np.full(shape=dimension, fill_value=ub)
    cloned = np.asarray(solutions, copy=True)
    assert cloned is not solutions
    cloned = batch_uniform_one_mutation(cloned, lbs, ubs)
    assert cloned is not solutions
    assert cloned.shape == solutions.shape
    for original, clone in zip(solutions, cloned):
        assert sum(1 for i, j in zip(original, clone) if i != j) <= 1


def test_uniform_one_raises(initialised_solutions):
    bounds = [(0, 100) for _ in range(len(initialised_solutions[0]) // 2)]
    solution, _ = initialised_solutions
    with pytest.raises(ValueError):
        _ = uniform_one_mutation(solution, bounds=bounds)
