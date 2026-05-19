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

import warnings

import numpy as np
import pytest

from digneapy import Instance, Solution
from digneapy.operators import batch_uniform_one_mutation, uniform_one_mutation


@pytest.mark.parametrize("dimension", (50, 100, 500, 1_000))
@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
def test_uniform_one_mutation_instances(dimension, lb, ub):
    rng = np.random.default_rng(seed=42)
    instance = Instance(rng.integers(low=lb, high=ub, size=dimension))
    original = instance.clone()
    lbs = np.full(shape=dimension, fill_value=lb)
    ubs = np.full(shape=dimension, fill_value=ub)
    instance = uniform_one_mutation(instance, lbs, ubs)
    try:
        assert instance != original
        assert sum(1 for i, j in zip(original, instance) if i != j) == 1
    except AssertionError:
        warnings.warn(
            f"Not all mutations should change the instance. Uniform Mutation didn't pass for instances with N {dimension}",
            UserWarning,
        )


@pytest.mark.parametrize("dimension", (50, 100, 500, 1_000))
@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
def test_uniform_one_mutation_solutions(dimension, lb, ub):
    rng = np.random.default_rng(seed=13)
    solution = Solution(rng.integers(low=lb, high=ub, size=dimension))
    original = solution.clone()
    lbs = np.full(shape=dimension, fill_value=lb)
    ubs = np.full(shape=dimension, fill_value=ub)
    assert solution == original
    solution = uniform_one_mutation(solution, lbs, ubs)
    try:
        assert solution != original
        assert sum(1 for i, j in zip(original, solution) if i != j) == 1
    except AssertionError:
        warnings.warn(
            f"Not all mutations should change the solutions. Uniform Mutation didn't pass for solutions with N {dimension}",
            UserWarning,
        )


@pytest.mark.parametrize("n_solutions", (10, 50, 100))
@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1_000))
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


@pytest.mark.parametrize("n_instances", (10, 50, 100))
@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1_000))
def test_batch_uniform_one_mutation_instances(n_instances, lb, ub, dimension):
    rng = np.random.default_rng(4213)
    instances = np.asarray(
        [
            Instance(rng.integers(low=lb, high=ub, size=dimension))
            for _ in range(n_instances)
        ]
    )
    lbs = np.full(shape=dimension, fill_value=lb)
    ubs = np.full(shape=dimension, fill_value=ub)
    cloned = np.asarray(instances, copy=True)
    assert cloned is not instances
    cloned = batch_uniform_one_mutation(cloned, lbs, ubs)
    assert cloned is not instances
    assert cloned.shape == instances.shape
    for original, clone in zip(instances, cloned):
        assert sum(1 for i, j in zip(original, clone) if i != j) <= 1


def test_uniform_one_raises():
    N = 100

    solution = Solution(list(range(N)))
    with pytest.raises(ValueError):
        lbs = np.full(shape=N, fill_value=0)
        ubs = np.full(shape=N // 2, fill_value=100)
        _ = uniform_one_mutation(solution, lbs, ubs)

    with pytest.raises(ValueError):
        lbs = np.full(shape=N // 2, fill_value=0)
        ubs = np.full(shape=N // 2, fill_value=100)
        _ = uniform_one_mutation(solution, lbs, ubs)


def test_batch_uniform_one_raises():
    N = 100
    solutions = np.random.default_rng().integers(low=0, high=100, size=(N, N))
    with pytest.raises(ValueError):
        # Raises because bounds shapes doesn't match
        lb = np.zeros(100)
        ub = np.ones(50)
        _ = batch_uniform_one_mutation(solutions, lb=lb, ub=ub)

    with pytest.raises(ValueError):
        # Raises because bounds and dimension
        lb = np.zeros(200)
        ub = np.ones(200)
        _ = batch_uniform_one_mutation(solutions, lb=lb, ub=ub)
