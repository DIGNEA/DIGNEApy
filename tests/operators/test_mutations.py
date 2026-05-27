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
from digneapy.operators import BatchUMut, ILMut, UniformMutation


@pytest.mark.parametrize("dimension", (50, 100, 500, 1_000))
@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize(argnames="ind_type_cls", argvalues=(Instance, Solution))
def test_uniform_one_mutation(ind_type_cls, dimension, lb, ub):
    rng = np.random.default_rng(seed=42)
    instance = ind_type_cls(rng.integers(low=lb, high=ub, size=dimension))
    original = instance.clone()
    lbs = np.full(shape=dimension, fill_value=lb)
    ubs = np.full(shape=dimension, fill_value=ub)
    instance = UniformMutation(seed=13)(instance, lbs, ubs)
    try:
        assert instance != original
        assert sum(1 for i, j in zip(original, instance) if i != j) == 1
    except AssertionError:
        warnings.warn(
            f"Not all mutations should change the individual. Uniform Mutation didn't pass for {ind_type_cls.__name__} with N {dimension}",
            UserWarning,
        )


@pytest.mark.parametrize("n_solutions", (10, 50, 100))
@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1_000))
@pytest.mark.parametrize(argnames="ind_type_cls", argvalues=(Instance, Solution))
def test_batch_uniform_one_mutation(ind_type_cls, n_solutions, lb, ub, dimension):
    rng = np.random.default_rng(7342389472389423)
    solutions = np.asarray([
        ind_type_cls(variables=rng.integers(low=lb, high=ub, size=dimension))
        for _ in range(n_solutions)
    ])
    lbs = np.full(shape=dimension, fill_value=lb)
    ubs = np.full(shape=dimension, fill_value=ub)
    cloned = np.asarray(solutions, copy=True)
    assert cloned is not solutions
    cloned = BatchUMut(seed=32)(cloned, lbs, ubs)
    assert cloned is not solutions
    assert cloned.shape == solutions.shape
    for original, clone in zip(solutions, cloned):
        assert sum(1 for i, j in zip(original, clone) if i != j) <= 1


@pytest.mark.parametrize("lb", (0,))
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1_000))
@pytest.mark.parametrize(argnames="ind_type_cls", argvalues=(Instance, Solution))
def test_iso_line_mutation(ind_type_cls, lb, ub, dimension):
    rng = np.random.default_rng()
    solutions = np.asarray([
        ind_type_cls(variables=rng.integers(low=lb, high=ub, size=dimension))
        for _ in range(2)
    ])
    lbs = np.full(shape=dimension, fill_value=lb)
    ubs = np.full(shape=dimension, fill_value=ub)
    cloned = np.asarray(solutions, copy=True)
    assert cloned is not solutions
    cloned = ILMut(0.01, 0.2, seed=32)(cloned, lbs, ubs)
    assert cloned is not solutions
    assert cloned.shape == solutions.shape


def test_uniform_one_raises():
    N = 100

    solution = Solution(list(range(N)))
    with pytest.raises(ValueError):
        lbs = np.full(shape=N, fill_value=0)
        ubs = np.full(shape=N // 2, fill_value=100)
        _ = UniformMutation(seed=13)(solution, lbs, ubs)

    with pytest.raises(ValueError):
        lbs = np.full(shape=N // 2, fill_value=0)
        ubs = np.full(shape=N // 2, fill_value=100)
        _ = UniformMutation(seed=13)(solution, lbs, ubs)


def test_batch_uniform_one_raises():
    N = 100
    solutions = np.random.default_rng().integers(low=0, high=100, size=(N, N))
    with pytest.raises(ValueError):
        # Raises because bounds shapes doesn't match
        lb = np.zeros(100)
        ub = np.ones(50)
        _ = BatchUMut(seed=32)(solutions, lb=lb, ub=ub)

    with pytest.raises(ValueError):
        # Raises because bounds and dimension
        lb = np.zeros(200)
        ub = np.ones(200)
        _ = BatchUMut(seed=32)(solutions, lb=lb, ub=ub)


def test_iso_line_raises():
    N = 100
    solutions = np.random.default_rng().integers(low=0, high=100, size=(N, N))
    with pytest.raises(ValueError):
        # Raises because bounds shapes doesn't match
        lb = np.zeros(100)
        ub = np.ones(50)
        _ = ILMut(sigma_iso="hello", sigma_line=0.0, seed=32)(solutions, lb=lb, ub=ub)

    with pytest.raises(ValueError):
        # Raises because bounds and dimension
        lb = np.zeros(200)
        ub = np.ones(200)
        _ = ILMut(sigma_iso=0.0, sigma_line="hello", seed=32)(solutions, lb=lb, ub=ub)
