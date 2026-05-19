#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_crossovers.py
@Time    :   2024/06/18 11:43:09
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
from digneapy.operators import one_point_crossover, uniform_crossover


@pytest.mark.parametrize(
    "cxpb", np.random.default_rng().uniform(low=0, high=1, size=10)
)
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("lb", argvalues=(0,))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1000))
def test_uniform_crossover_solutions(dimension, lb, ub, cxpb):
    rng = np.random.default_rng(seed=13)
    solution = Solution(rng.integers(low=lb, high=ub, size=dimension))
    other = Solution(rng.integers(low=lb, high=ub, size=dimension))
    assert solution != other
    offspring = uniform_crossover(solution, other, cxpb=cxpb)
    try:
        assert offspring != solution
        assert offspring != other
    except AssertionError:
        warnings.warn(
            f"Uniform crossover [solutions] didn't change anythin with cxpb {cxpb} and N {dimension}",
            UserWarning,
        )
    assert len(offspring) == dimension
    assert all(lb <= x <= ub for x in offspring)


@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("lb", argvalues=(0,))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1000))
def test_one_point_crossover_solutions(dimension, lb, ub):
    rng = np.random.default_rng(seed=13)
    solution = Solution(rng.integers(low=lb, high=ub, size=dimension))
    other = Solution(rng.integers(low=lb, high=ub, size=dimension))
    assert solution != other
    offspring = one_point_crossover(solution, other)
    assert offspring != solution
    assert offspring != other
    assert len(offspring) == dimension
    assert all(lb <= x <= ub for x in offspring)


@pytest.mark.parametrize(
    "cxpb", np.random.default_rng().uniform(low=0, high=1, size=10)
)
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("lb", argvalues=(0,))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1000))
def test_uniform_crossover_instances(dimension, lb, ub, cxpb):
    rng = np.random.default_rng(seed=42)
    instance = Instance(rng.integers(low=lb, high=ub, size=dimension))
    other = Instance(rng.integers(low=lb, high=ub, size=dimension))
    assert instance != other
    offspring = uniform_crossover(instance, other, cxpb=cxpb)
    try:
        assert offspring != instance
        assert offspring != other
    except AssertionError:
        warnings.warn(
            f"Uniform crossover [instances] didn't change anythin with cxpb {cxpb} and N {dimension}",
            UserWarning,
        )
    assert len(offspring) == dimension
    assert all(lb <= x <= ub for x in offspring)


@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("lb", argvalues=(0,))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1000))
def test_one_point_crossover_instances(dimension, lb, ub):
    rng = np.random.default_rng(seed=42)
    instance = Instance(rng.integers(low=lb, high=ub, size=dimension))
    other = Instance(rng.integers(low=lb, high=ub, size=dimension))
    assert instance != other
    offspring = one_point_crossover(instance, other)
    assert offspring != instance
    assert offspring != other
    assert len(offspring) == dimension
    assert all(lb <= x <= ub for x in offspring)


def test_uniform_crossover_raises():
    N = 100
    rng = np.random.default_rng(42)
    chr_1 = rng.integers(low=0, high=100, size=N)
    chr_2 = rng.integers(low=0, high=100, size=N * 2)
    instance_1 = Instance(chr_1)
    instance_2 = Instance(chr_2)

    with pytest.raises(ValueError):
        uniform_crossover(instance_1, instance_2)


def test_one_point_crossover_raises():
    N = 100
    rng = np.random.default_rng(42)
    chr_1 = rng.integers(low=0, high=100, size=N)
    chr_2 = rng.integers(low=0, high=100, size=N * 2)
    instance_1 = Instance(chr_1)
    instance_2 = Instance(chr_2)

    with pytest.raises(ValueError):
        one_point_crossover(instance_1, instance_2)
