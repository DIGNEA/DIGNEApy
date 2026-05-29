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
from digneapy.operators import OPCX, UCX, OnePointCrossover, UniformCrossover


@pytest.mark.parametrize("cxpb", argvalues=(0.5, 0.7, 0.8))
@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("lb", argvalues=(0,))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1000))
@pytest.mark.parametrize(argnames="ind_type_cls", argvalues=(Instance, Solution))
def test_uniform_crossover(ind_type_cls, dimension, lb, ub, cxpb):
    seed_sequence = np.random.SeedSequence()
    rng_seed, cx_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(seed=rng_seed)
    solution = ind_type_cls(rng.integers(low=lb, high=ub, size=dimension))
    other = ind_type_cls(rng.integers(low=lb, high=ub, size=dimension))
    assert solution != other
    offspring = UniformCrossover(cxpb, seed=cx_seed)(solution, other)
    try:
        assert offspring != solution
        assert offspring != other
    except AssertionError:
        warnings.warn(
            f"Uniform crossover [{ind_type_cls.__name__}] didn't change anything with cxpb {cxpb} and N {dimension}",
            UserWarning,
        )
    assert len(offspring) == dimension
    assert all(lb <= x <= ub for x in offspring)


@pytest.mark.parametrize("ub", (10, 50, 100))
@pytest.mark.parametrize("lb", argvalues=(0,))
@pytest.mark.parametrize("dimension", (50, 100, 500, 1000))
@pytest.mark.parametrize(argnames="ind_type_cls", argvalues=(Instance, Solution))
def test_one_point_crossover_(ind_type_cls, dimension, lb, ub):
    seed_sequence = np.random.SeedSequence()
    rng_seed, cx_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(seed=rng_seed)
    solution = ind_type_cls(rng.integers(low=lb, high=ub, size=dimension))
    other = ind_type_cls(rng.integers(low=lb, high=ub, size=dimension))
    assert solution != other
    offspring = OnePointCrossover(seed=cx_seed)(solution, other)
    try:
        assert offspring != solution
        assert offspring != other
    except AssertionError:
        warnings.warn(
            f"OnePointCrossover [{ind_type_cls.__name__}] didn't change anything and N {dimension}",
            UserWarning,
        )
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
        UCX(cxpb=0.0, seed=None)(instance_1, instance_2)


def test_one_point_crossover_raises():
    N = 100
    rng = np.random.default_rng(42)
    chr_1 = rng.integers(low=0, high=100, size=N)
    chr_2 = rng.integers(low=0, high=100, size=N * 2)
    instance_1 = Instance(chr_1)
    instance_2 = Instance(chr_2)

    with pytest.raises(ValueError):
        OPCX()(instance_1, instance_2)
