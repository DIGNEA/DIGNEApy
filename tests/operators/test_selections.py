#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_selections.py
@Time    :   2024/06/18 11:41:28
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest

from digneapy.core import Instance, Solution
from digneapy.operators import BinarySelection


def create_population(ind_cls, dimension: int, pop_size: int = 2):
    rng = np.random.default_rng()
    return [
        ind_cls(
            rng.integers(low=0, high=100, size=dimension),
            fitness=rng.uniform(0, 100, size=1)[0],
        )
        for _ in range(pop_size)
    ]


@pytest.mark.parametrize("dimension", argvalues=(50, 100, 500, 1_000))
@pytest.mark.parametrize(argnames="ind_type_cls", argvalues=(Instance, Solution))
def test_binary_selection(dimension, ind_type_cls):
    population = create_population(ind_type_cls, dimension)
    population[0].fitness = 100
    population[1].fitness = 50
    parent = BinarySelection()(population)
    assert population[0] > population[1]
    assert len(parent) == len(population[0])
    assert id(parent) == id(population[0]) or id(parent) == id(population[1])
    # We dont know for sure which individual will
    # be returned so we can only check that
    # the parent is in the population
    assert parent in population


def test_binary_selection_solutions_raises_empty():
    with pytest.raises(ValueError):
        _ = BinarySelection()(None)

    with pytest.raises(ValueError):
        _ = BinarySelection()(list())


@pytest.mark.parametrize("dimension", argvalues=(50, 100, 500, 1_000))
@pytest.mark.parametrize(argnames="ind_type_cls", argvalues=(Instance, Solution))
def test_binary_selection_one_ind(dimension, ind_type_cls):
    population = create_population(ind_type_cls, dimension, pop_size=1)
    expected = population[0]
    parent = BinarySelection()(population)
    assert isinstance(parent, expected.__class__)
    assert parent == expected
    assert id(parent) == id(expected)
