#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_replacements.py
@Time    :   2024/06/18 11:42:27
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import copy
import itertools
from operator import attrgetter

import numpy as np
import pytest

from digneapy.core import Instance, Solution
from digneapy.operators import (
    Elitist,
    Generational,
    GreedyReplacement,
)


def create_populations(ind_cls, pop_size: int):
    rng = np.random.default_rng()
    population = [
        ind_cls(
            rng.integers(low=0, high=100, size=100),
            fitness=rng.uniform(0, 100, size=1)[0],
        )
        for _ in range(pop_size)
    ]
    offspring = [
        ind_cls(
            rng.integers(low=0, high=100, size=100),
            fitness=rng.uniform(0, 100, size=1)[0],
        )
        for _ in range(pop_size)
    ]
    return population, offspring


@pytest.mark.parametrize("pop_size", argvalues=(16, 32, 64, 128))
@pytest.mark.parametrize("ind_type_cls", argvalues=(Instance, Solution))
def test_generational(pop_size, ind_type_cls):
    population, offspring = create_populations(ind_type_cls, pop_size)
    assert population != offspring
    assert all(i != j for i, j in zip(population, offspring))
    new_pop = Generational()(population, offspring)
    assert new_pop != population
    assert new_pop == offspring
    assert all(i == j for i, j in zip(offspring, new_pop))

    with pytest.raises(Exception):
        Generational()(population, [])


@pytest.mark.parametrize("pop_size", argvalues=(16, 32, 64, 128))
@pytest.mark.parametrize("ind_type_cls", argvalues=(Instance, Solution))
def test_greedy_replacement(pop_size, ind_type_cls):
    population, offspring = create_populations(ind_type_cls, pop_size)
    expected = [
        copy.copy(i) if i > j else copy.copy(j) for i, j in zip(population, offspring)
    ]

    assert population != offspring
    assert all(i != j for i, j in zip(population, offspring))
    new_pop = GreedyReplacement()(population, offspring)
    assert new_pop != population
    assert new_pop != offspring
    assert new_pop == expected

    with pytest.raises(Exception):
        GreedyReplacement()(population, [])


@pytest.mark.parametrize("pop_size", argvalues=(16, 32, 64, 128))
@pytest.mark.parametrize("ind_type_cls", argvalues=(Instance, Solution))
def test_elitist_replacement(pop_size, ind_type_cls):
    population, offspring = create_populations(ind_type_cls, pop_size)
    new_best_f = (
        max(itertools.chain(population, offspring), key=attrgetter("fitness")).fitness
        + 10
    )
    population[0].fitness = new_best_f

    assert population != offspring
    assert all(i != j for i, j in zip(population, offspring))
    new_pop = Elitist()(population, offspring)
    assert new_pop != population
    assert new_pop != offspring
    assert max(new_pop, key=attrgetter("fitness")).fitness == new_best_f
    assert new_pop[0].fitness == new_best_f

    with pytest.raises(Exception):
        Elitist()(population, [])
