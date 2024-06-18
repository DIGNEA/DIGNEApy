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
from digneapy.operators import replacement


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


@pytest.fixture
def population():
    instances = [
        Instance(
            variables=np.random.randint(low=0, high=100, size=100),
            fitness=np.random.randint(0, 100),
        )
        for _ in range(100)
    ]
    return instances


def test_generational(population):
    offspring = [
        Instance(
            variables=np.random.randint(low=0, high=100, size=100),
            fitness=np.random.randint(0, 100),
        )
        for _ in range(100)
    ]

    assert population != offspring
    assert all(i != j for i, j in zip(population, offspring))
    new_pop = replacement.generational(population, offspring)
    assert new_pop != population
    assert new_pop == offspring
    assert all(i == j for i, j in zip(offspring, new_pop))

    with pytest.raises(Exception):
        replacement.generational(population, [])


def test_first_improve_replacement(population):
    offspring = [
        Instance(
            variables=np.random.randint(low=0, high=100, size=100),
            fitness=np.random.randint(0, 100),
        )
        for _ in range(100)
    ]

    expected = [
        copy.copy(i) if i > j else copy.copy(j) for i, j in zip(population, offspring)
    ]

    assert population != offspring
    assert all(i != j for i, j in zip(population, offspring))
    new_pop = replacement.first_improve_replacement(population, offspring)
    assert new_pop != population
    assert new_pop != offspring
    assert new_pop == expected

    with pytest.raises(Exception):
        replacement.first_improve_replacement(population, [])


def test_elitist_replacement(population):
    offspring = [
        Instance(
            variables=np.random.randint(low=0, high=100, size=100),
            fitness=np.random.randint(0, 100),
        )
        for _ in range(100)
    ]
    new_best_f = (
        max(itertools.chain(population, offspring), key=attrgetter("fitness")).fitness
        + 10
    )
    population[0].fitness = new_best_f

    assert population != offspring
    assert all(i != j for i, j in zip(population, offspring))
    new_pop = replacement.elitist_replacement(population, offspring)
    assert new_pop != population
    assert new_pop != offspring
    assert max(new_pop, key=attrgetter("fitness")).fitness == new_best_f
    assert new_pop[0].fitness == new_best_f

    with pytest.raises(Exception):
        replacement.elitist_replacement(population, [])
