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
from digneapy.operators import selection


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


def test_binary_selection_solutions(initialised_solutions):
    population = list(initialised_solutions)
    population[0].fitness = 100
    population[1].fitness = 50
    parent = selection.binary_tournament_selection(population)
    assert population[0] > population[1]
    assert len(parent) == len(population[0])
    assert id(parent) != id(population[0])
    assert id(parent) != id(population[1])
    # We dont know for sure which individual will
    # be returned so we can only check that
    # the parent is in the population
    assert parent in population


def test_binary_selection_instances(initialised_instances):
    population = list(initialised_instances)
    population[0].fitness = 100
    population[1].fitness = 50
    parent = selection.binary_tournament_selection(population)
    assert population[0] > population[1]
    assert len(parent) == len(population[0])
    assert id(parent) != id(population[0])
    assert id(parent) != id(population[1])
    # We dont know for sure which individual will
    # be returned so we can only check that
    # the parent is in the population
    assert parent in population


def test_binary_selection_solutions_raises_empty():
    with pytest.raises(Exception):
        selection.binary_tournament_selection(None)


def test_binary_selection_one_ind(initialised_solutions):
    population = [initialised_solutions[0]]
    expected = population[0]
    parent = selection.binary_tournament_selection(population)
    assert type(parent) == type(expected)
    assert parent == expected
    assert id(parent) != id(expected)


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
