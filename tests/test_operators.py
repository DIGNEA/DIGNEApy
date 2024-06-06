#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   crossover.py
@Time    :   2023/11/03 11:04:36
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import pytest
import itertools
from operator import attrgetter
import copy
from digneapy.operators import crossover, selection, mutation, replacement
from digneapy.core import Solution, Instance
import numpy as np


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


def test_uniform_crossover_solutions(initialised_solutions):
    solution_1, solution_2 = initialised_solutions

    offspring = crossover.uniform_crossover(solution_1, solution_2)
    assert offspring != solution_1
    assert offspring != solution_2
    assert len(offspring) == 100


def test_one_point_crossover_solutions(initialised_solutions):
    solution_1, solution_2 = initialised_solutions

    offspring = crossover.one_point_crossover(solution_1, solution_2)
    assert offspring != solution_1
    assert offspring != solution_2
    assert len(offspring) == 100


def test_uniform_crossover_instances(initialised_instances):
    solution_1, solution_2 = initialised_instances

    offspring = crossover.uniform_crossover(solution_1, solution_2)
    assert offspring != solution_1
    assert offspring != solution_2
    assert len(offspring) == 100


def test_one_point_crossover_instances(initialised_instances):
    solution_1, solution_2 = initialised_instances

    offspring = crossover.one_point_crossover(solution_1, solution_2)
    assert offspring != solution_1
    assert offspring != solution_2
    assert len(offspring) == 100


def test_uniform_crossover_raises():
    N = 100
    chr_1 = np.random.randint(low=0, high=100, size=N)
    chr_2 = np.random.randint(low=0, high=100, size=N * 2)
    instance_1 = Instance(chr_1)
    instance_2 = Instance(chr_2)

    with pytest.raises(Exception):
        crossover.uniform_crossover(instance_1, instance_2)


def test_one_point_crossover_raises():
    N = 100
    chr_1 = np.random.randint(low=0, high=100, size=N)
    chr_2 = np.random.randint(low=0, high=100, size=N * 2)
    instance_1 = Instance(chr_1)
    instance_2 = Instance(chr_2)

    with pytest.raises(Exception):
        crossover.one_point_crossover(instance_1, instance_2)


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


def test_binary_selection_solutions(initialised_instances):
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
