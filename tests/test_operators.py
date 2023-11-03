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
import copy
from digneapy.operators import crossover, selection
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


def test_uniform_crossover_raises(initialised_instances):
    solution_1, _ = initialised_instances
    with pytest.raises(Exception):
        crossover.uniform_crossover(solution_1, None)


def test_one_point_crossover_raises(initialised_instances):
    solution_1, _ = initialised_instances
    with pytest.raises(Exception):
        crossover.one_point_crossover(solution_1, None)


def test_binary_selection_solutions(initialised_solutions):
    sol_1, sol_2 = initialised_solutions
    sol_1.fitness = 100
    sol_2.fitness = 50
    population = [sol_1, sol_2]
    parent = selection.binary_tournament_selection(population)
    assert sol_1 > sol_2
    assert len(parent) == len(sol_1)
    assert parent == sol_1
    assert parent != sol_2


def test_binary_selection_solutions_raises_empty():
    with pytest.raises(Exception):
        selection.binary_tournament_selection(None)
