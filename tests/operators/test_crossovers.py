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

import numpy as np
import pytest

from digneapy.core import Instance, Solution
from digneapy.operators import crossover


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
