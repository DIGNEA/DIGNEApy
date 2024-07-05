#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_knapsack_domain.py
@Time    :   2024/04/15 11:01:33
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import itertools
import os

import numpy as np
import pytest

from digneapy.core import Instance
from digneapy.domains import knapsack


@pytest.fixture
def default_kp():
    p = list(range(1, 101))
    w = list(range(1, 101))
    q = 50
    return knapsack.Knapsack(p, w, q)


def test_default_kp_instance(default_kp):
    assert len(default_kp) == 100
    assert len(default_kp.weights) == len(default_kp.profits)
    assert default_kp.capacity == 50
    assert default_kp.profits == list(range(1, 101))
    assert default_kp.weights == list(range(1, 101))
    expected_repr = "KP<n=100,C=50>"
    assert default_kp.__repr__() == expected_repr
    # Check is able to create a file
    default_kp.to_file()
    assert os.path.exists("instance.kp")
    os.remove("instance.kp")

    with pytest.raises(Exception):
        # Raises attribute error when passing an empty list
        default_kp.evaluate([])

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_kp.evaluate(list(range(1000)))

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_kp.evaluate(list(range(1)))

    # Feasible individual evaluation
    s = np.zeros(100, dtype=int)
    s[:10] = 1
    np.random.shuffle(s)
    profit = default_kp.evaluate(s)
    assert profit != 0.0


def test_default_kp_domain():
    dimension = 100
    domain = knapsack.KPDomain(dimension, capacity_approach="evolved")
    assert len(domain) == dimension
    assert domain.capacity_approach == "evolved"
    assert domain.max_capacity == 1e4
    assert domain.capacity_ratio == 0.8
    assert domain.min_p == 1
    assert domain.min_w == 1
    assert domain.max_p == 1000
    assert domain.max_w == 1000
    assert domain.bounds == [(1.0, 1e4)] + [
        (1, 1000) if i % 2 == 0 else (1, 1000) for i in range(2 * dimension)
    ]


def test_default_kp_domain_wrong_args():
    dimension = 100
    domain = knapsack.KPDomain(
        dimension, capacity_approach="random", capacity_ratio=-1.0
    )
    assert domain.capacity_approach == "evolved"
    assert domain.capacity_ratio == 0.8

    domain.capacity_approach = "random"
    assert domain.capacity_approach == "evolved"


def test_kp_domain_to_features():
    dimension = 100
    domain = knapsack.KPDomain(dimension, capacity_approach="fixed")
    instance = domain.generate_instance()
    features = domain.extract_features(instance)

    assert isinstance(features, tuple)
    assert features[0] == 1e4
    assert features[1] <= 1000
    assert features[2] <= 1000
    assert features[3] >= 1
    assert features[4] >= 1
    assert features[-3] != 0.0
    assert features[-2] == np.mean(instance.variables[1:])
    assert features[-1] == np.std(instance.variables[1:])

    domain.capacity_approach = "evolved"
    features = domain.extract_features(instance)
    assert features[0] == instance[0]

    domain.capacity_approach = "percentage"
    features = domain.extract_features(instance)
    expected_q = int(np.sum(instance.variables[1::2]) * 0.8)
    assert features[0] == expected_q


def test_kp_domain_to_features_dict():
    dimension = 100
    domain = knapsack.KPDomain(dimension, capacity_approach="fixed")
    instance = domain.generate_instance()
    features = domain.extract_features_as_dict(instance)
    assert isinstance(features, dict)
    assert features["capacity"] == 1e4
    assert features["max_p"] <= 1000
    assert features["max_w"] <= 1000
    assert features["min_w"] >= 1
    assert features["min_p"] >= 1
    assert features["avg_eff"] != 0.0
    assert features["mean"] == np.mean(instance.variables[1:])
    assert features["std"] == np.std(instance.variables[1:])


def test_kp_domain_to_instance():
    dimension = 100
    variables = np.random.randint(low=1, high=1000, size=201)
    instance = Instance(variables)

    domain = knapsack.KPDomain(dimension, capacity_approach="fixed")
    kp_instance = domain.from_instance(instance)
    assert len(kp_instance.weights) == dimension
    assert len(kp_instance.profits) == dimension
    assert kp_instance.capacity == 1e4

    domain.capacity_approach = "evolved"
    kp_instance = domain.from_instance(instance)
    assert len(kp_instance.weights) == dimension
    assert len(kp_instance.profits) == dimension
    assert kp_instance.capacity == instance[0]

    domain.capacity_approach = "percentage"
    kp_instance = domain.from_instance(instance)
    assert len(kp_instance.weights) == dimension
    assert len(kp_instance.profits) == dimension
    expected_q = int(np.sum(kp_instance.weights) * domain.capacity_ratio)
    assert kp_instance.capacity == expected_q


def test_knapsack_problems(default_kp):
    solution = default_kp.create_solution()
    assert len(solution) == len(default_kp)
    # Checks solution is in ranges
    assert all(0 <= i <= 1 for i in solution)
    fitness_s = default_kp(solution)
    fitness_ch = default_kp(solution.chromosome)
    assert fitness_s == fitness_ch


def test_knapsack_to_instance(default_kp):
    expected_vars = [default_kp.capacity] + list(
        itertools.chain.from_iterable([*zip(default_kp.weights, default_kp.profits)])
    )
    instance = default_kp.to_instance()
    np.testing.assert_array_equal(instance.variables, expected_vars)
