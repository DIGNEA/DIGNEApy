#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_domains.py
@Time    :   2024/04/15 11:01:33
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""


import pytest
import copy
import numpy as np
from digneapy.domains import knapsack
from digneapy.core import Instance
import os


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
    expected_repr = f"KP<n=100,C=50>"
    assert default_kp.__repr__() == expected_repr
    # Check is able to create a file
    default_kp.to_file()
    assert os.path.exists("instance.kp") == True
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
    assert domain.bounds == [(0.0, 1e4)] + [
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


def test_kp_domain_to_instance():
    dimension = 100
    domain = knapsack.KPDomain(dimension, capacity_approach="fixed")
    instance = domain.generate_instance()
    assert len(instance) == 201  # Twice profits plus Q
    assert instance._variables[0] == 1e4

    domain.capacity_approach = "evolved"
    instance = domain.generate_instance()
    assert instance._variables[0] != 1e4
    assert instance._variables[0] in range(1, 1e4)

    domain.capacity_approach = "percentage"
    instance = domain.generate_instance()
    assert instance._variables[0] != 1e4


def test_kp_domain_to_features():
    dimension = 100
    domain = knapsack.KPDomain(dimension, capacity_approach="fixed")
    instance = domain.generate_instance()
    features = domain.extract_features(instance)

    assert type(features) == tuple
    assert features[0] == 1e4
    assert features[1] <= 1000
    assert features[2] <= 1000
    assert features[3] >= 1
    assert features[4] >= 1
    assert features[-3] != 0.0
    assert features[-2] == np.mean(instance._variables[1:])
    assert features[-1] == np.std(instance._variables[1:])

    domain.capacity_approach = "evolved"
    features = domain.extract_features(instance)
    features[0] == instance[0]

    domain.capacity_approach = "percentage"
    features = domain.extract_features(instance)
    expected_q = int(np.sum(instance._variables[1::2]) * 0.8)
    assert features[0] == expected_q


def test_kp_domain_to_features_dict():
    dimension = 100
    domain = knapsack.KPDomain(dimension, capacity_approach="fixed")
    instance = domain.generate_instance()
    features = domain.extract_features_as_dict(instance)
    assert type(features) == dict
    assert features["capacity"] == 1e4
    assert features["max_p"] <= 1000
    assert features["max_w"] <= 1000
    assert features["min_w"] >= 1
    assert features["min_p"] >= 1
    assert features["avg_eff"] != 0.0
    assert features["mean"] == np.mean(instance._variables[1:])
    assert features["std"] == np.std(instance._variables[1:])


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
