#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_bpp_domain.py
@Time    :   2024/06/18 10:29:14
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import os

import numpy as np
import pytest

from digneapy.core import Instance
from digneapy.domains import BPP, BPPDomain


@pytest.fixture
def default_bpp():
    items = np.random.randint(low=0, high=1000, size=100)
    return BPP(items, capacity=100)


def test_default_bpp_instance(default_bpp):
    assert len(default_bpp) == 100
    assert default_bpp._capacity == 100
    items = default_bpp._items
    expected_repr = f"BPP<n=100,C=100,I={items}>"
    assert default_bpp.__repr__() == expected_repr
    # Check is able to create a file
    default_bpp.to_file()
    assert os.path.exists("instance.bpp")
    os.remove("instance.bpp")

    with pytest.raises(Exception):
        # Raises attribute error when passing an empty list
        default_bpp.evaluate([])

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_bpp.evaluate(list(range(1000)))

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_bpp.evaluate(list(range(1)))


def test_default_bpp_domain():
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach="fixed")
    assert len(domain) == dimension
    assert domain.capacity_approach == "fixed"
    assert domain._max_capacity == 100
    assert domain.capacity_ratio == 0.8
    assert domain._min_i == 1
    assert domain._max_i == 1000
    assert domain.bounds == [(1, domain._max_capacity)] + [
        (1, 1000) for _ in range(dimension)
    ]

    with pytest.raises(ValueError):
        BPPDomain(dimension=-1)

    with pytest.raises(ValueError):
        BPPDomain(min_i=-1)

    with pytest.raises(ValueError):
        BPPDomain(max_i=-1)

    with pytest.raises(ValueError):
        BPPDomain(min_i=100, max_i=1)


def test_default_bpp_domain_wrong_args():
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach="random", capacity_ratio=-1.0)
    assert domain.capacity_approach == "fixed"
    assert domain.capacity_ratio == 0.8

    domain.capacity_approach = "random"
    assert domain.capacity_approach == "fixed"


def test_bpp_domain_to_features():
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach="fixed", max_capacity=100)
    instance = domain.generate_instance()
    features = domain.extract_features(instance)

    capacity = instance.variables[0]
    items = np.asarray(instance.variables[1:])
    items_norm = items / capacity

    assert isinstance(features, tuple)
    expected_f = (
        np.mean(items_norm),
        np.std(items_norm),
        np.median(items_norm),
        np.max(items_norm),
        np.min(items_norm),
    )
    assert expected_f == features[:5]

    domain.capacity_approach = "evolved"
    features = domain.extract_features(instance)
    new_capacity = instance.variables[0]
    items = np.asarray(instance.variables[1:])
    items_norm = items / new_capacity
    expected_f = (
        np.mean(items_norm),
        np.std(items_norm),
        np.median(items_norm),
        np.max(items_norm),
        np.min(items_norm),
    )
    assert expected_f == features[:5]

    domain.capacity_approach = "percentage"
    features = domain.extract_features(instance)
    new_capacity = instance.variables[0]
    items = np.asarray(instance.variables[1:])
    items_norm = items / new_capacity
    expected_f = (
        np.mean(items_norm),
        np.std(items_norm),
        np.median(items_norm),
        np.max(items_norm),
        np.min(items_norm),
    )
    assert expected_f == features[:5]
    assert instance.variables[0] == new_capacity


def test_bpp_domain_to_features_dict():
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach="fixed")
    instance = domain.generate_instance()
    features = domain.extract_features_as_dict(instance)

    capacity = instance.variables[0]
    items = np.asarray(instance.variables[1:])
    items_norm = items / capacity

    assert isinstance(features, dict)
    assert features["mean"] == np.mean(items_norm)
    assert features["std"] == np.std(items_norm)
    assert features["median"] == np.median(items_norm)
    assert features["max"] == max(items_norm)
    assert features["min"] == min(items_norm)
    assert isinstance(features["tiny"], float)
    assert isinstance(features["small"], float)
    assert isinstance(features["medium"], float)
    assert isinstance(features["large"], float)
    assert isinstance(features["huge"], float)


def test_bpp_domain_to_instance():
    dimension = 100
    variables = np.random.randint(low=1, high=1000, size=101)
    instance = Instance(variables)

    domain = BPPDomain(dimension, capacity_approach="fixed")
    bpp_instance = domain.from_instance(instance)
    assert len(bpp_instance) == dimension
    assert len(bpp_instance._items) == dimension
    assert bpp_instance._capacity == 100
    assert len(instance) == dimension + 1

    domain.capacity_approach = "evolved"
    bpp_instance = domain.from_instance(instance)
    assert len(bpp_instance._items) == dimension
    assert bpp_instance._capacity == instance[0]

    domain.capacity_approach = "percentage"
    bpp_instance = domain.from_instance(instance)
    assert len(bpp_instance._items) == dimension
    assert len(bpp_instance) == dimension
    items = instance.variables[1:]
    expected_q = np.sum(items) * domain.capacity_ratio
    assert instance.variables[0] == bpp_instance._capacity
    assert instance.variables[0] == int(expected_q)
    assert bpp_instance._capacity == int(expected_q)


def test_bpp_problem(default_bpp):
    solution = default_bpp.create_solution()
    expected_vars = list(range(100))
    assert all(s_i == e_i for s_i, e_i in zip(solution, expected_vars))

    fitness_s = default_bpp(solution)
    fitness_ch = default_bpp(solution.chromosome)
    assert fitness_s == fitness_ch

    assert isinstance(fitness_s, tuple)
    assert fitness_s[0] >= 1.0

    instance = default_bpp.to_instance()
    expected_vars = [default_bpp._capacity, *default_bpp._items]
    assert all(v_i == e_i for v_i, e_i in zip(instance.variables, expected_vars))
