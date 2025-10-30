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

from digneapy import Instance
from digneapy.domains.bpp import BPP, BPPDomain


@pytest.fixture
def default_bpp():
    rng = np.random.default_rng(seed=42)
    items = rng.integers(low=0, high=1000, size=100)
    return BPP(items, capacity=100)


def test_default_bpp_instance_attrs(default_bpp):
    assert len(default_bpp) == 100
    assert default_bpp._capacity == 100
    items = default_bpp._items
    expected_repr = f"BPP<n=100,C=100,I={items}>"
    assert default_bpp.__repr__() == expected_repr


def test_default_bpp_instance_to_be_saved_to_disk(default_bpp):
    # Check is able to create a file
    default_bpp.to_file()
    assert os.path.exists("instance.bpp")
    os.remove("instance.bpp")


def test_default_bpp_instance_to_raise_evaluate(default_bpp):
    with pytest.raises(Exception):
        # Raises attribute error when passing an empty list
        default_bpp.evaluate([])

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_bpp.evaluate(list(range(1000)))

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_bpp.evaluate(list(range(1)))


def test_default_bpp_domain_attrs():
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach="fixed")
    assert len(domain) == dimension
    assert domain.capacity_approach == "fixed"
    assert domain._max_capacity == 100
    assert np.isclose(domain.capacity_ratio, 0.8)
    assert domain._min_i == 1
    assert domain._max_i == 1000
    assert domain.bounds == [(1, domain._max_capacity)] + [
        (1, 1000) for _ in range(dimension)
    ]


def test_default_bpp_domain_raises_with_wrongs_parameters():
    with pytest.raises(ValueError):
        BPPDomain(dimension=-1)

    with pytest.raises(ValueError):
        BPPDomain(min_i=-1)

    with pytest.raises(ValueError):
        BPPDomain(max_i=-1)

    with pytest.raises(ValueError):
        BPPDomain(min_i=100, max_i=1)


def test_default_bpp_domain_wrong_capacity_approach_fixed():
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach="random", capacity_ratio=-1.0)
    assert domain.capacity_approach == "fixed"
    assert np.isclose(domain.capacity_ratio, 0.8)
    domain.capacity_approach = "random"
    assert domain.capacity_approach == "fixed"


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_bpp_domain_to_extract_features_with_capacity_approach(capacity_approach):
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach=capacity_approach, max_capacity=100)
    instances = np.asarray(domain.generate_instances(100))
    features = domain.extract_features(instances)

    assert isinstance(features, np.ndarray)
    norm_variables = np.asarray(instances, copy=True)
    norm_variables[:, 1:] = norm_variables[:, 1:] / norm_variables[:, [0]]
    expected = np.column_stack(
        [
            np.mean(norm_variables, axis=1),
            np.std(norm_variables, axis=1),
            np.median(norm_variables, axis=1),
            np.max(norm_variables, axis=1),
            np.min(norm_variables, axis=1),
            np.mean(norm_variables > 0.5, axis=1),  # Huge
            np.mean((0.5 >= norm_variables) & (norm_variables > 0.33333333333), axis=1),
            np.mean(
                (0.33333333333 >= norm_variables) & (norm_variables > 0.25), axis=1
            ),
            np.mean(0.25 >= norm_variables, axis=1),  # Small
            np.mean(0.1 >= norm_variables, axis=1),  # Tiny
        ],
    ).astype(np.float32)

    np.testing.assert_allclose(features, expected)


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_bpp_domain_to_features_dict(capacity_approach):
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach=capacity_approach, max_capacity=100)
    instances = np.asarray(domain.generate_instances(n=1))
    features = domain.extract_features_as_dict(instances)
    assert isinstance(features, list)
    assert all(isinstance(d, dict) for d in features)
    features = features[0]

    normalised_items = np.asarray(instances[0], copy=True)
    normalised_items[1:] = normalised_items[1:] / normalised_items[0]
    assert np.isclose(features["mean"], np.mean(normalised_items))
    assert np.isclose(features["std"], np.std(normalised_items))
    assert np.isclose(features["median"], np.median(normalised_items))
    assert np.isclose(features["max"], np.max(normalised_items))
    assert np.isclose(features["min"], np.min(normalised_items))


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_bpp_domain_to_generate_problem_from_instances(capacity_approach):
    dimension = 100

    domain = BPPDomain(dimension, capacity_approach=capacity_approach)
    instance = domain.generate_instances(n=1)
    problem = domain.generate_problems_from_instances(instance)[0]
    instance = instance[0]
    assert len(problem) == dimension
    assert len(problem._items) == dimension
    assert len(instance) == dimension + 1

    if capacity_approach == "fixed":
        assert problem._capacity == 100
    if capacity_approach == "evolved":
        assert problem._capacity > 0
        assert problem._capacity == instance[0]
    if capacity_approach == "percentage":
        expected_q = (np.sum(instance[1:]) * domain.capacity_ratio).astype(np.int32)
        assert instance.variables[0] == problem._capacity
        assert instance.variables[0] == expected_q
        assert problem._capacity == expected_q


def test_bin_packing_problem_to_solve_instance(default_bpp):
    solution = default_bpp.create_solution()
    expected_vars = list(range(100))
    assert all(s_i == e_i for s_i, e_i in zip(solution, expected_vars))

    fitness_s = default_bpp(solution)
    fitness_ch = default_bpp(solution.variables)
    assert fitness_s == fitness_ch

    assert isinstance(fitness_s, tuple)
    assert fitness_s[0] >= 1.0

    instance = default_bpp.to_instance()
    expected_vars = [default_bpp._capacity, *default_bpp._items]
    assert all(v_i == e_i for v_i, e_i in zip(instance.variables, expected_vars))
