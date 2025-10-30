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

from digneapy.domains import kp


@pytest.fixture
def default_kp():
    p = list(range(1, 101))
    w = list(range(1, 101))
    q = 50
    return kp.Knapsack(p, w, q)


def test_default_kp_instance_can_attrs_can_be_modified(default_kp):
    assert len(default_kp) == 100
    assert len(default_kp.weights) == len(default_kp.profits)
    assert default_kp.capacity == 50
    assert default_kp.profits == list(range(1, 101))
    assert default_kp.weights == list(range(1, 101))
    expected_repr = "KP<n=100,C=50>"
    assert default_kp.__repr__() == expected_repr


def test_default_kp_instance_can_be_saved_to_file(default_kp):
    # Check is able to create a file
    default_kp.to_file()
    assert os.path.exists("instance.kp")
    os.remove("instance.kp")


def test_default_kp_instance_can_evaluate_correctly(default_kp):
    # Feasible individual evaluation
    s = np.zeros(100, dtype=int)
    s[:10] = 1

    np.random.default_rng(seed=42).shuffle(s)
    profit = default_kp.evaluate(s)
    assert not np.isclose(profit, 0.0)


def test_default_kp_instance_can_raises_when_wrong_evaluation(default_kp):
    with pytest.raises(Exception):
        # Raises attribute error when passing an empty list
        default_kp.evaluate([])

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_kp.evaluate(list(range(1000)))

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_kp.evaluate(list(range(1)))


def test_default_kp_domain_attrs_are_as_expected():
    dimension = 100
    domain = kp.KnapsackDomain(dimension, capacity_approach="evolved")
    assert len(domain) == dimension
    assert domain.capacity_approach == "evolved"
    assert np.isclose(domain.max_capacity, 1e5)
    assert np.isclose(domain.capacity_ratio, 0.8)
    assert domain.min_p == 1
    assert domain.min_w == 1
    assert domain.max_p == 1000
    assert domain.max_w == 1000
    assert domain.bounds == [(1.0, 1e5)] + [(1, 1000) for _ in range(2 * dimension)]


def test_default_kp_domain_wrong_args():
    dimension = 100
    domain = kp.KnapsackDomain(
        dimension, capacity_approach="random", capacity_ratio=-1.0
    )
    assert domain.capacity_approach == "evolved"
    assert np.isclose(domain.capacity_ratio, 0.8)

    domain.capacity_approach = "random"
    assert domain.capacity_approach == "evolved"


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_kp_domain_extract_features_for_N_instances_with_capacity_approach_(
    capacity_approach,
):
    N_INSTANCES = 100
    dimension = 100
    domain = kp.KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = np.asarray(domain.generate_instances(N_INSTANCES))
    features = domain.extract_features(instances)

    assert isinstance(features, np.ndarray)
    assert (features[:, 0] == instances[:, 0]).all()
    assert (features[:, 1] <= 1000).all()
    assert (features[:, 2] <= 1000).all()
    assert (features[:, 3] >= 1).all()
    assert (features[:, 4] >= 1).all()

    expected_eff = np.mean(instances[:, 2::2] / instances[:, 1::2])
    expected_mean = np.mean(instances[:, 1:], axis=1, dtype=np.float32)
    expected_std = np.std(instances[:, 1:], axis=1, dtype=np.float32)
    assert np.isclose(features[:, 5], expected_eff).all()
    assert np.isclose(features[:, 6], expected_mean).all()
    assert np.isclose(features[:, 7], expected_std).all()


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_kp_domain_to_features_dict(capacity_approach):
    dimension = 100
    domain = kp.KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = np.asarray(domain.generate_instances(n=100))
    features = domain.extract_features_as_dict(instances)
    expected_eff = np.mean(instances[:, 2::2] / instances[:, 1::2])

    assert isinstance(features, list)
    features = features[0]
    assert np.isclose(features["capacity"], instances[0, 0])
    assert features["max_p"] <= 1000
    assert features["max_w"] <= 1000
    assert features["min_w"] >= 1
    assert features["min_p"] >= 1
    assert np.isclose(features["avg_eff"], expected_eff)
    assert np.isclose(features["mean"], np.mean(instances[0, 1:], dtype=np.float32))
    assert np.isclose(features["std"], np.std(instances[0, 1:], dtype=np.float32))


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_kp_domain_generate_problems_from_instances(capacity_approach):
    dimension = 100
    n_instances = 100
    domain = kp.KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = domain.generate_instances(n=n_instances)

    problems = domain.generate_problems_from_instances(instances)
    assert all(len(problem.weights) == dimension for problem in problems)
    assert all(len(problem.profits) == dimension for problem in problems)

    if capacity_approach == "fixed":
        assert all(problem.capacity == 1e5 for problem in problems)
    if capacity_approach == "evolved":
        assert all(problems[i].capacity == instances[i][0] for i in range(n_instances))
    if capacity_approach == "percentage":
        expected_capacities: np.float32 = (
            np.sum(np.asarray(instances)[:, 1::2], axis=1) * domain.capacity_ratio
        ).astype(np.int32)

        capacities = np.asarray([problem.capacity for problem in problems])
        assert np.isclose(capacities, expected_capacities).all()


def test_knapsack_problems(default_kp):
    solution = default_kp.create_solution()
    assert len(solution) == len(default_kp)
    # Checks solution is in ranges
    assert all(0 <= i <= 1 for i in solution)
    fitness_s = default_kp(solution)
    fitness_ch = default_kp(solution.variables)
    assert fitness_s == fitness_ch


def test_knapsack_to_instance(default_kp):
    expected_vars = [default_kp.capacity] + list(
        itertools.chain.from_iterable([*zip(default_kp.weights, default_kp.profits)])
    )
    instance = default_kp.to_instance()
    np.testing.assert_array_equal(instance.variables, expected_vars)
