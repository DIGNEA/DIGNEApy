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

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy.domains import Knapsack, KnapsackDomain


def test_knapsack_problem_attrs():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    assert len(knapsack) == dimension
    assert knapsack.capacity == capacity

    assert_equal(knapsack.profits, profits)
    assert_equal(knapsack.weights, weights)

    assert len(knapsack.bounds) == dimension
    assert all(knapsack.get_bounds_at(i) == (0, 1) for i in range(dimension))


def test_knapsack_raises_with_wrong_args():
    with pytest.raises(ValueError):
        _ = Knapsack(profits=list(range(100)), weights=list(range(10)))

    with pytest.raises(ValueError):
        _ = Knapsack(profits=list(range(10)), weights=list(range(10)), capacity=-100)


def test_knapsack_bounds_raises_wrong_dimensions():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    with pytest.raises(IndexError):
        _ = knapsack.get_bounds_at(-1)

    with pytest.raises(IndexError):
        _ = knapsack.get_bounds_at(dimension + 1)


def test_default_kp_instance_can_be_saved_to_file():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)
    filename = "testing_knapsack.kp"
    # Check is able to create a file
    knapsack.to_file(filename=filename)
    filename = Path(filename)
    assert filename.exists()
    filename.unlink()


def test_default_kp_instance_can_be_saved_to_file_with_path():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    filename = Path("testing_knapsack.kp")
    # Check is able to create a file

    knapsack.to_file(filename=filename)
    assert filename.exists()
    filename.unlink()


def test_knapsack_can_evaluate_correctly():
    dimension = 100
    capacity = 100
    profits = np.arange(dimension)
    weights = np.ones(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    # Feasible individual evaluation
    solution = np.zeros(dimension, dtype=np.uint32)
    solution[:50] = 1
    expected_objective = np.sum(profits[:50]).astype(np.float64)

    solution_profit = knapsack.evaluate(solution)[0]
    assert_equal(solution_profit, expected_objective)


def test_default_kp_instance_can_raises_when_wrong_evaluation():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    with pytest.raises(Exception):
        # Raises attribute error when passing an empty list
        _ = knapsack.evaluate([])

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        _ = knapsack.evaluate(list(range(dimension * 2)))

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        _ = knapsack.evaluate(list(range(1)))


####### Domain tests


def test_default_kp_domain_attrs_are_as_expected():
    dimension = 100
    domain = KnapsackDomain(dimension, capacity_approach="evolved")
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
    domain = KnapsackDomain(dimension, capacity_approach="random", capacity_ratio=-1.0)
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
    domain = KnapsackDomain(dimension, capacity_approach=capacity_approach)
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
def test_kp_domain_to_descritor_dict(capacity_approach):
    dimension = 100
    n_instances = 10
    domain = KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = np.asarray(domain.generate_instances(n=n_instances))
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
    domain = KnapsackDomain(dimension, capacity_approach=capacity_approach)
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


""" def test_knapsack_problems():
    solution = default_kp.create_solution()
    assert len(solution) == len(default_kp)
    # Checks solution is in ranges
    assert all(0 <= i <= 1 for i in solution)
    fitness_s = default_kp(solution)
    fitness_ch = default_kp(solution.variables)
    assert fitness_s == fitness_ch


def test_knapsack_to_instance():
    expected_vars = [default_kp.capacity] + list(
        itertools.chain.from_iterable([*zip(default_kp.weights, default_kp.profits)])
    )
    instance = default_kp.to_instance()
    np.testing.assert_array_equal(instance.variables, expected_vars)
 """
