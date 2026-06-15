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
from numpy.testing import assert_allclose, assert_equal

from digneapy import Instance
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
        _ = Knapsack(capacity=100, profits=list(range(100)), weights=list(range(10)))

    with pytest.raises(ValueError):
        _ = Knapsack(capacity=-100, profits=list(range(10)), weights=list(range(10)))


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


def test_knapsack_can_be_cast_to_array():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    knapsack_array = np.asarray(knapsack)
    assert len(knapsack_array) == (dimension * 2) + 1
    assert_equal(knapsack_array[0], capacity)

    expected_items = np.empty(dimension * 2, dtype=np.uint64)
    expected_items[::2] = np.arange(dimension)
    expected_items[1::2] = np.arange(dimension)
    assert_equal(knapsack_array[1:], expected_items)


def test_knapsack_can_create_a_solution():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    solution = knapsack.create_solution()
    assert len(solution) == dimension
    assert all(0 <= si <= 1 for si in solution)


def test_knapsack_can_be_saved_to_file():
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


def test_knapsack_can_be_saved_to_file_with_path():
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


def test_knapsack_can_be_loaded_from_file():
    expected_dimension = 50
    expected_capacity = 10_000
    expected_profits = np.arange(expected_dimension)
    expected_weights = np.arange(expected_dimension)

    filename = Path(__file__).parent / "data" / "testing_knapsack.kp"
    knapsack = Knapsack.from_file(filename)

    assert len(knapsack) == expected_dimension
    assert knapsack.capacity == expected_capacity

    assert_equal(knapsack.profits, expected_profits)
    assert_equal(knapsack.weights, expected_weights)

    assert len(knapsack.bounds) == expected_dimension
    assert all(knapsack.get_bounds_at(i) == (0, 1) for i in range(expected_dimension))


def test_knapsack_can_evaluate_correctly():
    dimension = 100
    capacity = 100
    profits = np.arange(1, dimension + 1, dtype=np.uint32)
    weights = np.ones(dimension, dtype=np.uint32)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    # Feasible individual evaluation
    solution = np.zeros(dimension, dtype=np.uint32)
    solution[:50] = 1
    expected_objective = np.sum(profits[:50]).astype(np.float64)

    solution_profit = knapsack.evaluate(solution)[0]
    assert_equal(solution_profit, expected_objective)


def test_knapsack_call_returns_evaluate_correctly():
    dimension = 100
    capacity = 100
    profits = np.arange(1, dimension + 1, dtype=np.uint32)
    weights = np.ones(dimension, dtype=np.uint32)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    # Feasible individual evaluation
    solution = np.zeros(dimension, dtype=np.uint32)
    solution[:50] = 1
    expected_objective = np.sum(profits[:50]).astype(np.float64)

    call_profit = knapsack(solution)[0]
    evaluate_profit = knapsack.evaluate(solution)[0]
    assert_equal(call_profit, evaluate_profit)
    assert_equal(call_profit, expected_objective)


def test_knapsack_can_raises_when_wrong_evaluation():
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


def test_knapsack_can_be_casted_to_instance():
    dimension = 50
    capacity = 10_000
    profits = np.arange(dimension)
    weights = np.arange(dimension)
    knapsack = Knapsack(profits=profits, weights=weights, capacity=capacity)

    instance = knapsack.to_instance()
    assert isinstance(instance, Instance)
    assert len(instance) == (dimension * 2) + 1
    assert instance[0] == 10_000
    assert_equal(instance[1::2], weights)
    assert_equal(instance[2::2], profits)


####### Domain tests


@pytest.mark.parametrize("capacity_approach", ("evolved", "percentage", "fixed"))
def test_knapsack_domain_attrs(capacity_approach):
    number_of_items = 100
    expected_bounds = [(1.0, 1e5), *[(1, 1_000) for _ in range(number_of_items * 2)]]
    domain = KnapsackDomain(number_of_items, capacity_approach=capacity_approach)

    assert len(domain) == (number_of_items * 2) + 1
    assert domain.capacity_approach == capacity_approach
    assert domain.capacity_ratio == 0.8
    assert_equal(domain.bounds, expected_bounds)
    assert all(lbi == 1 for lbi in domain.lbs)

    upper_bounds = domain.ubs
    assert upper_bounds[0] == 1e5
    assert all(ubi == 1_000 for ubi in upper_bounds[1:])


def test_knapsack_domain_raises_negative_items():
    with pytest.raises(ValueError):
        number_of_items = -100
        _ = KnapsackDomain(number_of_items)


def test_knapsack_domain_raises_negative_capacity():
    with pytest.raises(ValueError):
        _ = KnapsackDomain(number_of_items=100, maximum_capacity=-100)


def test_knapsack_domain_raises_negative_minimum_ranges():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, minimum_weight=-100)

    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, minimum_profit=-100)


def test_knapsack_domain_raises_negative_maximum_ranges():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, maximum_weight=-100)

    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, maximum_profit=-100)


def test_knapsack_domain_raises_ranges_overlap():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, minimum_weight=100, maximum_weight=50)

    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, minimum_profit=100, maximum_profit=50)


def test_knapsack_domain_raises_wrong_capacity_ratio():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, capacity_ratio=-1.0)

    with pytest.raises(ValueError):
        number_of_items = 100
        _ = KnapsackDomain(number_of_items, capacity_ratio=10.0)


def test_knapsack_domain_fallback_wrong_capacity_approach():
    dimension = 100
    domain = KnapsackDomain(dimension, capacity_approach="random")
    assert domain.capacity_approach == "evolved"


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_kp_domain_can_extract_features(capacity_approach):
    n_instances = 10
    dimension = 100
    domain = KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = np.asarray(domain.generate_instances(n_instances))
    features = domain.extract_features(instances)

    assert isinstance(features, np.ndarray)
    assert_equal(features[:, 0], instances[:, 0])
    assert (features[:, 1] <= 1000).all()
    assert (features[:, 2] <= 1000).all()
    assert (features[:, 3] >= 1).all()
    assert (features[:, 4] >= 1).all()

    expected_eff = np.mean(
        instances[:, 2::2] / instances[:, 1::2], axis=1, dtype=np.float64
    )
    expected_mean = np.mean(instances[:, 1:], axis=1)
    expected_std = np.std(instances[:, 1:], axis=1)

    assert_allclose(features[:, 5], expected_eff)
    assert_allclose(features[:, 6], expected_mean)
    assert_allclose(features[:, 7], expected_std)


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_kp_domain_can_extract_features_from_instances(capacity_approach):
    n_instances = 10
    dimension = 100
    domain = KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = domain.generate_instances(n_instances)
    features = domain.extract_features(instances)

    # Now we can cast to ndarray to facilitate the indexing
    instances = np.asarray(instances)
    assert isinstance(features, np.ndarray)
    assert_equal(features[:, 0], instances[:, 0])
    assert (features[:, 1] <= 1000).all()
    assert (features[:, 2] <= 1000).all()
    assert (features[:, 3] >= 1).all()
    assert (features[:, 4] >= 1).all()

    expected_eff = np.mean(
        instances[:, 2::2] / instances[:, 1::2], axis=1, dtype=np.float64
    )
    expected_mean = np.mean(instances[:, 1:], axis=1)
    expected_std = np.std(instances[:, 1:], axis=1)

    assert_allclose(features[:, 5], expected_eff)
    assert_allclose(features[:, 6], expected_mean)
    assert_allclose(features[:, 7], expected_std)


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_kp_domain_can_extract_features_as_dict(capacity_approach):
    dimension = 100
    n_instances = 10

    domain = KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = np.asarray(domain.generate_instances(n=n_instances))

    features = domain.extract_features_as_dict(instances)
    # We check that the keys of each instance corresponds to the expected ones
    expected_keys = set("capacity,max_p,max_w,min_p,min_w,avg_eff,mean,std".split(","))

    assert isinstance(features, list)
    assert all(isinstance(d, dict) for d in features)
    assert all(
        expected_keys == set(instance_features.keys()) for instance_features in features
    )
    # Now, check all the instances
    expected_eff = np.mean(
        instances[:, 2::2] / instances[:, 1::2], axis=1, dtype=np.float64
    )
    expected_mean = np.mean(instances[:, 1:], axis=1, dtype=np.float64)
    expected_std = np.std(instances[:, 1:], axis=1, dtype=np.float64)
    for i, instance_features in enumerate(features):
        assert_allclose(instance_features["capacity"], instances[i, 0])
        assert instance_features["max_p"] <= 1000
        assert instance_features["max_w"] <= 1000
        assert instance_features["min_w"] >= 1
        assert instance_features["min_p"] >= 1
        assert_allclose(instance_features["avg_eff"], expected_eff[i])
        assert_allclose(instance_features["mean"], expected_mean[i])
        assert_allclose(instance_features["std"], expected_std[i])


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_kp_domain_can_generate_problems_from_instances(capacity_approach):
    dimension = 100
    n_instances = 10
    domain = KnapsackDomain(dimension, capacity_approach=capacity_approach)
    instances = domain.generate_instances(n=n_instances)

    problems = domain.generate_problems_from_instances(instances)
    assert all(isinstance(problem, Knapsack) for problem in problems)
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
