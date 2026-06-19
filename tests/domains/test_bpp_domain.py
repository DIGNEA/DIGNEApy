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

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from digneapy.core import Instance
from digneapy.domains import BPP, BPPDomain


def test_bin_packing_problem_attrs():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)
    bpp = BPP(items=items, maximum_capacity=capacity)

    assert len(bpp) == dimension
    assert bpp.maximum_capacity == capacity
    assert_equal(bpp.items, items)
    # They're not the same object
    assert bpp.items is not items
    assert len(bpp.bounds) == dimension
    assert all(bi == (0, dimension - 1) for bi in bpp.bounds)


def test_bin_packing_problem_raises_negative_capacity():

    with pytest.raises(ValueError):
        dimension = 50
        capacity = -150
        items = np.arange(dimension)
        _ = BPP(items=items, maximum_capacity=capacity)


def test_bin_packing_problem_raises_empty_items():

    with pytest.raises(ValueError):
        dimension = 0
        capacity = 150
        items = np.arange(dimension)
        _ = BPP(items=items, maximum_capacity=capacity)


def test_bin_packing_problem_raises_negative_items():

    with pytest.raises(ValueError):
        dimension = 0
        capacity = 150
        items = np.arange(-1, dimension + 1)
        _ = BPP(items=items, maximum_capacity=capacity)


def test_bin_packing_problem_to_array():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)

    bpp = BPP(items=items, maximum_capacity=capacity)
    bpp_array = np.asarray(bpp)
    expected_array = np.asarray([capacity, *items])
    assert isinstance(bpp_array, np.ndarray)
    assert_equal(bpp_array, expected_array)


def test_bin_packing_can_create_a_solution():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)

    bpp = BPP(items=items, maximum_capacity=capacity)
    solution = bpp.create_solution()
    assert len(solution) == dimension
    assert_equal(solution, items)


def test_bin_packing_can_be_saved_to_file():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)

    bpp = BPP(items=items, maximum_capacity=capacity)
    filename = "testing_bin_packing.bpp"
    # Check is able to create a file
    bpp.to_file(filename=filename)
    filename = Path(filename)
    assert filename.exists()
    filename.unlink()


def test_bin_packing_can_be_saved_to_file_with_path():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)

    bpp = BPP(items=items, maximum_capacity=capacity)
    filename = Path("testing_bin_packing.bpp")
    # Check is able to create a file
    bpp.to_file(filename=filename)
    assert filename.exists()
    filename.unlink()


def test_bin_packing_can_be_loaded_from_file():
    expected_dimension = 50
    expected_capacity = 150
    expected_items = np.arange(expected_dimension)

    filename = Path(__file__).parent / "data" / "testing_bin_packing.bpp"
    bpp = BPP.from_file(filename)

    assert len(bpp) == expected_dimension
    assert bpp.maximum_capacity == expected_capacity
    assert_equal(np.asarray(bpp)[1:], expected_items)


def test_bin_packing_can_evaluate_correctly():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)
    bpp = BPP(items=items, maximum_capacity=capacity)

    filled_bins = np.arange(dimension)
    used_bins = dimension
    filled_per_bin = filled_bins / capacity
    expected_fitness = np.sum(filled_per_bin * filled_per_bin) / used_bins

    solution = bpp.create_solution()
    fitness = bpp.evaluate(solution)[0]
    assert_equal(fitness, expected_fitness)


def test_bin_packing_call_evaluate_correctly():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)
    bpp = BPP(items=items, maximum_capacity=capacity)

    filled_bins = np.arange(dimension)
    used_bins = dimension
    filled_per_bin = filled_bins / capacity
    expected_fitness = np.sum(filled_per_bin * filled_per_bin) / used_bins

    solution = bpp.create_solution()
    fitness = bpp.evaluate(solution)[0]
    assert_equal(fitness, expected_fitness)
    call_fitness = bpp(solution)[0]
    assert_equal(call_fitness, fitness)


def test_bin_packing_problem_raises_evaluate_wrong():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)

    bpp = BPP(items=items, maximum_capacity=capacity)
    with pytest.raises(Exception):
        # Raises attribute error when passing an empty list
        bpp.evaluate([])

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(bpp)
        solution = np.arange(dimension * 2)
        bpp.evaluate(solution)

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        solution = np.empty(1)
        bpp.evaluate(solution)


def test_bin_packing_problem_can_be_casted_to_instance():
    dimension = 50
    capacity = 150
    items = np.arange(dimension)

    bpp = BPP(items=items, maximum_capacity=capacity)
    instance = bpp.to_instance()

    assert isinstance(instance, Instance)
    assert len(instance) == dimension + 1
    assert instance[0] == capacity
    assert_equal(instance[1:], items)


####### Domain tests


@pytest.mark.parametrize("capacity_approach", ("evolved", "percentage", "fixed"))
def test_bin_packing_domain_attrs(capacity_approach):
    number_of_items = 100
    minimum_weight = 10
    maximum_weight = 120
    maximum_capacity = 150
    capacity_ratio = 0.8
    domain = BPPDomain(
        number_of_items=number_of_items,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        maximum_capacity=maximum_capacity,
        capacity_approach=capacity_approach,
        capacity_ratio=capacity_ratio,
    )
    assert len(domain) == number_of_items + 1
    assert domain.capacity_approach == capacity_approach
    assert domain.capacity_ratio == 0.8 if capacity_approach == "percentage" else 1.0
    assert domain.maximum_capacity == maximum_capacity
    assert domain.minimum_weight == minimum_weight
    assert domain.maximum_weight == maximum_weight
    expected_bounds = [
        (1, maximum_capacity),
        *[(minimum_weight, maximum_weight) for _ in range(number_of_items)],
    ]

    assert_equal(domain.bounds, expected_bounds)


def test_bin_packing_domain_raises_negative_items():
    with pytest.raises(ValueError):
        number_of_items = -100
        _ = BPPDomain(number_of_items=number_of_items)


def test_bin_packing_domain_raises_negative_minimum_weight():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = BPPDomain(number_of_items=number_of_items, minimum_weight=-number_of_items)


def test_bin_packing_domain_raises_negative_maximum_weight():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = BPPDomain(number_of_items=number_of_items, maximum_weight=-number_of_items)


def test_bin_packing_domain_raises_ranges_overlap():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = BPPDomain(
            number_of_items=number_of_items,
            minimum_weight=number_of_items,
            maximum_weight=-number_of_items,
        )


def test_bin_packing_domain_raises_negative_capacity():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = BPPDomain(
            number_of_items=number_of_items, maximum_capacity=-number_of_items
        )


def test_bin_packing_domain_raises_negative_capacity_ratio():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = BPPDomain(number_of_items=number_of_items, capacity_ratio=-1.0)


def test_bin_packing_domain_raises_oub_capacity_ratio():
    with pytest.raises(ValueError):
        number_of_items = 100
        _ = BPPDomain(number_of_items=number_of_items, capacity_ratio=10.0)


def test_bin_packing_domain_wrong_capacity_approach_fallback():
    number_of_items = 100
    domain = BPPDomain(number_of_items, capacity_approach="random")
    assert domain.capacity_approach == "fixed"


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_bin_packing_domain_can_generate_instances(capacity_approach):
    number_of_items = 100
    minimum_weight = 10
    maximum_weight = 120
    maximum_capacity = 150
    capacity_ratio = 0.8
    n_instances = 10
    domain = BPPDomain(
        number_of_items=number_of_items,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        maximum_capacity=maximum_capacity,
        capacity_approach=capacity_approach,
        capacity_ratio=capacity_ratio,
    )
    instances = domain.generate_instances(n=n_instances)
    assert all(isinstance(x, Instance) for x in instances)
    assert all(len(x) == number_of_items + 1 for x in instances)

    if capacity_approach == "fixed":
        assert all(x[0] == maximum_capacity for x in instances)
    if capacity_approach == "evolved":
        assert all(1 <= x[0] <= maximum_capacity for x in instances)
    if capacity_approach == "percentage":
        assert all(
            x[0] == (np.sum(x[1:]) * capacity_ratio).astype(np.int32) for x in instances
        )


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_bpp_domain_can_extract_features(capacity_approach):
    number_of_items = 100
    minimum_weight = 10
    maximum_weight = 120
    maximum_capacity = 150
    capacity_ratio = 0.8
    n_instances = 10
    domain = BPPDomain(
        number_of_items=number_of_items,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        maximum_capacity=maximum_capacity,
        capacity_approach=capacity_approach,
        capacity_ratio=capacity_ratio,
    )
    instances = domain.generate_instances(n=n_instances)
    features = domain.extract_features(instances)

    assert isinstance(features, np.ndarray)
    norm_variables = np.asarray(instances, copy=True, dtype=np.float64)
    norm_variables[:, 1:] = norm_variables[:, 1:] / norm_variables[:, 0:1]
    expected = np.column_stack(
        [
            np.mean(norm_variables, axis=1),
            np.std(norm_variables, axis=1),
            np.median(norm_variables, axis=1),
            np.max(norm_variables, axis=1),
            np.min(norm_variables, axis=1),
            np.mean(norm_variables > 0.5, axis=1),  # Huge
            np.mean((0.5 >= norm_variables) & (norm_variables > 0.33333), axis=1),
            np.mean((0.33333 >= norm_variables) & (norm_variables > 0.25), axis=1),
            np.mean(0.25 >= norm_variables, axis=1),  # Small
            np.mean(0.1 >= norm_variables, axis=1),  # Tiny
        ],
    ).astype(np.float64)

    assert_allclose(features, expected)


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_bpp_domain_can_extract_features_as_dict(capacity_approach):
    number_of_items = 100
    minimum_weight = 10
    maximum_weight = 120
    maximum_capacity = 150
    capacity_ratio = 0.8
    n_instances = 10
    domain = BPPDomain(
        number_of_items=number_of_items,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        maximum_capacity=maximum_capacity,
        capacity_approach=capacity_approach,
        capacity_ratio=capacity_ratio,
    )
    instances = domain.generate_instances(n=n_instances)
    instances_features = domain.extract_features_as_dict(instances)
    expected_keys = set(
        "mean,std,median,max,min,tiny,small,medium,large,huge".split(",")
    )
    assert isinstance(instances_features, list)
    assert all(isinstance(d, dict) for d in instances_features)
    assert all(expected_keys == set(x.keys()) for x in instances_features)

    norm_variables = np.asarray(instances, copy=True, dtype=np.float64)
    norm_variables[:, 1:] = norm_variables[:, 1:] / norm_variables[:, 0:1]
    for i, features in enumerate(instances_features):
        assert_allclose(features["mean"], np.mean(norm_variables[i]))
        assert_allclose(features["std"], np.std(norm_variables[i]))
        assert_allclose(features["median"], np.median(norm_variables[i]))
        assert_allclose(features["max"], np.max(norm_variables[i]))
        assert_allclose(features["min"], np.min(norm_variables[i]))


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_bpp_domain_can_generate_problem_from_instances(capacity_approach):
    number_of_items = 100
    minimum_weight = 10
    maximum_weight = 120
    maximum_capacity = 150
    capacity_ratio = 0.8
    n_instances = 10
    domain = BPPDomain(
        number_of_items=number_of_items,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        maximum_capacity=maximum_capacity,
        capacity_approach=capacity_approach,
        capacity_ratio=capacity_ratio,
    )
    instances = domain.generate_instances(n=n_instances)
    instances = np.asarray(instances)
    problems = domain.generate_problems_from_instances(instances)

    assert all(len(problem) == number_of_items for problem in problems)
    assert all(len(problem.items) == number_of_items for problem in problems)

    capacities = np.asarray(
        [problem.maximum_capacity for problem in problems], dtype=np.int32
    )

    if capacity_approach == "fixed":
        assert_equal(
            capacities,
            np.full(n_instances, fill_value=maximum_capacity, dtype=np.int32),
        )

    if capacity_approach == "evolved":
        assert all(1 <= max_q <= maximum_capacity for max_q in capacities)

    if capacity_approach == "percentage":
        expected_capacities = np.sum(instances[:, 1:], axis=1) * capacity_ratio

        assert_equal(capacities, expected_capacities.astype(capacities.dtype))
