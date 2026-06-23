#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_instance.py
@Time    :   2024/06/18 11:37:50
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_equal

from digneapy.core import Instance, Solution


def test_instance_attrs():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )

    assert len(instance) == dimension

    assert_equal(instance.fitness, 100.0)
    assert_equal(instance.novelty, 1.0)
    assert_equal(instance.performance_bias, 1.0)
    assert_equal(instance.descriptor, np.arange(descriptor_dim))
    assert_equal(instance.portfolio_scores, np.arange(portfolio_dim))
    assert_equal(instance.variables, np.arange(dimension))
    assert instance.dtype == np.uint32


def test_instance_default_attrs():
    dimension = 10

    instance = Instance(
        variables=list(range(dimension)),
    )

    assert len(instance) == dimension

    assert_equal(instance.fitness, 0.0)
    assert_equal(instance.novelty, 0.0)
    assert_equal(instance.performance_bias, 0.0)
    assert_equal(instance.descriptor, np.empty(0))
    assert_equal(instance.portfolio_scores, np.empty(0))
    assert_equal(instance.variables, np.arange(dimension))


@pytest.mark.parametrize("variables", argvalues=([], None))
def test_instance_raises_init_wrong_variables(variables):
    descriptor_dim = 4
    portfolio_dim = 4

    with pytest.raises(ValueError):
        _ = Instance(
            variables=variables,
            fitness=1.0,
            performance_bias=1.0,
            novelty=1.0,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        )


@pytest.mark.parametrize("fitness", argvalues=([], "abc"))
def test_instance_raises_init_wrong_fitness(fitness):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    with pytest.raises(TypeError):
        _ = Instance(
            variables=list(range(dimension)),
            fitness=fitness,
            performance_bias=1.0,
            novelty=1.0,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        )


@pytest.mark.parametrize("performance_bias", argvalues=([], "abc"))
def test_instance_raises_init_wrong_perf_bias(performance_bias):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    with pytest.raises(TypeError):
        _ = Instance(
            variables=list(range(dimension)),
            fitness=100.0,
            performance_bias=performance_bias,
            novelty=1.0,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        )


@pytest.mark.parametrize("novelty", argvalues=([], "abc"))
def test_instance_raises_init_wrong_novelty(novelty):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    with pytest.raises(TypeError):
        _ = Instance(
            variables=list(range(dimension)),
            fitness=100.0,
            performance_bias=1.0,
            novelty=novelty,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        )


def test_instance_raises_init_empty_descriptor():
    dimension = 10
    portfolio_dim = 4

    with pytest.raises(ValueError):
        _ = Instance(
            variables=list(range(dimension)),
            fitness=1.0,
            performance_bias=1.0,
            novelty=1.0,
            descriptor=[],
            portfolio_scores=tuple(range(portfolio_dim)),
        )


def test_instance_raises_init_empty_portfolio_scores():
    dimension = 10
    descriptor_dim = 4

    with pytest.raises(ValueError):
        _ = Instance(
            variables=list(range(dimension)),
            fitness=1.0,
            performance_bias=1.0,
            novelty=1.0,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=[],
        )


def test_instance_can_be_cloned():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    cloned = instance.clone()
    assert len(cloned) == len(instance)

    assert_equal(cloned.fitness, instance.fitness)
    assert_equal(cloned.novelty, instance.novelty)
    assert_equal(cloned.performance_bias, instance.performance_bias)
    assert_equal(cloned.descriptor, instance.descriptor)
    assert_equal(cloned.portfolio_scores, instance.portfolio_scores)
    assert_equal(cloned.variables, instance.variables)


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("fitness", 10),
        ("novelty", 2.0),
        ("performance_bias", 100.0),
        ("descriptor", (0.0, 1.0, 2.0)),
        ("portfolio_scores", (100.0, 100.0, 0.0, 50.0)),
    ],
)
def test_instance_can_be_cloned_with(keyword, value):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    cloned = instance.clone_with(**{keyword: value})
    # They share the same variables
    assert len(cloned) == len(instance)
    assert cloned is not instance
    assert_equal(getattr(cloned, keyword), value)


def test_instance_can_be_iter():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    assert_equal(list(instance), np.arange(dimension))


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("fitness", 10),
        ("descriptor", (2.0, 5.0)),
        ("portfolio_scores", (0.5, 0.0)),
        ("variables", np.zeros(10)),
        ("novelty", 5.0),
        ("performance_bias", 0.5),
    ],
)
def test_instance_setters_work_as_expected(keyword, value):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    previous_value = getattr(instance, keyword)
    setattr(instance, keyword, value)
    current_value = getattr(instance, keyword)
    assert not np.array_equal(previous_value, current_value)
    assert_equal(value, current_value)


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("fitness", None),
        ("fitness", "None"),
        ("variables", np.zeros(4)),
        ("variables", np.zeros(0)),
        ("variables", None),
        ("novelty", None),
        ("novelty", "None"),
        ("performance_bias", None),
        ("performance_bias", "None"),
    ],
)
def test_instance_setters_raises_if_wrong(keyword, value):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(ValueError):
        setattr(instance, keyword, value)


def test_instances_can_be_compared_true():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    cloned = instance.clone()
    assert len(cloned) == len(instance)


def test_instances_can_be_compared_false_diff_dim():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    other = Instance(
        variables=list(range(dimension // 2)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    assert other != instance


def test_instances_can_be_compared_false_same_dim():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    other = Instance(
        variables=list(range(1, dimension + 1)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    assert other != instance


def test_instances_can_be_compared_by_fitness():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    cloned = instance.clone()
    assert cloned == instance

    cloned.fitness = instance.fitness + 1.0
    assert cloned > instance
    assert cloned >= instance

    cloned.fitness = instance.fitness - 1.0
    assert cloned < instance
    assert cloned <= instance


def test_instances_cannot_be_compared_other_types():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(TypeError):
        instance == None

    with pytest.raises(TypeError):
        instance == list()

    with pytest.raises(TypeError):
        instance == Solution()

    # Greater than
    with pytest.raises(TypeError):
        instance > None

    with pytest.raises(TypeError):
        instance > list()

    with pytest.raises(TypeError):
        instance > Solution()

    # Greater equal
    with pytest.raises(TypeError):
        instance >= None

    with pytest.raises(TypeError):
        instance >= list()

    with pytest.raises(TypeError):
        instance >= Solution()


def test_instance_to_dict():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    data = instance.to_dict()
    expected_keys = (
        "target",
        "fitness",
        "novelty",
        "performance_bias",
        "portfolio_scores",
        "variables",
        *(f"d{i}" for i in range(descriptor_dim)),
    )
    # I prefer the loop rather than all
    # to check which key is failing is so
    # assert all(key in data.keys() for key in expected_keys)
    for key in expected_keys:
        assert key in data.keys()

    # Without custom solver names the default are solver_i
    assert data["target"] == "alg0"
    solvers_dict = {f"alg{i}": i for i in range(portfolio_dim)}
    assert data["portfolio_scores"] == solvers_dict

    # The same happers to the variables
    # And descriptor which was tested before
    variables_dict = {f"v{i}": i for i in range(dimension)}
    assert data["variables"] == variables_dict

    assert data["fitness"] == 100.0
    assert data["performance_bias"] == 1.0
    assert data["novelty"] == 1.0


def test_instance_to_dict_custom_names():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    descriptor_names = ("x0", "x1", "dx0", "dx1")
    solvers_names = ("default", "shortest_edge", "vlns", "sa")
    variables_names = list(f"x{i}" for i in range(dimension))

    expected_keys = (
        "target",
        "fitness",
        "novelty",
        "performance_bias",
        "portfolio_scores",
        "variables",
        *descriptor_names,
    )
    data = instance.to_dict(
        variables_names=variables_names,
        descriptor_names=descriptor_names,
        portfolio_names=solvers_names,
    )

    # I prefer the loop rather than all
    # to check which key is failing is so
    # assert all(key in data.keys() for key in expected_keys)
    for key in expected_keys:
        assert key in data.keys()

    # Without custom solver names the default are solver_i
    assert data["target"] == "default"
    solvers_dict = {solvers_names[i]: i for i in range(len(solvers_names))}
    assert data["portfolio_scores"] == solvers_dict

    # The same happers to the variables
    # And descriptor which was tested before
    variables_dict = {variables_names[i]: i for i in range(len(variables_names))}
    assert data["variables"] == variables_dict

    assert data["fitness"] == 100.0
    assert data["performance_bias"] == 1.0
    assert data["novelty"] == 1.0


def test_instance_to_df():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    df = instance.to_df()
    assert isinstance(df, pl.DataFrame)
    expected_keys = (
        "target",
        "fitness",
        "novelty",
        "performance_bias",
        # variables is flattened",
        *(f"v{i}" for i in range(dimension)),
        # Portfolio_scores is flattened "portfolio_scores",
        *(f"alg{i}" for i in range(portfolio_dim)),
        *(f"d{i}" for i in range(descriptor_dim)),
    )
    # I prefer the loop rather than all
    # to check which key is failing is so
    # assert all(key in data.keys() for key in expected_keys)
    for key in expected_keys:
        assert key in df.columns


def test_instance_to_df_with_custom_names():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    descriptor_names = ("x0", "x1", "dx0", "dx1")
    solvers_names = ("default", "shortest_edge", "vlns", "sa")
    variables_names = list(f"x{i}" for i in range(dimension))

    expected_keys = (
        "target",
        "fitness",
        "novelty",
        "performance_bias",
        # variables is flattened",
        *variables_names,
        # Portfolio_scores is flattened "portfolio_scores",
        *solvers_names,
        *descriptor_names,
    )
    df = instance.to_df(
        variables_names=variables_names,
        descriptor_names=descriptor_names,
        portfolio_names=solvers_names,
    )
    assert isinstance(df, pl.DataFrame)
    # I prefer the loop rather than all
    # to check which key is failing is so
    # assert all(key in data.keys() for key in expected_keys)
    for key in expected_keys:
        assert key in df.columns


def test_instance_can_be_accessed():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    assert instance[0] == 0
    assert instance[-1] == 9
    assert_equal(instance[2:4], [2, 3])


@pytest.mark.parametrize("index", (None, 2.5, "abc"))
def test_instance_cannot_be_accessed_with_others(index):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(TypeError):
        _ = instance[index]


def test_instance_can_be_accessed_and_raise():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(IndexError):
        _ = instance[1000]


def test_instance_can_be_updated_setitem():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    expected = 100
    instance[0] = expected
    assert instance[0] == expected

    instance[-1] = expected
    assert instance[-1] == expected

    expected_slice = [100, 200, 300]
    instance[2:5] = expected_slice
    assert_equal(instance[2:5], expected_slice)


@pytest.mark.parametrize("index", (None, 2.5, "abc"))
def test_instance_cannot_be_updated_with_others(index):
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(TypeError):
        instance[index] = 1


def test_instance_can_be_updated_and_raise():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(IndexError):
        instance[1000] = 100


def test_instance_setitem_raise_slice_mismatch():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(ValueError):
        instance[1:3] = [10, 20, 30]


def test_instance_setitem_raise_slice_and_scalar_value():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(TypeError):
        instance[1:3] = 100


def test_instance_setitem_raise_index_and_slice_value():
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4

    instance = Instance(
        variables=list(range(dimension)),
        fitness=100.0,
        performance_bias=1.0,
        novelty=1.0,
        descriptor=tuple(range(descriptor_dim)),
        portfolio_scores=tuple(range(portfolio_dim)),
    )
    with pytest.raises(ValueError):
        instance[1] = [100, 200, 300]
