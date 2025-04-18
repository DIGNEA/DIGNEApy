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

import copy

import numpy as np
import pytest

from digneapy import Instance


@pytest.fixture
def default_instance():
    return Instance()


@pytest.fixture
def initialised_instance():
    _vars = list(range(100))
    return Instance(variables=_vars)


def test_default_instance_attrs(default_instance):
    assert np.isclose(default_instance.p, 0.0)
    assert np.isclose(default_instance.s, 0.0)
    assert np.isclose(default_instance.fitness, 0.0)
    np.testing.assert_array_equal(default_instance.variables, np.zeros(0))
    assert default_instance.descriptor.size == 0
    assert default_instance.portfolio_scores.size == 0

    default_instance.descriptor = list(range(3))
    default_instance.portfolio_scores = [float(i) for i in range(3)]
    default_instance.fitness = 100.0
    default_instance.s = 10.0
    default_instance.p = 5.0
    assert (
        default_instance.__repr__()
        == "Instance<f=100.0,p=5.0,s=10.0,vars=0,features=0,descriptor=3,performance=3>"
    )
    assert (
        format(default_instance, "p")
        == "Instance(f=100.0,p=5.0, s=10.0, descriptor=(0.0,1.0,2.0))"
    )
    assert (
        format(default_instance)
        == "Instance(f=100.0,p=5.0, s=10.0, descriptor=(0,1,2))"
    )


def test_default_instance_raises(default_instance):
    # Setters work when using proper data types
    default_instance.p = 100.0
    default_instance.s = 50.0
    default_instance.fitness = 500.0
    assert np.isclose(default_instance.p, 100.0)
    assert np.isclose(default_instance.s, 50.0)
    assert np.isclose(default_instance.fitness, 500.0)

    with pytest.raises(ValueError):
        default_instance.p = "hello world"

    with pytest.raises(ValueError):
        default_instance.s = "hello world"

    with pytest.raises(ValueError):
        default_instance.fitness = "hello world"

    with pytest.raises(ValueError):
        _ = Instance(variables=list(range(100)), fitness="hello", p=100.0, s=100.0)

    with pytest.raises(ValueError):
        _ = Instance(variables=list(range(100)), fitness=100.0, p="hello", s=100.0)

    with pytest.raises(ValueError):
        _ = Instance(variables=list(range(100)), fitness=100.0, p=100.0, s="hello")


def test_init_instance(initialised_instance):
    assert np.isclose(initialised_instance.p, 0.0)
    assert np.isclose(initialised_instance.s, 0.0)
    assert np.isclose(initialised_instance.fitness, 0.0)
    expected = np.asarray(list(range(100)))
    np.testing.assert_array_equal(initialised_instance.variables, expected)
    assert initialised_instance.descriptor.size == 0
    assert initialised_instance.portfolio_scores.size == 0


def test_properties(initialised_instance):
    assert initialised_instance.portfolio_scores.size == 0
    performances = tuple(range(4))
    initialised_instance.portfolio_scores = performances
    assert np.array_equal(initialised_instance.portfolio_scores, np.array(performances))

    assert initialised_instance.descriptor.size == 0
    f = list(range(10))
    initialised_instance.descriptor = f
    assert np.array_equal(initialised_instance.descriptor, np.array(f))


def test_equal_instances(initialised_instance, default_instance):
    assert initialised_instance != default_instance
    instance_2 = copy.copy(initialised_instance)

    assert initialised_instance == instance_2

    assert default_instance.__eq__(list()) == NotImplemented
    assert default_instance.__ge__(list()) == NotImplemented
    assert default_instance.__gt__(list()) == NotImplemented

    instance_2.fitness = default_instance.fitness + 100.0
    assert instance_2 >= default_instance


def test_boolean(initialised_instance, default_instance):
    assert not default_instance
    assert initialised_instance


def test_str():
    instance = Instance(fitness=100, p=10.0, s=3.0)
    expected = "Instance(f=100.0,p=10.0,s=3.0,features=0,descriptor=array([], dtype=float64),performance=([], dtype=float64))"
    assert str(instance) == expected


def test_instance_iterable(initialised_instance):
    expected = list(range(100))
    assert all(a == b for a, b in zip(initialised_instance, expected))


def test_hash_instances(initialised_instance, default_instance):
    assert 0 == hash(default_instance)
    assert hash(default_instance) != hash(initialised_instance)
    instance_2 = Instance(variables=list(range(100)))
    assert hash(initialised_instance) == hash(instance_2)


def test_instance_as_dict(initialised_instance):
    data = initialised_instance.asdict()
    assert isinstance(data, dict)
    assert "fitness" in data
    assert "s" in data
    assert "p" in data
    assert "portfolio_scores" in data
    assert "variables" in data
    assert len(list(data["variables"].values())) == len(initialised_instance)


def test_instance_as_series(initialised_instance):
    import pandas as pd

    data = initialised_instance.to_series()
    assert isinstance(data, pd.Series)

    assert "fitness" in data
    assert "s" in data
    assert "p" in data
    for i in range(len(initialised_instance.portfolio_scores)):
        assert f"portfolio_scores_{i}" in data
    for i in range(len(initialised_instance)):
        assert f"v{i}" in data
