#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_domain.py
@Time    :   2023/10/25 08:34:10
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import pytest
import copy
from digneapy.domain import Instance


@pytest.fixture
def default_instance():
    return Instance()


@pytest.fixture
def initialised_instance():
    vars = list(range(100))
    return Instance(variables=vars)


def test_default_instance_attrs(default_instance):
    assert default_instance.p == 0.0
    assert default_instance.s == 0.0
    assert default_instance.fitness == 0.0
    assert not default_instance._variables
    assert not default_instance.features
    assert not default_instance.performance


def test_init_instance(initialised_instance):
    assert initialised_instance.p == 0.0
    assert initialised_instance.s == 0.0
    assert initialised_instance.fitness == 0.0
    assert initialised_instance._variables == list(range(100))
    assert not initialised_instance.features
    assert not initialised_instance.performance


def test_properties(initialised_instance):
    assert not initialised_instance.performance
    performances = list(range(4))
    initialised_instance.performance = performances
    assert initialised_instance.performance == performances

    assert not initialised_instance.features
    f = list(range(10))
    initialised_instance.features = f
    assert initialised_instance.features == f


def test_equal_instances(initialised_instance, default_instance):
    assert not initialised_instance == default_instance
    instance_2 = copy.copy(initialised_instance)
    assert initialised_instance == instance_2


def test_boolean(initialised_instance, default_instance):
    assert not default_instance
    assert initialised_instance


def test_str():
    instance = Instance(fitness=100, p=10.0, s=3.0)
    expected = f"Instance(f=100,p=10.0,s=3.0,features=[],performance=[])"
    assert str(instance) == expected


def test_instance_iterable(initialised_instance):
    expected = list(range(100))
    assert all(a == b for a, b in zip(initialised_instance, expected))


def test_hash_instances(initialised_instance, default_instance):
    assert 0 == hash(default_instance)
    assert hash(default_instance) != hash(initialised_instance)
    instance_2 = Instance(variables=list(range(100)))
    assert hash(initialised_instance) == hash(instance_2)
