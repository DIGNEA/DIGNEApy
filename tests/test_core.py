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

from typing import Tuple
import pytest
import copy
import numpy as np
from digneapy.core import Instance, Domain, Solution, Problem
from deap import base, creator, tools


def gen_dignea_ind(icls, size: int, min_value, max_value):
    """Auxiliar function to generate individual based on
    the Solution class of digneapy
    """
    chromosome = list(np.random.randint(low=min_value, high=max_value, size=size))
    return icls(chromosome=chromosome, fitness=creator.Fitness)


@pytest.fixture
def default_solution():
    return Solution(
        chromosome=np.random.randint(low=0, high=100, size=100),
        objectives=(
            np.random.uniform(low=0, high=10),
            np.random.uniform(low=0, high=10),
        ),
        constraints=(
            np.random.uniform(low=0, high=10),
            np.random.uniform(low=0, high=10),
        ),
        fitness=np.random.random(),
    )


def test_default_solution_attrs(default_solution):
    assert default_solution
    assert len(default_solution) == 100
    assert all(0 <= i <= 100 for i in default_solution)
    cloned = copy.deepcopy(default_solution)
    assert cloned == default_solution
    assert not cloned > default_solution
    cloned.fitness = 10000
    assert cloned > default_solution
    # ValueError
    cloned.chromosome[0] = 100.0
    assert cloned != default_solution

    chr_slice = default_solution[:15]
    assert len(chr_slice) == 15
    assert chr_slice == default_solution[:15]
    # Check access is correct
    cloned.chromosome = list(range(100))
    for i in range(len(cloned)):
        assert cloned[i] == i

    other_solution = Solution(
        chromosome=list(range(200)),
        objectives=(1.0, 1.0),
        fitness=100.0,
        constraints=(0.0, 0.0),
    )
    assert len(other_solution) == 200
    assert len(other_solution.chromosome) == 200
    # Str comparison
    assert (
        other_solution.__str__()
        == "Solution(dim=200,f=100.0,objs=(1.0, 1.0),const=(0.0, 0.0))"
    )
    # Equal comparison
    assert default_solution.__eq__(list()) == NotImplemented
    assert default_solution.__gt__(list()) == NotImplemented
    empty_s = Solution()
    assert len(empty_s) == 0


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
    assert not default_instance.portfolio_scores

    default_instance.features = list(range(3))
    default_instance.portfolio_scores = [float(i) for i in range(3)]
    default_instance.fitness = 100.0
    default_instance.s = 10.0
    default_instance.p = 5.0
    assert (
        default_instance.__repr__()
        == f"Instance<f=100.0,p=5.0,s=10.0,vars=0,features=3,performance=3>"
    )
    assert (
        format(default_instance, "p")
        == f"Instance(f=100.0,p=5.0, s=10.0, descriptor=(0.0,1.0,2.0))"
    )
    assert (
        format(default_instance)
        == f"Instance(f=100.0,p=5.0, s=10.0, descriptor=(0,1,2))"
    )


def test_default_instance_raises(default_instance):
    # Setters work when using proper data types
    default_instance.p = 100.0
    default_instance.s = 50.0
    default_instance.fitness = 500.0
    assert default_instance.p == 100.0
    assert default_instance.s == 50.0
    assert default_instance.fitness == 500.0

    with pytest.raises(AttributeError):
        default_instance.p = "hello world"

    with pytest.raises(AttributeError):
        default_instance.s = "hello world"

    with pytest.raises(AttributeError):
        default_instance.fitness = "hello world"

    with pytest.raises(AttributeError):
        s = Instance(variables=list(range(100)), fitness="hello", p=100.0, s=100.0)

    with pytest.raises(AttributeError):
        s = Instance(variables=list(range(100)), fitness=100.0, p="hello", s=100.0)

    with pytest.raises(AttributeError):
        s = Instance(variables=list(range(100)), fitness=100.0, p=100.0, s="hello")


def test_init_instance(initialised_instance):
    assert initialised_instance.p == 0.0
    assert initialised_instance.s == 0.0
    assert initialised_instance.fitness == 0.0
    assert initialised_instance._variables == list(range(100))
    assert not initialised_instance.features
    assert not initialised_instance.portfolio_scores


def test_properties(initialised_instance):
    assert not initialised_instance.portfolio_scores
    performances = list(range(4))
    initialised_instance.portfolio_scores = performances
    assert initialised_instance.portfolio_scores == performances

    assert not initialised_instance.features
    f = list(range(10))
    initialised_instance.features = f
    assert initialised_instance.features == f


def test_equal_instances(initialised_instance, default_instance):
    assert not initialised_instance == default_instance
    instance_2 = copy.copy(initialised_instance)
    assert initialised_instance == instance_2
    assert default_instance.__eq__(list()) == NotImplemented
    assert default_instance.__gt__(list()) == NotImplemented
    assert default_instance.__ge__(list()) == NotImplemented
    instance_2.fitness = default_instance.fitness + 100.0
    assert instance_2 >= default_instance


def test_boolean(initialised_instance, default_instance):
    assert not default_instance
    assert initialised_instance


def test_str():
    instance = Instance(fitness=100, p=10.0, s=3.0)
    expected = "Instance(f=100.0,p=10.0,s=3.0,features=(),performance=())"
    assert str(instance) == expected


def test_instance_iterable(initialised_instance):
    expected = list(range(100))
    assert all(a == b for a, b in zip(initialised_instance, expected))


def test_hash_instances(initialised_instance, default_instance):
    assert 0 == hash(default_instance)
    assert hash(default_instance) != hash(initialised_instance)
    instance_2 = Instance(variables=list(range(100)))
    assert hash(initialised_instance) == hash(instance_2)


@pytest.fixture
def initialised_domain():
    class FixturedDomain(Domain):
        def extract_features(self, instance: Instance) -> Tuple:
            return tuple()

        def extract_features_as_dict(self, instance: Instance):
            return dict()

        def from_instance(self, instance: Instance) -> Problem:
            return None

        def generate_instance(self) -> Instance:
            return Instance()

    bounds = list((0.0, 100.0) for _ in range(100))
    return FixturedDomain(name="Fixtured_Domain", dimension=100, bounds=bounds)


def test_init_domain_attrs(initialised_domain):
    assert initialised_domain.name == "Fixtured_Domain"
    assert initialised_domain.dimension == 100
    assert initialised_domain.bounds == [(0.0, 100.0) for _ in range(100)]
    assert len(initialised_domain) == initialised_domain.dimension


def test_lower_bounds_init(initialised_domain):
    assert all(initialised_domain.lower_i(i) == 0.0 for i in range(100))
    with pytest.raises(AttributeError):
        initialised_domain.lower_i(-1)
    with pytest.raises(AttributeError):
        initialised_domain.lower_i(10000)


def test_upper_bounds_init(initialised_domain):
    assert all(initialised_domain.lower_i(i) == 0.0 for i in range(100))
    with pytest.raises(AttributeError):
        initialised_domain.upper_i(-1)
    with pytest.raises(AttributeError):
        initialised_domain.upper_i(10000)
