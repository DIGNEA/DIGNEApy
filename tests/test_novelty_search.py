#!/usr/bin/env python

"""Tests for `digneapy` package."""

import pytest
import copy
import numpy as np
from digneapy.archives import Archive
from digneapy.qd import NS, instance_strategy
from digneapy.core import Instance


def transformer(l):
    """
    Dummy transformer that takes a List and returns a list of list with two random numbers each
    """
    return [[np.random.rand(), np.random.rand()] for _ in range(len(l))]


@pytest.fixture
def nsf():
    return NS(k=3, descriptor="features")


@pytest.fixture
def nsp():
    return NS(k=3, descriptor="performance")


@pytest.fixture
def nsi():
    return NS(k=3, descriptor="instance")


def test_default_nsf(nsf):
    assert nsf.k == 3
    assert nsf._describe_by == "features"
    assert len(nsf.archive) == 0
    assert len(nsf.solution_set) == 0
    assert nsf.__str__() == "NS(descriptor=features,k=3,A=(),S_S=())"
    assert nsf.__repr__() == "NS<descriptor=features,k=3,A=(),S_S=()>"


def test_default_nsp(nsp):
    assert nsp.k == 3
    assert nsp._describe_by == "performance"
    assert len(nsp.archive) == 0
    assert len(nsp.solution_set) == 0
    assert nsp.__str__() == "NS(descriptor=performance,k=3,A=(),S_S=())"
    assert nsp.__repr__() == "NS<descriptor=performance,k=3,A=(),S_S=()>"


def test_default_nsi(nsi):
    assert nsi.k == 3
    assert nsi._describe_by == "instance"
    assert len(nsi.archive) == 0
    assert len(nsi.solution_set) == 0
    assert nsi.__str__() == "NS(descriptor=instance,k=3,A=(),S_S=())"
    assert nsi.__repr__() == "NS<descriptor=instance,k=3,A=(),S_S=()>"


def test_default_nsf_by_features():
    ns = NS(descriptor="a_brand_new_descriptor")
    assert ns._describe_by == "features"


def __random_descriptors(n, size: int = 100):
    return [np.random.uniform(low=0, high=100, size=n) for _ in range(size)]


@pytest.fixture
def random_population():
    features = __random_descriptors(n=10)
    performances = __random_descriptors(n=(4, 4))
    instances = [
        Instance(variables=np.random.randint(low=0, high=100, size=100))
        for _ in range(100)
    ]
    for i, instance in enumerate(instances):
        instance.features = features[i]
        instance.portfolio_scores = performances[i]

    return instances


def test_run_nsf(nsf, random_population):
    assert nsf._describe_by == "features"
    assert all(len(instance.features) != 0 for instance in random_population)
    sparseness = nsf.sparseness(random_population)
    assert len(sparseness) == len(random_population)

    # Here we check that the NS includes the novel_ta amount of
    # instances that are supposed to has a s >= t_a
    novel_ta = sum(1 for i in sparseness if i >= nsf.archive.threshold)
    nsf.archive.extend(random_population)
    assert len(nsf.archive) == novel_ta

    nsf.solution_set.extend(random_population)
    assert len(nsf.solution_set) != 0

    current_len = len(nsf.archive)
    nsf.archive.extend(list())
    assert current_len == len(nsf.archive)

    current_len = len(nsf.solution_set)
    nsf.solution_set.extend(list())
    assert current_len == len(nsf.solution_set)

    # If empty population it should raise
    with pytest.raises(Exception):
        nsf.sparseness([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        nsf.sparseness(random_population[:3])

    # Here we check the sparseness calculation on the solution set
    spars_ss = nsf.sparseness_solution_set(random_population)
    assert len(spars_ss) == len(spars_ss)
    # Raises because the list is empty
    with pytest.raises(AttributeError):
        nsf.sparseness_solution_set(list())
    # Raises because the one element of the list is empty
    with pytest.raises(AttributeError):
        new_pop = random_population + [[]]
        nsf.sparseness_solution_set(new_pop)
    # Raises because we need at least to elements to calculate the sparseness
    with pytest.raises(AttributeError):
        nsf.sparseness_solution_set(random_population[:1])


def test_run_nsf_with_transformer(random_population):
    nsft = NS(k=3, descriptor="features", transformer=transformer)
    assert all(len(instance.features) != 0 for instance in random_population)
    sparseness = nsft.sparseness(random_population)
    assert len(sparseness) == len(random_population)


def test_run_nsp(nsp, random_population):
    assert nsp._describe_by == "performance"
    assert all(len(instance.portfolio_scores) != 0 for instance in random_population)
    sparseness = nsp.sparseness(random_population)
    assert len(sparseness) == len(random_population)

    # Here we check that the NS includes the novel_ta amount of
    # instances that are supposed to has a s >= t_a
    novel_ta = sum(1 for i in sparseness if i >= nsp.archive.threshold)
    nsp.archive.extend(random_population)
    assert len(nsp.archive) == novel_ta
    nsp.solution_set.extend(random_population)
    assert len(nsp.solution_set) != 0

    current_len = len(nsp.archive)
    nsp.archive.extend(list())
    assert current_len == len(nsp.archive)
    current_len = len(nsp.solution_set)
    nsp.solution_set.extend(list())
    assert current_len == len(nsp.solution_set)

    # If empty population it should raise
    with pytest.raises(Exception):
        nsp.sparseness([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        nsp.sparseness(random_population[:3])


def test_run_ns_instance(nsi, random_population):
    assert nsi._describe_by == "instance"
    assert all(len(instance.features) != 0 for instance in random_population)
    assert all(len(instance.portfolio_scores) != 0 for instance in random_population)
    assert all(len(instance) != 0 for instance in random_population)

    # Sparseness is calculated with the instance
    sparseness = nsi.sparseness(random_population)
    assert len(sparseness) == len(random_population)
    # Here we check that the NS includes the novel_ta amount of
    # instances that are supposed to has a s >= t_a
    novel_ta = sum(1 for i in sparseness if i >= nsi.archive.threshold)
    nsi.archive.extend(random_population)
    assert len(nsi.archive) == novel_ta
    nsi.solution_set.extend(random_population)
    assert len(nsi.solution_set) != 0

    current_len = len(nsi.archive)
    nsi.archive.extend(list())
    assert current_len == len(nsi.archive)

    current_len = len(nsi.solution_set)
    nsi.solution_set.extend(list())
    assert current_len == len(nsi.solution_set)

    # If empty population it should raise
    with pytest.raises(Exception):
        nsi.sparseness([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        nsi.sparseness(random_population[:3])
