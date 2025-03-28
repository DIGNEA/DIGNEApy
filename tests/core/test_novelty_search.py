#!/usr/bin/env python

"""Tests for `digneapy` package."""

import numpy as np
import pytest

from digneapy import NS, DominatedNS, Instance, Archive


@pytest.fixture
def ns():
    return NS(k=3, archive=Archive(threshold=0.0001))


@pytest.fixture
def dns():
    return DominatedNS(k=3)


def test_default_ns(ns):
    assert ns.k == 3
    assert len(ns.archive) == 0
    assert ns.__str__() == "NS(k=3,A=())"
    assert ns.__repr__() == "NS<k=3,A=()>"


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
        instance.descriptor = features[i]
        instance.features = features[i]
        instance.portfolio_scores = performances[i]

    return instances


def test_run_ns(ns, random_population):
    assert all(len(instance.descriptor) != 0 for instance in random_population)
    random_population, sparseness = ns(random_population)
    assert len(sparseness) == len(random_population)
    print(sparseness)
    print(random_population)
    # Here we check that the NS includes the novel_ta amount of
    # instances that are supposed to has a s >= t_a
    novel_ta = sum(1 for i in sparseness if i >= ns.archive.threshold)
    ns.archive.extend(random_population)
    assert len(ns.archive) == novel_ta
    # If empty population it should raise
    with pytest.raises(Exception):
        ns([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        ns(random_population[:3])


def test_run_dns(dns, random_population):
    assert all(len(instance.descriptor) != 0 for instance in random_population)
    random_population, sparseness = dns(random_population)
    assert len(sparseness) == len(random_population)
    assert dns.archive is None

    # If empty population it should raise
    with pytest.raises(Exception):
        dns([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        dns(random_population[:3])
