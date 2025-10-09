#!/usr/bin/env python

"""Tests for `digneapy` package."""

import numpy as np
import pytest


from digneapy import (
    NS,
    Archive,
    dominated_novelty_search,
    Instance,
)

N_INSTANCES = 100
DIMENSION = 101
N_FEATURES = 10


@pytest.fixture
def random_population():
    rng = np.random.default_rng()
    features = rng.uniform(low=0, high=1_000, size=(N_INSTANCES, N_FEATURES))
    performances = rng.random(size=(N_INSTANCES, 4), dtype=np.float64)
    variables = rng.integers(low=0, high=100, size=(N_INSTANCES, DIMENSION))
    instances = [
        Instance(
            variables=variables[i],
            features=features[i],
            descriptor=features[i],
            portfolio_scores=performances[i],
            fitness=performances[i][0],
            p=performances[i][0],
        )
        for i in range(N_INSTANCES)
    ]
    return instances


@pytest.mark.parametrize("k, threshold", [(3, 5.0), (15, 5.0), (30, 5.0)])
def test_default_novelty_search_object(k, threshold):
    novelty_search = NS(Archive(threshold=threshold), k=k)
    assert novelty_search.k == k
    assert len(novelty_search.archive) == 0
    assert novelty_search.__str__() == f"NS(k={k},A=())"
    assert novelty_search.__repr__() == f"NS<k={k},A=()>"


@pytest.mark.parametrize("k", [3, 15, 30])
@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_novelty_search_with_random_instances_and_k_and_threshold(
    k, threshold, random_population
):
    novelty_search = NS(Archive(threshold=threshold), k=k)
    descriptors = np.asarray([instance.descriptor for instance in random_population])
    assert descriptors.shape == (N_INSTANCES, N_FEATURES)

    sparseness = novelty_search(descriptors)
    assert len(sparseness) == len(random_population)
    assert sparseness.shape == (N_INSTANCES,)
    assert all(score != 0.0 for score in sparseness)


@pytest.mark.parametrize("k", [3, 15, 30])
@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_novelty_search_raises_if_empty_population(k, threshold):
    novelty_search = NS(Archive(threshold=threshold), k=k)
    # If empty population it should raise
    with pytest.raises(Exception):
        novelty_search(np.empty(0))


@pytest.mark.parametrize("k", [3, 15, 30])
@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_novelty_search_returns_zeros_if_population_smaller_than_K(
    k, threshold, random_population
):
    novelty_search = NS(Archive(threshold=threshold), k=k)
    descriptors = np.asarray(
        [instance.descriptor for instance in random_population[:k]]
    )
    expected = np.zeros(len(descriptors), dtype=np.float64)
    print(descriptors.shape)
    # If len(pop) < k it should return zeros
    result = novelty_search(descriptors)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skip(reason="Impossible to pass this tests now")
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
