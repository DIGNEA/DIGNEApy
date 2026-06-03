#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_archives.py
@Time    :   2024/06/07 12:47:37
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import copy

import numpy as np
import pytest

from digneapy import Instance, UnstructuredArchive

N_INSTANCES = 128
DIMENSION = 100
N_FEATURES = 2


@pytest.fixture
def random_population():
    rng = np.random.default_rng()
    features = rng.uniform(low=0, high=100, size=(N_INSTANCES, N_FEATURES))
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


@pytest.fixture
def default_archive():
    rng = np.random.default_rng()
    instances = [
        Instance(
            variables=rng.integers(low=0, high=100, size=DIMENSION),
            descriptor=rng.integers(low=0, high=10, size=N_FEATURES),
        )
        for _ in range(N_INSTANCES)
    ]
    return UnstructuredArchive(threshold=0.0, k=1, instances=instances)


@pytest.fixture
def empty_archive():
    return UnstructuredArchive(threshold=0.0, k=1)


@pytest.mark.parametrize("k", [3, 15, 30])
@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_default_empty_archive(k, threshold):
    archive = UnstructuredArchive(threshold=threshold, k=k)
    assert archive.k == k
    assert len(archive) == 0


def test_archive_to_array(default_archive):
    np_archive = np.asarray(default_archive)
    assert len(np_archive) == len(default_archive)
    assert isinstance(np_archive, np.ndarray)
    assert np.isclose(default_archive.threshold, 0.0)


def test_default_Archive_is_iterable(default_archive):
    instances = default_archive.instances
    assert len(instances) == len(default_archive)
    assert all(a == b for a, b in zip(instances, default_archive))
    assert all(a == b for a, b in zip(iter(instances), iter(default_archive)))


def test_not_equal_archives(default_archive, empty_archive):
    assert default_archive != empty_archive


def test_equal_archives(default_archive):
    a1 = copy.copy(default_archive)
    assert default_archive == a1


def test_extend_iterable(empty_archive, default_archive):
    assert 0 == len(empty_archive)
    instances = default_archive.instances
    scores = np.full(len(instances), fill_value=10)
    empty_archive.extend(instances, novelty_scores=scores)
    assert len(empty_archive) == len(default_archive)
    assert empty_archive == default_archive


def test_bool_on_empty_archive(empty_archive):
    assert not empty_archive


def test_bool_on_default_archive(default_archive):
    assert default_archive


def test_archive_magic(default_archive):
    assert (
        default_archive.__str__()
        == f"UnstructuredArchive(threshold=0.0,data=(|{N_INSTANCES}|))"
    )
    duplicated = copy.deepcopy(default_archive)
    assert hash(duplicated) == hash(default_archive)


def test_Archive_can_be_indexed(default_archive):
    assert len(default_archive) == N_INSTANCES
    assert isinstance(default_archive[0], Instance)
    assert len(default_archive[:2]) == 2
    with pytest.raises(IndexError):
        default_archive[N_INSTANCES + 1]


def test_archive_repr(default_archive):
    assert f"UnstructuredArchive(threshold=0.0,data=(|{N_INSTANCES}|))" == format(
        repr(default_archive),
    )


@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_archive_extend(threshold):
    empty_archive = UnstructuredArchive(k=1, threshold=threshold)
    rng = np.random.default_rng()
    scores = rng.uniform(low=threshold - 0.5, high=threshold + 1.5, size=N_INSTANCES)
    instances = [
        Instance(
            variables=np.arange(DIMENSION),
            fitness=0.0,
            p=0.0,
            descriptor=np.arange(N_FEATURES),
            s=scores[i],
        )
        for i in range(N_INSTANCES)
    ]
    expected = len(np.where(scores >= threshold)[0])
    empty_archive.extend(instances, scores)
    assert len(empty_archive) == expected


def test_unstructured_archive_raises():
    with pytest.raises(ValueError) as k_error:
        _ = UnstructuredArchive(k=-10, threshold=1.0)
    assert "UnstructuredArchive expects k to be a positive integer. Got -10" in str(
        k_error.value
    )

    with pytest.raises(ValueError) as k_error:
        _ = UnstructuredArchive(k=10, threshold=-1.0)
    assert (
        "UnstructuredArchive expects a floating point threshold >= 0. Got -1.0"
        in str(k_error.value)
    )


@pytest.mark.parametrize("k", [3, 15, 30])
@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_unstructured_archive_random_instances_and_k_and_threshold(
    k, threshold, random_population
):
    archive = UnstructuredArchive(threshold=threshold, k=k)
    descriptors = np.asarray([instance.descriptor for instance in random_population])
    assert descriptors.shape == (N_INSTANCES, N_FEATURES)

    sparseness = archive(descriptors)
    assert len(sparseness) == len(random_population)
    assert sparseness.shape == (N_INSTANCES,)
    assert all(score != 0.0 for score in sparseness)


def test_unstructured_archive_decreases_threshold():
    def generate_population():
        N_INSTANCES = 10
        N_FEATURES = 2
        DIMENSION = 100
        rng = np.random.default_rng()
        features = rng.uniform(low=0, high=100, size=(N_INSTANCES, N_FEATURES))
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
        return instances, variables, features, performances

    initial_threshold = 100.0
    decay = 0.9
    iterations = 10
    MAX_ITERATIONS = 1000
    archive = UnstructuredArchive(
        threshold=initial_threshold,
        k=np.uint32(15),
        local_competition=False,
        iterations_without_improve=iterations,
        decay=decay,
    )
    for _ in range(MAX_ITERATIONS):
        instances, _, descriptors, performances = generate_population()
        scores = archive.compute_novelty(descriptors)
        archive.extend(instances, scores, descriptors, performances[:, 0])

    assert archive.threshold < initial_threshold
    assert archive.threshold > 0.0
    assert len(archive) != 0


def test_unstructured_archive_local_competition(random_population):
    archive = UnstructuredArchive(
        threshold=100.0,
        k=np.uint32(15),
        local_competition=True,
    )
    descriptors = np.asarray([instance.descriptor for instance in random_population])
    performances = np.asarray([instance.p for instance in random_population])
    scores = archive.compute_novelty(descriptors)
    archive.extend(
        instances=random_population,
        novelty_scores=scores,
        descriptors=descriptors,
        objectives=performances,
    )

    assert len(archive) <= len(random_population)
    expected_len = len(archive)
    updated_scores = archive.compute_novelty(descriptors)
    np.testing.assert_array_equal(updated_scores, scores)
    new_performances = np.asarray(performances, copy=True)
    new_performances += 500.0
    archive.extend(
        instances=random_population,
        novelty_scores=updated_scores,
        descriptors=descriptors,
        objectives=new_performances,
    )
    # Doesnt change, only udpates best niches
    assert len(archive) == expected_len


@pytest.mark.parametrize("k", [3, 15, 30])
def test_unstructured_archive_computes_only_batch(k):
    archive = UnstructuredArchive(threshold=1.0, k=k)
    scores = archive(np.ones((k, k)))
    assert not np.array_equal(scores, np.zeros(k // 2))


@pytest.mark.parametrize("k", [3, 15, 30])
@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_unstructured_archive_doesnt_returns_zeros_if_population_smaller_than_K(
    k, threshold, random_population
):
    archive = UnstructuredArchive(threshold=threshold, k=k)
    descriptors = np.asarray([
        instance.descriptor for instance in random_population[:k]
    ])
    expected = np.zeros(len(descriptors), dtype=np.float64)
    # If len(pop) < k it should return zeros
    result = archive(descriptors)
    assert not np.array_equal(result, expected)
