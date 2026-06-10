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

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from digneapy import GridArchive, Instance, UnstructuredArchive

from .conftest import default_incremental_population


def test_unstructured_archive_attrs():
    k = 10
    threshold = 0.5
    instances = None
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)

    assert archive.novelty_threshold == threshold
    assert archive.k == k
    assert len(archive) == 0

    assert_equal(archive.instances, [])
    assert_equal(archive.descriptors, [])
    assert_equal(np.asarray(archive), [])

    data = archive.to_dict()
    assert isinstance(data, dict)
    assert "instances" in data.keys()


@pytest.mark.parametrize("k", (-10.0, "abc"))
def test_unstructured_archive_raises_wrong_k(k):
    threshold = 0.5
    instances = None
    with pytest.raises(ValueError):
        _ = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)


@pytest.mark.parametrize("novelty_threshold", (-10.0, "abc"))
def test_unstructured_archive_raises_wrong_novelty_threshold(novelty_threshold):
    k = 10
    instances = None
    with pytest.raises(ValueError):
        _ = UnstructuredArchive(
            k=k, novelty_threshold=novelty_threshold, instances=instances
        )


def test_unstructured_archive_raises_wrong_initial_instances():
    k = 10
    threshold = 0.5
    instances = default_incremental_population(n_instances=2)
    instances.append(np.zeros(shape=(10)))
    with pytest.raises(TypeError):
        _ = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)


def test_unstructured_archive_with_initial_instances():
    k = 10
    threshold = 0.5
    n_instances = 10
    instances = default_incremental_population(n_instances=n_instances)
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    assert archive.novelty_threshold == threshold
    assert archive.k == k
    assert len(archive) == n_instances

    descriptors = np.asarray([x.descriptor for x in instances])
    assert_equal(archive.instances, instances)
    assert_equal(archive.descriptors, descriptors)
    assert_equal(np.asarray(archive), instances)

    data = archive.to_dict()
    assert isinstance(data, dict)
    assert "instances" in data.keys()


def test_unstructured_archive_is_iterable():
    k = 10
    threshold = 0.5
    n_instances = 10
    instances = default_incremental_population(n_instances=n_instances)
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    assert archive.novelty_threshold == threshold
    assert archive.k == k
    assert len(archive) == n_instances
    for instance in archive:
        assert isinstance(instance, Instance)


def test_unstructured_archive_can_be_cmp_true():
    k = 10
    threshold = 0.5
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=None)
    other = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=None)
    assert archive == other


def test_unstructured_archive_filled_can_be_cmp_true():
    k = 10
    threshold = 0.5
    n_instances = 10
    instances = default_incremental_population(n_instances=n_instances)
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    other = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    assert archive == other


def test_unstructured_archive_can_be_cmp_false():
    k = 10
    threshold = 0.5
    instances = default_incremental_population(n_instances=1)
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    other = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=None)
    assert archive != other


def test_unstructured_archive_filled_can_be_cmp_false():
    k = 10
    threshold = 0.5
    n_instances = 10
    instances = default_incremental_population(n_instances=n_instances)
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)

    other_instances = default_incremental_population(n_instances=n_instances // 2)
    other = UnstructuredArchive(
        k=k, novelty_threshold=threshold, instances=other_instances
    )

    assert archive != other


def test_unstructured_archive_compare_raises_if_wrong_object():
    k = 10
    threshold = 0.5
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=None)
    with pytest.raises(TypeError):
        archive == np.ndarray


def test_unstructured_archive_compare_raises_if_wrong_class():
    k = 10
    threshold = 0.5
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=None)
    grid_archive = GridArchive(dimensions=(2, 2), ranges=[(0.0, 1.0), (0.0, 1.0)])
    with pytest.raises(TypeError):
        archive == grid_archive


def test_unstructured_archive_compute_when_empty():
    k = 10
    threshold = 0.5
    n_instances = 10
    instances = default_incremental_population(n_instances=n_instances)
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=None)
    descriptors = [x.descriptor for x in instances]
    novelty_scores = archive.compute_novelty(descriptors=descriptors)
    assert isinstance(novelty_scores, np.ndarray)
    assert_equal(novelty_scores, np.full(n_instances, fill_value=threshold))


def test_unstructured_archive_compute_with_duplicated():
    k = 10
    threshold = 0.5
    n_instances = 10
    descriptor_dim = 4

    instances = default_incremental_population(
        n_instances=n_instances, descriptor_dim=descriptor_dim
    )
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)

    new_descriptors = np.asarray([
        range(descriptor_dim) for _ in range(n_instances // 2)
    ])
    novelty_scores = archive.compute_novelty(new_descriptors)
    assert isinstance(novelty_scores, np.ndarray)
    assert len(novelty_scores) == n_instances // 2
    # Returns zeros because all descriptors are already inside the archive
    assert_equal(novelty_scores, np.zeros(n_instances // 2))


def test_unstructured_archive_compute_non_duplicated():
    k = 10
    threshold = 0.5
    n_instances = 10
    descriptor_dim = 4

    instances = default_incremental_population(
        n_instances=n_instances, descriptor_dim=descriptor_dim
    )
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)

    new_descriptors = np.asarray([
        range(descriptor_dim, descriptor_dim * 2) for _ in range(n_instances // 2)
    ])
    novelty_scores = archive.compute_novelty(new_descriptors)
    assert isinstance(novelty_scores, np.ndarray)
    assert len(novelty_scores) == n_instances // 2
    # Returns the same because all descriptors are equal
    assert_allclose(novelty_scores, np.full(n_instances // 2, fill_value=5.09090909))


def test_unstructured_archive_compute_random_descriptors():
    k = 10
    threshold = 0.5
    n_instances = 10
    descriptor_dim = 4

    instances = default_incremental_population(
        n_instances=n_instances, descriptor_dim=descriptor_dim
    )
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)

    new_descriptors = np.random.default_rng().integers(
        low=0, high=10, size=(n_instances, descriptor_dim)
    )
    novelty_scores = archive.compute_novelty(new_descriptors)
    assert isinstance(novelty_scores, np.ndarray)
    assert len(novelty_scores) == n_instances
    # Non-duplicated descriptor, shouldn't return zero
    assert all(0.0 <= n_score_i >= threshold for n_score_i in novelty_scores)


@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_unstructured_archive_extend_with_only_instances(threshold):
    k = 10
    n_instances = 10
    dimension = 10
    descriptor_dim = 4

    instances = default_incremental_population(
        n_instances=n_instances, dimension=dimension, descriptor_dim=descriptor_dim
    )
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    rng = np.random.default_rng()
    new_instances = [
        Instance(
            variables=list(range(dimension)),
            descriptor=rng.integers(low=0, high=1_000, size=descriptor_dim),
        )
        for _ in range(n_instances)
    ]
    archive.extend(new_instances)
    assert len(archive) == (n_instances * 2)


@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_unstructured_archive_extend_with_instances_and_scores(threshold):
    k = 10
    n_instances = 10
    dimension = 10
    descriptor_dim = 4

    instances = default_incremental_population(
        n_instances=n_instances, dimension=dimension, descriptor_dim=descriptor_dim
    )
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    new_instances = [
        Instance(
            variables=list(range(dimension)),
            descriptor=tuple(range(descriptor_dim)),
        )
        for _ in range(n_instances)
    ]
    novelty_scores = np.full(n_instances, fill_value=threshold)
    archive.extend(new_instances, novelty_scores=novelty_scores)
    assert len(archive) == (n_instances * 2)


@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_unstructured_archive_extend_with_instances_scores_descriptor(threshold):
    k = 10
    n_instances = 10
    dimension = 10
    descriptor_dim = 4

    instances = default_incremental_population(
        n_instances=n_instances, dimension=dimension, descriptor_dim=descriptor_dim
    )
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    new_instances = [
        Instance(
            variables=list(range(dimension)),
        )
        for _ in range(n_instances)
    ]
    descriptors = np.random.default_rng().integers(
        low=0, high=10, size=(n_instances, descriptor_dim)
    )
    novelty_scores = np.full(n_instances, fill_value=threshold)
    archive.extend(
        new_instances, novelty_scores=novelty_scores, descriptors=descriptors
    )
    assert len(archive) == (n_instances * 2)


def test_unstructured_archive_extend_raises_not_instances():
    k = 10
    n_instances = 10
    threshold = 0.5
    instances = list(range(n_instances))

    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=None)
    with pytest.raises(TypeError):
        archive.extend(instances)


def test_unstructured_archive_extend_raises_len_mismatch():
    k = 10
    n_instances = 10
    dimension = 10
    descriptor_dim = 4
    threshold = 0.5

    instances = default_incremental_population(
        n_instances=n_instances, dimension=dimension, descriptor_dim=descriptor_dim
    )
    archive = UnstructuredArchive(k=k, novelty_threshold=threshold, instances=instances)
    new_instances = [
        Instance(
            variables=list(range(dimension)),
        )
        for _ in range(n_instances)
    ]
    # Only half of descriptor were given
    with pytest.raises(ValueError):
        descriptors = np.random.default_rng().integers(
            low=0, high=10, size=(n_instances // 2, descriptor_dim)
        )
        archive.extend(new_instances, descriptors=descriptors)

    # Only half of scores were given
    with pytest.raises(ValueError):
        novelty_scores = np.full(n_instances // 2, fill_value=threshold)
        archive.extend(new_instances, novelty_scores=novelty_scores)

    # Half of scores and double of descriptors
    with pytest.raises(ValueError):
        novelty_scores = np.full(n_instances // 2, fill_value=threshold)
        descriptors = np.random.default_rng().integers(
            low=0, high=10, size=(n_instances * 2, descriptor_dim)
        )
        archive.extend(
            new_instances, novelty_scores=novelty_scores, descriptors=descriptors
        )
