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

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy.archives._cvt_archive import CVTArchive, compute_centroids
from digneapy.core import Instance

from .conftest import population_with_custom_descriptors


def test_compute_centroids():
    n_centroids = 10
    descriptor_dimension = 100
    n_samples = 10_000
    lwi = 0
    hwi = 100
    lower_bounds = np.full(descriptor_dimension, fill_value=lwi)
    upper_bounds = np.full(descriptor_dimension, fill_value=hwi)

    centroids, samples = compute_centroids(
        n_centroids, descriptor_dimension, lower_bounds, upper_bounds, n_samples
    )

    assert isinstance(centroids, np.ndarray)
    assert_equal(centroids.shape, (n_centroids, descriptor_dimension))

    assert isinstance(samples, np.ndarray)
    assert_equal(samples.shape, (n_samples, descriptor_dimension))
    for sample in samples:
        # All samples are inside the expected bounds
        assert all(lwi <= s_i <= hwi for s_i in sample)


def test_compute_centroids_raises_not_enough_centroids():
    n_centroids = 1_000
    descriptor_dimension = 100
    n_samples = 100
    lwi = 0
    hwi = 100
    lower_bounds = np.full(descriptor_dimension, fill_value=lwi)
    upper_bounds = np.full(descriptor_dimension, fill_value=hwi)

    with pytest.raises(RuntimeError):
        # Not enough samples to compute 1,000 centroids
        centroids, samples = compute_centroids(
            n_centroids, descriptor_dimension, lower_bounds, upper_bounds, n_samples
        )


def test_compute_centroids_with_samples_ndarray():
    n_centroids = 10
    descriptor_dimension = 100
    n_samples = 10_000
    lwi = 0
    hwi = 100
    lower_bounds = np.full(descriptor_dimension, fill_value=lwi)
    upper_bounds = np.full(descriptor_dimension, fill_value=hwi)
    expected_samples = np.random.uniform(
        lwi, hwi, size=(n_samples, descriptor_dimension)
    )

    centroids, samples = compute_centroids(
        n_centroids, descriptor_dimension, lower_bounds, upper_bounds, expected_samples
    )

    assert isinstance(centroids, np.ndarray)
    assert_equal(centroids.shape, (n_centroids, descriptor_dimension))

    assert_equal(samples, expected_samples)
    assert samples is expected_samples


def test_compute_centroids_with_samples_ndarray_raises_wrong_dim():
    n_centroids = 10
    descriptor_dimension = 100
    n_samples = 10_000
    lwi = 0
    hwi = 100
    lower_bounds = np.full(descriptor_dimension, fill_value=lwi)
    upper_bounds = np.full(descriptor_dimension, fill_value=hwi)
    expected_samples = np.random.uniform(
        lwi, hwi, size=(n_samples, descriptor_dimension * 2)
    )

    with pytest.raises(ValueError):
        # Raises because the expected_samples has the double of dimensions
        centroids, samples = compute_centroids(
            n_centroids,
            descriptor_dimension,
            lower_bounds,
            upper_bounds,
            expected_samples,
        )


def test_cvt_archive_attrs():
    dimensions = 100
    n_centroids = 10
    low_i = 0
    high_i = 100
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    archive = CVTArchive(dimensions=dimensions, centroids=n_centroids, ranges=ranges)

    assert archive.dimensions == dimensions
    assert len(archive) == 0
    assert_equal(archive.lower_bounds, np.full(dimensions, fill_value=low_i))
    assert_equal(archive.upper_bounds, np.full(dimensions, fill_value=high_i))

    centroids = archive.centroids
    assert isinstance(centroids, np.ndarray)
    assert_equal(centroids.shape, (n_centroids, dimensions))

    data = archive.to_dict()
    assert isinstance(data, dict)
    expected_keys = ("instances", "dimensions", "lbs", "ubs", "centroids")
    assert all(key in data.keys() for key in expected_keys)


def test_cvt_archive_can_be_iterated():
    dimensions = 2
    n_centroids = 10
    low_i = 0
    high_i = 100
    n_instances = 3
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    # All unique descriptors that must fall in differente cells
    descriptors = np.asarray([[0, 0], [50, 50], [90, 90]])
    instances = population_with_custom_descriptors(
        descriptors,
        n_instances=n_instances,
        dimension=dimensions,
    )
    archive = CVTArchive(dimensions=dimensions, centroids=n_centroids, ranges=ranges)
    archive.extend(instances)
    # The archive is preloaded with initial instances
    assert len(archive) == n_instances
    assert all(isinstance(x, Instance) for x in archive)


def test_cvt_archive_with_initial_instances():
    dimensions = 2
    n_centroids = 10
    low_i = 0
    high_i = 100
    n_instances = 3
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    # All unique descriptors that must fall in differente cells
    descriptors = np.asarray([[0, 0], [50, 50], [90, 90]])
    instances = population_with_custom_descriptors(
        descriptors,
        n_instances=n_instances,
        dimension=dimensions,
    )
    archive = CVTArchive(
        dimensions=dimensions, centroids=n_centroids, ranges=ranges, instances=instances
    )
    # The archive is preloaded with initial instances
    assert len(archive) == n_instances
    assert all(isinstance(x, Instance) for x in archive.instances)


def test_cvt_archive_init_raises():
    dimensions = 100
    n_centroids = 10
    low_i = 0
    high_i = 100
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    with pytest.raises(ValueError):
        # Raises because dimensions is negative
        _ = CVTArchive(dimensions=-100, centroids=n_centroids, ranges=ranges)

    with pytest.raises(ValueError):
        # Raises because dimensions is not a valid integer
        _ = CVTArchive(dimensions="abc", centroids=n_centroids, ranges=ranges)

    with pytest.raises(ValueError):
        # Raises because dimensions != len(ranges)
        _ = CVTArchive(dimensions=dimensions * 2, centroids=n_centroids, ranges=ranges)


def test_cvt_archive_can_save_centroids():
    dimensions = 100
    n_centroids = 10
    low_i = 0
    high_i = 100
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    archive = CVTArchive(dimensions=dimensions, centroids=n_centroids, ranges=ranges)

    filename = "sample_centroids.npy"
    archive.to_file(filename)

    expected_path = Path(filename)
    assert expected_path.exists()
    expected_path.unlink()


def test_cvt_archive_centroids_load_from_file():
    centroids = Path(__file__).parent / "testing_centroids_10_100_dimensions.npy"
    dimensions = 100
    n_centroids = 10
    low_i = 0
    high_i = 100
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    archive = CVTArchive(dimensions=dimensions, centroids=centroids, ranges=ranges)
    centroids = np.load(centroids)
    assert_equal(archive.centroids, centroids)
    assert_equal(archive.centroids.shape, (n_centroids, dimensions))


def test_cvt_archive_centroids_load_from_file_wrong_dimensions():
    centroids = Path(__file__).parent / "testing_centroids_10_100_dimensions.npy"
    dimensions = 200
    low_i = 0
    high_i = 100
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    with pytest.raises(ValueError):
        # Raises because centroids have the 100d and CVT expected 200
        _ = CVTArchive(dimensions=dimensions, centroids=centroids, ranges=ranges)


def test_cvt_archive_centroids_load_from_ndarray():
    centroids = np.load(
        Path(__file__).parent / "testing_centroids_10_100_dimensions.npy"
    )
    dimensions = 100
    n_centroids = 10
    low_i = 0
    high_i = 100
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    archive = CVTArchive(dimensions=dimensions, centroids=centroids, ranges=ranges)
    assert_equal(archive.centroids, centroids)
    assert_equal(archive.centroids.shape, (n_centroids, dimensions))


def test_cvt_archive_centroids_load_from_ndarray_raises():
    centroids = np.load(
        Path(__file__).parent / "testing_centroids_10_100_dimensions.npy"
    )
    dimensions = 200
    low_i = 0
    high_i = 100
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    with pytest.raises(ValueError):
        # Raises because centroids have the 100d and CVT expected 200
        _ = CVTArchive(dimensions=dimensions, centroids=centroids, ranges=ranges)


def test_cvt_archive_can_be_extended():
    dimensions = 2
    n_centroids = 10
    low_i = 0
    high_i = 1000
    n_instances = 10
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    archive = CVTArchive(dimensions=dimensions, centroids=n_centroids, ranges=ranges)

    # New descriptors should be unique but that doesn't mean that they fall in empty cells with CVTArchive
    descriptors = np.random.default_rng().integers(
        low=low_i, high=high_i, size=(n_instances, dimensions)
    )
    instances = population_with_custom_descriptors(
        descriptors,
        n_instances=n_instances,
        dimension=dimensions,
    )

    archive.extend(instances, descriptors)
    # Expects archive not to be empty and to have less than n_instances
    current_length = len(archive)
    assert 0 < current_length <= n_instances
    assert all(isinstance(x, Instance) for x in archive.instances)


def test_cvt_archive_preloaded_can_be_extended():
    dimensions = 2
    n_centroids = 10
    low_i = 0
    high_i = 100
    n_instances = 1
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    descriptors = np.asarray([[0, 10]])
    instances = population_with_custom_descriptors(
        descriptors,
        n_instances=n_instances,
        dimension=dimensions,
    )
    archive = CVTArchive(
        dimensions=dimensions, centroids=n_centroids, ranges=ranges, instances=instances
    )

    # New descriptors should be unique and fall in empty cells
    descriptors = np.asarray([[50, 50], [90, 90]])
    instances = population_with_custom_descriptors(
        descriptors,
        n_instances=2,
        dimension=dimensions,
    )

    archive.extend(instances, descriptors)
    # Expects to have 3 instances by now
    expected_len = n_instances + 2
    assert len(archive) == expected_len
    assert all(isinstance(x, Instance) for x in archive.instances)


def test_cvt_archive_retrieve_from_descriptors():
    dimensions = 2
    n_centroids = 10
    low_i = 0
    high_i = 100
    n_instances = 3
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    # All unique descriptors that must fall in differente cells
    descriptors = np.asarray([[0, 0], [50, 50], [90, 90]])
    instances = population_with_custom_descriptors(
        descriptors,
        n_instances=n_instances,
        dimension=dimensions,
    )
    archive = CVTArchive(
        dimensions=dimensions, centroids=n_centroids, ranges=ranges, instances=instances
    )

    retrieve_instances = archive.retrieve(descriptors)
    assert len(retrieve_instances) == len(descriptors)
    assert all(isinstance(x, Instance) for x in retrieve_instances)
    retrieve_descriptors = [x.descriptor for x in retrieve_instances]
    assert_equal(retrieve_descriptors, descriptors)


def test_cvt_archive_retrieve_from_filled_cells():
    dimensions = 2
    n_centroids = 10
    low_i = 0
    high_i = 100
    n_instances = 3
    ranges = [(low_i, high_i) for _ in range(dimensions)]

    # All unique descriptors that might fall in differente cells
    descriptors = np.asarray([[0, 0], [50, 50], [90, 90]])
    instances = population_with_custom_descriptors(
        descriptors,
        n_instances=n_instances,
        dimension=dimensions,
    )
    archive = CVTArchive(
        dimensions=dimensions, centroids=n_centroids, ranges=ranges, instances=instances
    )

    filled_cells = archive.filled_cells

    filled_cells = np.asarray(list(filled_cells))
    retrieve_instances = archive.retrieve_filled_cells(filled_cells)

    assert len(retrieve_instances) == len(filled_cells)
    assert all(isinstance(x, Instance) for x in retrieve_instances)
