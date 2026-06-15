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

from digneapy import GridArchive, Instance

from .conftest import default_incremental_population, population_with_custom_descriptors


def test_grid_archive_attrs():
    archive = GridArchive(
        dimensions=(2, 2),
        ranges=[
            (-1.0, 1.0),
            (-1.0, 1.0),
        ],
    )

    assert len(archive) == 0
    assert_equal(archive.dimensions, (2, 2))
    assert archive.n_cells == 4
    assert isinstance(archive.filled_cells, set)
    assert len(archive.filled_cells) == len(archive)
    assert_equal(archive.coverage, np.float64(0.0))

    assert len(list(archive.instances)) == 0
    assert_equal(archive.bounds, [[-1.0, 1.0], [-1.0, 1.0]])

    # Checking the bounds of each dimension
    for i in range(2):
        assert archive.lower_i(i) == -1.0
        assert archive.upper_i(i) == 1.0

    data = archive.to_dict()
    assert isinstance(data, dict)
    expected_keys = ("instances", "dimensions", "lbs", "ubs", "n_cells")
    assert all(key in data.keys() for key in expected_keys)


def test_grid_archive_with_initial_instances():
    dimensions = 2
    n_instances = 10
    # All unique descriptors
    descriptors = np.random.choice(
        np.arange(100, dtype=np.float64),
        size=(n_instances, dimensions),
        replace=False,
    )
    instances = population_with_custom_descriptors(
        descriptors, n_instances=10, dimension=2
    )
    archive = GridArchive(
        dimensions=(10, 10),
        ranges=[
            (0.0, 100.0),
            (0.0, 100.0),
        ],
        instances=instances,
    )
    # We cannot know for sure how many instances will
    # be included in the archive. But, at least the
    # half should be included.
    minimum_expected = n_instances // 2
    assert minimum_expected <= len(archive) <= n_instances
    assert all(isinstance(x, Instance) for x in archive.instances)


def test_grid_archive_init_raises_if_wrong_args():
    # Raises ValueError when dimension < 1
    rng = np.random.default_rng()

    with pytest.raises(ValueError):
        _ = GridArchive(dimensions=[], ranges=[])

    # Raises because instances are not of type Instance
    with pytest.raises(TypeError):
        _ = GridArchive(
            dimensions=(2, 2),
            ranges=[(0, 10), (0, 10)],
            instances=rng.integers(low=0, high=10, size=(10, 2)),
        )

    # Raises because the dimension and ranges have different lengths
    with pytest.raises(ValueError):
        _ = GridArchive(
            dimensions=[],
            ranges=[(0, 10), (0, 10), (0, 10)],
        )

    # Raises because the dimension and ranges have different lengths
    with pytest.raises(ValueError):
        _ = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10), (0, 10)])

    # Raises because the dimension and ranges have different lengths
    with pytest.raises(ValueError):
        _ = GridArchive(dimensions=(2, 2), ranges=[])


def test_grid_archive_lower_bound():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    assert all(archive.lower_i(i) == -1.0 for i in range(2))


def test_grid_archive_upper_bound():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    assert all(archive.upper_i(i) == 1.0 for i in range(2))


def test_grid_archive_lower_bound_raises():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    # Raises out-of lower bound
    with pytest.raises(IndexError):
        _ = archive.lower_i(-1)

    # Raises out-of lower bound
    with pytest.raises(IndexError):
        _ = archive.lower_i(100)

    with pytest.raises(TypeError):
        _ = archive.lower_i("abc")


def test_grid_archive_upper_bound_raises():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    # Raises out-of lower bound
    with pytest.raises(IndexError):
        _ = archive.upper_i(-1)

    # Raises out-of lower bound
    with pytest.raises(IndexError):
        _ = archive.upper_i(100)

    with pytest.raises(TypeError):
        _ = archive.upper_i("abc")


def test_grid_archive_iterable():
    archive = GridArchive(dimensions=(10, 10), ranges=[(0, 10), (0, 10)])
    instances = default_incremental_population(n_instances=10, descriptor_dim=2)
    archive.extend(instances)
    assert all(isinstance(x, Instance) for x in archive)


def test_grid_archive_index_of_returns_zero():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    assert_equal(archive.index_of([]), np.empty(0))


def test_grid_archive_index_raises_1d_len():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    descriptor = [1, 2, 3, 4]
    # Descriptor has 4d and the grid is a 2d space
    with pytest.raises(ValueError):
        _ = archive.index_of(descriptor)


def test_grid_archive_index_raises_2d_shape():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    n_instances = 10
    descriptors = [[1, 2, 3, 4] for _ in range(n_instances)]
    # Descriptor has 4d and the grid is a 2d space
    with pytest.raises(ValueError):
        _ = archive.index_of(descriptors)


def test_grid_archive_extend_only_instances():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    instance = Instance(variables=[1, 2, 3, 4], descriptor=[0, 0])

    assert len(archive) == 0
    assert len(archive.filled_cells) == 0
    archive.extend([instance])

    assert len(archive) == 1
    assert len(archive.filled_cells) == 1
    expected_coverage = 1 / 4
    assert_allclose(archive.coverage, expected_coverage)


def test_grid_archive_extend_instances_and_descriptor():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    instance = Instance(variables=[1, 2, 3, 4])
    descriptor = [0, 0]

    assert len(archive) == 0
    assert len(archive.filled_cells) == 0
    archive.extend([instance], [descriptor])

    assert len(archive) == 1
    assert len(archive.filled_cells) == 1
    expected_coverage = 1 / 4
    assert_allclose(archive.coverage, expected_coverage)


def test_grid_archive_extend_instances_improves():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    descriptor = [0, 0]

    instance = Instance(variables=[1, 2, 3, 4], fitness=1.0, descriptor=descriptor)

    assert len(archive) == 0
    assert len(archive.filled_cells) == 0
    archive.extend([instance])

    assert len(archive) == 1
    assert len(archive.filled_cells) == 1
    expected_coverage = 1 / 4
    assert_allclose(archive.coverage, expected_coverage)

    new_fitness = 100
    new_instance = Instance(
        variables=[1, 2, 3, 4], fitness=new_fitness, descriptor=descriptor
    )
    # This new instance occupies the same cell than the previous one
    archive.extend([new_instance])
    assert len(archive) == 1
    assert len(archive.filled_cells) == 1
    assert_allclose(archive.coverage, expected_coverage)


def test_grid_archive_extend_instance_and_descriptor_improves():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    descriptor = [0, 0]

    instance = Instance(variables=[1, 2, 3, 4], fitness=1.0)

    assert len(archive) == 0
    assert len(archive.filled_cells) == 0
    archive.extend([instance], [descriptor])

    assert len(archive) == 1
    assert len(archive.filled_cells) == 1
    expected_coverage = 1 / 4
    assert_allclose(archive.coverage, expected_coverage)

    new_fitness = 100
    new_instance = Instance(variables=[1, 2, 3, 4], fitness=new_fitness)
    # This new instance occupies the same cell than the previous one
    archive.extend([new_instance], [descriptor])
    assert len(archive) == 1
    assert len(archive.filled_cells) == 1
    assert_allclose(archive.coverage, expected_coverage)


def test_grid_archive_extend_instances_and_descriptor_raises():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    instance = Instance(variables=[1, 2, 3, 4])
    descriptors = [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        archive.extend([instance], descriptors)


def test_grid_archive_extend_not_valid_instances_raises():
    archive = GridArchive(dimensions=(2, 2), ranges=[(-1.0, 1.0), (-1.0, 1.0)])
    instances = [[0, 0], [1, 1]]

    with pytest.raises(TypeError):
        archive.extend(instances)


def test_grid_archive_extends_large_dimensions():
    dimensions = 100
    archive = GridArchive(
        dimensions=(2,) * dimensions, ranges=[[-1, 1.0] for _ in range(dimensions)]
    )
    instances = default_incremental_population(
        n_instances=10, descriptor_dim=dimensions
    )
    assert len(archive) == 0
    archive.extend(instances)
    assert len(archive) > 0
    assert len(archive) <= 10
