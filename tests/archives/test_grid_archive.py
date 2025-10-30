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

import pytest


from digneapy import GridArchive, Instance
from digneapy.domains.kp import KnapsackDomain
import numpy as np


@pytest.fixture
def grid_5d():
    return GridArchive(
        dimensions=(20, 20, 20, 20, 20),
        ranges=[
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
        ],
    )


def test_grid_archive_raises():
    # Raises ValueError when dimension < 1
    rng = np.random.default_rng(seed=42)
    with pytest.raises(ValueError):
        _ = GridArchive(dimensions=[], ranges=[])

    # Raises because instances are not of type Instance
    with pytest.raises(TypeError):
        _ = GridArchive(
            dimensions=(2, 2),
            ranges=[(0, 10), (0, 10)],
            instances=rng.integers(low=0, high=10, size=(10, 2)),
        )

    with pytest.raises(ValueError):
        _ = GridArchive(
            dimensions=[],
            ranges=[(0, 10), (0, 10), (0, 10)],
        )
    with pytest.raises(ValueError):
        _ = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10), (0, 10)])

    with pytest.raises(ValueError):
        _ = GridArchive(dimensions=(2, 2), ranges=[])

    # Raises because instance is not of type Instance
    with pytest.raises(TypeError):
        archive = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10)])
        archive.append(rng.integers(low=0, high=10, size=(10, 2)))

    # Raises out-of lower bound
    with pytest.raises(ValueError):
        archive = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10)])
        _ = archive.lower_i(-1)

    # Raises out-of lower bound
    with pytest.raises(ValueError):
        archive = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10)])
        _ = archive.lower_i(100)

    # Raises out-of upper bound
    with pytest.raises(ValueError):
        archive = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10)])
        _ = archive.upper_i(-1)

    # Raises out-of upper bound
    with pytest.raises(ValueError):
        archive = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10)])
        _ = archive.upper_i(100)

    # Raises index shape is not valid
    with pytest.raises(ValueError):
        archive = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10)])
        _ = archive.index_of(rng.integers(low=0, high=10, size=(10, 10)))


def test_grid_archive_populated():
    instances = []
    rng = np.random.default_rng(seed=42)
    for _ in range(10):
        instance = Instance()
        instance.features = rng.integers(low=0, high=10, size=2)
        instance.descriptor = tuple(instance.features)
        instances.append(instance)
    archive = GridArchive(
        dimensions=(2, 2),
        ranges=[(0, 10), (0, 10)],
        instances=instances,
    )
    assert len(archive) != 0
    assert all(archive.lower_i(i) == 0 for i in range(len(archive.bounds)))
    assert all(archive.upper_i(i) == 10 for i in range(len(archive.bounds)))
    assert all(isinstance(i, Instance) for i in archive)


def test_grid_5d(grid_5d):
    assert len(grid_5d) == 0
    assert len(grid_5d.bounds) == len(grid_5d.dimensions)
    grid_zero = list(0 for _ in range(5))
    grid_one = list(1 for _ in range(5))
    index_of_zero = 0
    index_of_one = 168421
    assert grid_5d._grid_to_int_index(grid_zero) == index_of_zero
    assert grid_5d._grid_to_int_index(grid_one) == index_of_one
    np.testing.assert_array_equal(
        grid_5d.int_to_grid_index(index_of_zero), np.asarray(grid_zero)
    )
    np.testing.assert_array_equal(
        grid_5d.int_to_grid_index(index_of_one), np.asarray(grid_one)
    )

    expected_str = f"GridArchive(dim={grid_5d.dimensions},cells={grid_5d._cells},bounds={grid_5d.bounds})"
    assert grid_5d.__str__() == expected_str
    assert grid_5d.__repr__() == expected_str

    data = grid_5d.asdict()
    assert isinstance(data, dict)
    assert "dimensions" in data
    assert len(data["dimensions"]) == len(grid_5d.dimensions)
    assert "lbs" in data
    assert "ubs" in data
    assert len(data["lbs"]) == len(data["ubs"])
    assert len(data["lbs"]) == len(grid_5d.dimensions)
    assert "n_cells" in data
    assert "instances" in data
    assert isinstance(data["instances"], dict)
    assert all(isinstance(i, Instance) for i in data["instances"])
    assert isinstance(grid_5d.to_json(), str)


def test_grid_archive_5d_storage(grid_5d):
    n_instances = 10
    rng = np.random.default_rng(seed=42)
    domain = KnapsackDomain(dimension=100)
    instances = domain.generate_instances(n=n_instances)
    descriptors = rng.random(size=(n_instances, 5))

    for i in range(10):
        instances[i].fitness = rng.random(size=1)

    assert len(grid_5d) == 0
    grid_5d.extend(instances, descriptors=descriptors)
    assert len(grid_5d) == len(instances)

    instance = Instance()
    instance.descriptor = rng.random(size=5)
    grid_5d.append(instance, descriptor=instance.descriptor)
    assert len(grid_5d) == len(instances) + 1

    grid_5d.remove([instance.descriptor])
    assert len(grid_5d) == len(instances)

    with pytest.raises(ValueError):
        grid_5d.remove([None])


def test_grid_limits():
    rng = np.random.default_rng(seed=42)

    archive = GridArchive(
        dimensions=(5, 5),
        ranges=[(0, 100), (0, 100)],
        dtype=np.int32,
    )

    assert np.isclose(archive.coverage, 0.0)  # Empty archive
    assert len(list(iter(archive))) == 0
    instances = []
    max_allowed = 25
    n_instances = 1000
    for _ in range(n_instances):
        inst = Instance([], fitness=0.0, p=rng.integers(0, 100), s=rng.random())
        inst.features = tuple(rng.integers(low=0, high=100, size=2))
        inst.descriptor = inst.features
        instances.append(inst)

    assert len(archive) == 0
    assert archive.n_cells == max_allowed
    archive.extend(instances)
    assert len(archive) == max_allowed
    assert archive.coverage <= 1.0


def test_grid_extend_outside_bounds():
    rng = np.random.default_rng(seed=42)

    archive = GridArchive(
        dimensions=(5, 5),
        ranges=[(0, 100), (0, 100)],
        dtype=np.int32,
    )
    instances = []
    max_allowed = 25
    n_instances = 1000
    for _ in range(n_instances):
        inst = Instance(
            [],
            fitness=rng.integers(100, 1000),
            p=rng.integers(100, 1000),
            s=rng.random(),
        )
        inst.features = tuple(rng.integers(low=1000, high=10000, size=2))
        inst.descriptor = inst.features
        instances.append(inst)

    assert len(archive) == 0
    assert archive.n_cells == max_allowed
    archive.extend(instances)
    # Out-of-bounds only inserts one in the very last cell available
    assert len(archive) == 1
    filled_cells = list(archive.filled_cells)
    assert filled_cells[0] == 24


def test_grid_extend_under_bounds():
    rng = np.random.default_rng(seed=42)

    archive = GridArchive(
        dimensions=(5, 5),
        ranges=[(100, 1000), (100, 1000)],
        dtype=np.int32,
    )
    instances = []
    max_allowed = 25
    n_instances = 1000
    for _ in range(n_instances):
        inst = Instance(
            [],
            fitness=rng.integers(100, 1000),
            p=rng.integers(100, 1000),
            s=rng.random(),
        )
        inst.features = tuple(rng.integers(low=1, high=99, size=2))
        inst.descriptor = inst.features
        instances.append(inst)

    assert len(archive) == 0
    assert archive.n_cells == max_allowed
    archive.extend(instances)
    # Out-of-bounds only inserts one in the very first cell available
    assert len(archive) == 1
    filled_cells = list(archive.filled_cells)
    assert filled_cells[0] == 0


def test_grid_archive_with_KP_instances_and_features_descriptor():
    archive = GridArchive(
        dimensions=(20, 20, 20, 20, 20, 20, 20, 20),
        ranges=[
            (700, 30000),
            (890, 1000),
            (860, 1000.0),
            (1.0, 200),
            (1.0, 230.0),
            (0.10, 12.0),
            (400, 610),
            (240, 330),
        ],
    )
    n_instances = 1_000
    domain = KnapsackDomain(dimension=50)
    raw_instances = domain.generate_instances(n_instances)
    features = domain.extract_features(raw_instances)
    instances = [
        Instance(
            variables=raw_instances[i], features=features[i], descriptor=features[i]
        )
        for i in range(n_instances)
    ]

    assert len(archive) == 0
    assert archive.n_cells == np.prod(np.array((20,) * 8))
    archive.extend(instances)
    assert len(archive) > 0
    assert all(idx > 0 and idx < archive.n_cells for idx in archive.filled_cells)


def test_grid_archive_with_KP_instances_separated_descriptors():
    archive = GridArchive(
        dimensions=(20, 20, 20, 20, 20, 20, 20, 20),
        ranges=[
            (700, 30000),
            (890, 1000),
            (860, 1000.0),
            (1.0, 200),
            (1.0, 230.0),
            (0.10, 12.0),
            (400, 610),
            (240, 330),
        ],
    )
    n_instances = 1_000
    domain = KnapsackDomain(dimension=50)
    instances = domain.generate_instances(n_instances)
    features = domain.extract_features(instances)

    assert len(archive) == 0
    assert archive.n_cells == np.prod(np.array((20,) * 8))
    archive.extend(instances, descriptors=features)
    assert len(archive) > 0 and len(archive) <= 1000
    assert all(idx > 0 and idx < archive.n_cells for idx in archive.filled_cells)


def test_grid_archive_getitem():
    rng = np.random.default_rng(seed=42)

    instances = []
    for _ in range(1000):
        instance = Instance()
        instance.features = rng.integers(low=0, high=10, size=2)
        instance.descriptor = instance.features
        instances.append(instance)

    archive = GridArchive(
        dimensions=(10, 10),
        ranges=[(0, 10), (0, 10)],
        instances=instances,
    )
    results = archive[[0, 11], [0, 5]]
    assert isinstance(results, list)
    assert len(results) == 2
    results = archive[[0, 5]]
    assert isinstance(results, list)
    assert len(results) == 2

    result_simple = archive[0, 5]
    assert isinstance(result_simple, list)
    assert len(result_simple) == 2
    assert result_simple == results
