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
import os

import numpy as np
import pytest

from digneapy import Archive, CVTArchive, GridArchive, Instance
from digneapy.domains.kp import KnapsackDomain


@pytest.fixture
def setup_and_teardown_cvt():
    def_filename = "default_cvt"

    cvt = CVTArchive(
        k=100, ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)], n_samples=100
    )
    print("Creating all the .npy files")
    # Creates all the files
    cvt.to_file(def_filename)
    _ = cvt.to_json(filename=def_filename)

    yield cvt
    # Cleaning all the files
    print("Cleaning all the .npy files")

    os.remove(f"{def_filename}_centroids.npy")
    os.remove(f"{def_filename}_samples.npy")
    os.remove(f"{def_filename}.json")


@pytest.fixture
def default_archive():
    instances = [Instance(variables=list(range(d, d + 5))) for d in range(10)]
    return Archive(threshold=0.0, instances=instances)


@pytest.fixture
def empty_archive():
    return Archive(threshold=0.0)


def test_empty_archive(empty_archive):
    assert 0 == len(empty_archive)
    assert empty_archive.threshold == 0.0
    with pytest.raises(TypeError):
        empty_archive.threshold = "HELLO"


def test_archive_to_array(default_archive):
    np_archive = np.array(default_archive)
    assert len(np_archive) == len(default_archive)
    assert isinstance(np_archive, np.ndarray)
    assert default_archive.threshold == 0.0


def test_iterable_default_archive(default_archive):
    descriptors = [Instance(range(d, d + 5)) for d in range(10)]
    assert len(descriptors) == len(default_archive)
    assert all(a == b for a, b in zip(descriptors, default_archive))
    assert all(a == b for a, b in zip(iter(descriptors), iter(default_archive)))


def test_not_equal_archives(default_archive, empty_archive):
    assert default_archive != empty_archive


def test_equal_archives(default_archive):
    a1 = copy.copy(default_archive)
    assert default_archive == a1


def test_append_instance(empty_archive):
    assert 0 == len(empty_archive)
    instance = Instance(variables=list(range(100)))
    empty_archive.append(instance)
    assert 1 == len(empty_archive)
    assert [instance] == empty_archive.instances
    d = list(range(10))
    with pytest.raises(TypeError):
        empty_archive.append(d)


def test_extend_iterable(empty_archive, default_archive):
    assert 0 == len(empty_archive)
    instances = default_archive.instances
    for i in instances:
        i.s = 10.0

    empty_archive.extend(instances)
    assert len(empty_archive) == len(default_archive)
    assert empty_archive == default_archive


def test_bool_on_empty_archive(empty_archive):
    assert not empty_archive


def test_bool_on_default_archive(default_archive):
    assert default_archive


def test_archive_magic(default_archive):
    assert default_archive.__str__() == "Archive(threshold=0.0,data=(|10|))"
    duplicated = copy.deepcopy(default_archive)
    assert hash(duplicated) == hash(default_archive)


def test_archive_access(default_archive):
    assert len(default_archive) == 10
    assert isinstance(default_archive[0], Instance)
    assert len(default_archive[:2]) == 2
    with pytest.raises(IndexError):
        default_archive[100]


def test_archive_format(default_archive):
    assert (
        "(Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()))"
        == format(
            default_archive,
        )
    )


def test_archive_repr(default_archive):
    assert "Archive(threshold=0.0,data=(|10|))" == format(
        repr(default_archive),
    )


def test_archive_extend(empty_archive):
    new_threshold = 1.0
    empty_archive.threshold = new_threshold
    instances = [
        Instance([], fitness=0.0, p=0.0, s=np.random.random()) for _ in range(10)
    ]

    expected = len(list(filter(lambda i: i.s >= empty_archive.threshold, instances)))
    empty_archive.extend(instances)
    assert len(empty_archive) == expected


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
    with pytest.raises(ValueError):
        _ = GridArchive(dimensions=[], ranges=[])

    # Raises because instances are not of type Instance
    with pytest.raises(TypeError):
        instances = np.random.randint(low=0, high=10, size=(10, 2))
        _ = GridArchive(
            dimensions=(2, 2),
            ranges=[(0, 10), (0, 10)],
            instances=instances,
        )

    # Raises because instance is not of type Instance
    with pytest.raises(TypeError):
        instance = np.random.randint(low=0, high=10, size=2)
        archive = GridArchive(dimensions=(2, 2), ranges=[(0, 10), (0, 10)])
        archive.append(instance)

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
        descriptors = np.random.randint(low=0, high=10, size=(10, 10))
        _ = archive.index_of(descriptors)


def test_grid_archive_populated():
    instances = []
    for _ in range(10):
        instance = Instance()
        instance.features = np.random.randint(low=0, high=10, size=2)
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


def test_grid_5d_storage(grid_5d):
    instances = []
    for _ in range(10):
        inst = Instance(
            [], fitness=0.0, p=np.random.randint(0, 100), s=np.random.random()
        )
        inst.features = tuple(np.random.random(size=5))
        inst.descriptor = inst.features
        instances.append(inst)

    assert len(grid_5d) == 0
    grid_5d.extend(instances)
    assert len(grid_5d) == len(instances)

    instance = Instance()
    instance.features = tuple(np.random.random(size=5))
    instance.descriptor = instance.features
    grid_5d.append(instance)
    assert len(grid_5d) == len(instances) + 1


def test_grid_limits():
    archive = GridArchive(
        dimensions=(5, 5),
        ranges=[(0, 100), (0, 100)],
        dtype=np.int32,
    )

    assert archive.coverage == 0.0  # Empty archive
    assert len(list(iter(archive))) == 0
    instances = []
    max_allowed = 25
    n_instances = 1000
    for _ in range(n_instances):
        inst = Instance(
            [], fitness=0.0, p=np.random.randint(0, 100), s=np.random.random()
        )
        inst.features = tuple(np.random.randint(low=0, high=100, size=2))
        inst.descriptor = inst.features
        instances.append(inst)

    assert len(archive) == 0
    assert archive.n_cells == max_allowed
    archive.extend(instances)
    assert len(archive) == max_allowed
    assert archive.coverage <= 1.0


def test_grid_extend_outside_bounds():
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
            fitness=np.random.randint(100, 1000),
            p=np.random.randint(100, 1000),
            s=np.random.random(),
        )
        inst.features = tuple(np.random.randint(low=1000, high=10000, size=2))
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
            fitness=np.random.randint(100, 1000),
            p=np.random.randint(100, 1000),
            s=np.random.random(),
        )
        inst.features = tuple(np.random.randint(low=1, high=99, size=2))
        inst.descriptor = inst.features
        instances.append(inst)

    assert len(archive) == 0
    assert archive.n_cells == max_allowed
    archive.extend(instances)
    # Out-of-bounds only inserts one in the very first cell available
    assert len(archive) == 1
    filled_cells = list(archive.filled_cells)
    assert filled_cells[0] == 0


def test_grid_with_kp_instances():
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
    domain = KnapsackDomain(dimension=50, capacity_approach="percentage")
    instances = [domain.generate_instance() for _ in range(n_instances)]
    for instance in instances:
        instance.features = domain.extract_features(instance)
        instance.descriptor = instance.features

    assert len(archive) == 0
    assert archive.n_cells == np.prod(np.array((20,) * 8))
    archive.extend(instances)
    assert len(archive) > 0
    assert all(idx > 0 and idx < archive.n_cells for idx in archive.filled_cells)


def test_grid_archive_getitem():
    instances = []
    for _ in range(1000):
        instance = Instance()
        instance.features = np.random.randint(low=0, high=10, size=2)
        instance.descriptor = instance.features
        instances.append(instance)

    archive = GridArchive(
        dimensions=(10, 10),
        ranges=[(0, 10), (0, 10)],
        instances=instances,
    )
    results = archive[[0, 11], [0, 5]]
    assert isinstance(results, dict)
    assert len(results) == 2
    results = archive[[0, 5]]
    assert isinstance(results, dict)
    assert len(results) == 1


def test_cvt_can_be_create():
    cvt = CVTArchive(
        k=100, ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)], n_samples=100
    )
    assert cvt.dimensions == 3
    assert cvt.regions == 100
    assert cvt._n_samples == 100
    assert len(cvt.samples) == 100
    assert len(cvt.centroids) == 100
    assert all(min(c_i) >= 0.0 for c_i in cvt.centroids)
    assert all(max(c_i) <= 100.0 for c_i in cvt.centroids)
    assert all(len(c_i) == 3 for c_i in cvt.centroids)
    lbs, ubs = cvt.bounds
    assert all(lbs_i == 0.0 for lbs_i in lbs)
    assert all(ubs_i == 100.0 for ubs_i in ubs)
    assert all(min(s_i) >= 0.0 for s_i in cvt.samples)
    assert all(max(s_i) <= 100.0 for s_i in cvt.samples)

    def_filename = "default_cvt_1_use"
    cvt.to_file(def_filename)
    assert os.path.exists(f"{def_filename}_centroids.npy")
    assert os.path.exists(f"{def_filename}_samples.npy")
    os.remove(f"{def_filename}_centroids.npy")
    os.remove(f"{def_filename}_samples.npy")

    json_data = cvt.to_json(filename=def_filename)
    assert isinstance(json_data, str)
    assert os.path.exists(f"{def_filename}.json")
    os.remove(f"{def_filename}.json")


def test_cvt_init_raises(setup_and_teardown_cvt):
    with pytest.raises(ValueError) as e:
        _ = CVTArchive(
            k=-1, ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)], n_samples=100
        )
    assert e.value.args[0] == "The number of regions (k = -1) must be >= 1"

    with pytest.raises(ValueError) as e:
        _ = CVTArchive(k=100, ranges=[], n_samples=100)
    assert e.value.args[0] == "ranges must have length >= 1 and it has length 0"

    with pytest.raises(ValueError) as e:
        _ = CVTArchive(
            k=100, ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)], n_samples=-1
        )
    assert (
        e.value.args[0]
        == "The number of samples (n_samples = -1) must be >= 1 and >= regions (k = 100)"
    )

    with pytest.raises(ValueError) as e:
        _ = CVTArchive(
            k=100, ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)], n_samples=99
        )
    assert (
        e.value.args[0]
        == "The number of samples (n_samples = 99) must be >= 1 and >= regions (k = 100)"
    )

    with pytest.raises(ValueError) as e:
        _ = CVTArchive(
            k=100,
            ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
            n_samples=100,
            centroids="example_centroid_file.npy",
        )
    assert (
        e.value.args[0]
        == "Error in CVTArchive.__init__() loading the centroids file example_centroid_file.npy."
    )

    with pytest.raises(ValueError) as e:
        loaded_centroids = np.load("default_cvt_centroids.npy")
        _ = CVTArchive(
            k=10,
            ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
            n_samples=100,
            centroids=loaded_centroids,
        )
    assert (
        e.value.args[0]
        == "The number of centroids 100 must be equal to the number of regions (k = 10)"
    )

    with pytest.raises(ValueError) as e:
        _ = CVTArchive(
            k=100,
            ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
            n_samples=100,
            samples="example_samples_file.npy",
        )
    assert (
        e.value.args[0]
        == "Error in CVTArchive.__init__() loading the samples file example_samples_file.npy."
    )

    with pytest.raises(ValueError) as e:
        loaded_samples = np.load("default_cvt_samples.npy")
        _ = CVTArchive(
            k=10,
            ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
            n_samples=1000,
            samples=loaded_samples,
        )
    assert (
        e.value.args[0]
        == "The number of samples 100 must be equal to the number of expected samples (n_samples = 1000)"
    )


def test_cvt_loading_works(setup_and_teardown_cvt):
    cvt = setup_and_teardown_cvt
    loaded_samples = np.load("default_cvt_samples.npy")
    loaded_centroids = np.load("default_cvt_centroids.npy")
    cvt_from_data = CVTArchive(
        k=100,
        ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
        n_samples=100,
        samples=loaded_samples,
        centroids=loaded_centroids,
    )
    cvt_from_file = CVTArchive(
        k=100,
        ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
        n_samples=100,
        samples="default_cvt_samples.npy",
        centroids="default_cvt_centroids.npy",
    )
    np.testing.assert_array_equal(loaded_samples, cvt.samples)
    np.testing.assert_array_equal(cvt_from_data.samples, loaded_samples)
    np.testing.assert_array_equal(cvt_from_file.samples, loaded_samples)
    np.testing.assert_array_equal(cvt_from_file.samples, cvt_from_data.samples)

    np.testing.assert_array_equal(loaded_centroids, cvt.centroids)
    np.testing.assert_array_equal(cvt_from_data.centroids, loaded_centroids)
    np.testing.assert_array_equal(cvt_from_file.centroids, loaded_centroids)
    np.testing.assert_array_equal(cvt_from_file.centroids, cvt_from_data.centroids)
