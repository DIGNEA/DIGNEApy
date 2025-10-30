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

import os

import numpy as np
import pytest

from digneapy import CVTArchive, Instance


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
    assert all(np.isclose(lbs_i, 0.0) for lbs_i in lbs)
    assert all(np.isclose(ubs_i, 100.0) for ubs_i in ubs)
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

    expected_str = f"CVArchive(dim=3,regions=100,centroids={cvt.centroids})"
    assert cvt.__str__() == expected_str
    assert cvt.__repr__() == expected_str


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


def test_cvt_bounds():
    cvt = CVTArchive(
        k=100, ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)], n_samples=100
    )
    assert cvt.dimensions == 3
    assert np.isclose(cvt.lower_i(0), 0.0)
    assert np.isclose(cvt.upper_i(0), 100.0)

    with pytest.raises(ValueError):
        cvt.lower_i(-1)

    with pytest.raises(ValueError):
        cvt.lower_i(100)

    with pytest.raises(ValueError):
        cvt.upper_i(-1)

    with pytest.raises(ValueError):
        cvt.upper_i(100)


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


def test_cvt_archive_append():
    cvt = CVTArchive(
        k=100, ranges=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)], n_samples=100
    )
    rng = np.random.default_rng(seed=42)
    instance = Instance(
        variables=rng.integers(low=1, high=10, size=10),
        fitness=100,
        s=100.0,
        descriptor=rng.integers(low=0, high=10000, size=cvt.dimensions),
    )
    assert len(cvt) == 0
    cvt.append(instance)
    assert len(cvt) == 1

    with pytest.raises(TypeError):
        cvt.append(None)


def test_cvt_archive_extend():
    dimension = 10
    cvt = CVTArchive(
        k=100, ranges=[(0.0, 10000.0) for _ in range(dimension)], n_samples=100
    )
    rng = np.random.default_rng(seed=42)
    instances = [
        Instance(
            variables=rng.integers(low=1, high=10, size=10),
            fitness=rng.integers(low=1, high=10000),
            s=rng.random(),
            descriptor=rng.integers(low=0, high=10000, size=cvt.dimensions),
        )
        for _ in range(10)
    ]
    assert len(cvt) == 0
    cvt.extend(instances)
    assert len(cvt) > 0
    with pytest.raises(TypeError):
        cvt.extend([None])


def test_cvt_archive_remove():
    dimension = 10
    cvt = CVTArchive(
        k=100, ranges=[(0.0, 10000.0) for _ in range(dimension)], n_samples=100
    )
    rng = np.random.default_rng(seed=42)
    instances = [
        Instance(
            variables=rng.integers(low=1, high=10, size=10),
            fitness=rng.integers(low=1, high=10000),
            s=rng.random(),
            descriptor=rng.integers(low=0, high=10000, size=cvt.dimensions),
        )
        for _ in range(10)
    ]
    assert len(cvt) == 0
    cvt.extend(instances)
    current_len = len(cvt)
    assert current_len > 0
    descriptors_to_remove = np.asarray(
        [instance.descriptor for instance in instances[:5]]
    )
    cvt.remove(descriptors_to_remove)
    assert len(cvt) < current_len

    with pytest.raises(ValueError):
        cvt.remove([None])
