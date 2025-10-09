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

from digneapy import Archive, Instance

N_INSTANCES = 100
DIMENSION = 101
N_FEATURES = 10


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
    return Archive(threshold=0.0, instances=instances)


@pytest.fixture
def empty_archive():
    return Archive(threshold=0.0)


def test_empty_archive(empty_archive):
    assert 0 == len(empty_archive)
    assert np.isclose(empty_archive.threshold, 0.0)
    with pytest.raises(TypeError):
        empty_archive.threshold = "HELLO"


def test_archive_to_array(default_archive):
    np_archive = np.array(default_archive)
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
    assert default_archive.__str__() == f"Archive(threshold=0.0,data=(|{N_INSTANCES}|))"
    duplicated = copy.deepcopy(default_archive)
    assert hash(duplicated) == hash(default_archive)


def test_Archive_can_be_indexed(default_archive):
    assert len(default_archive) == N_INSTANCES
    assert isinstance(default_archive[0], Instance)
    assert len(default_archive[:2]) == 2
    with pytest.raises(IndexError):
        default_archive[N_INSTANCES + 1]


def test_archive_repr(default_archive):
    assert f"Archive(threshold=0.0,data=(|{N_INSTANCES}|))" == format(
        repr(default_archive),
    )


@pytest.mark.parametrize("threshold", [0.01, 1.0, 5.0, 10.0])
def test_archive_extend(threshold, empty_archive):
    empty_archive.threshold = threshold
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
    empty_archive.extend(instances)
    assert len(empty_archive) == expected
