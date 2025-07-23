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


@pytest.fixture
def default_archive():
    instances = [Instance(variables=list(range(d, d + 5))) for d in range(10)]
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
    instance = Instance(variables=list(range(100)), s=1.0)
    empty_archive.append(instance)
    assert 1 == len(empty_archive)
    assert [instance] == empty_archive.instances


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
    rng = np.random.default_rng(seed=42)
    instances = [Instance([], fitness=0.0, p=0.0, s=rng.random()) for _ in range(10)]

    expected = len(list(filter(lambda i: i.s >= empty_archive.threshold, instances)))
    empty_archive.extend(instances)
    assert len(empty_archive) == expected
