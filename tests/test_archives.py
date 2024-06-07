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
import pytest
import numpy as np
from digneapy.core import Instance
from digneapy.archives import Archive


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
    with pytest.raises(Exception):
        empty_archive.append(d)


def test_extend_iterable(empty_archive, default_archive):
    assert 0 == len(empty_archive)
    d = default_archive.instances
    empty_archive.extend(d)
    assert len(empty_archive) == len(default_archive)
    assert empty_archive == default_archive


def test_bool_on_empty_archive(empty_archive):
    assert not empty_archive


def test_bool_on_default_archive(default_archive):
    assert default_archive


def test_archive_magic(default_archive):
    assert default_archive.__str__() == f"Archive(threshold=0.0,data=(|10|))"
    duplicated = copy.deepcopy(default_archive)
    assert hash(duplicated) == hash(default_archive)


def test_archive_access(default_archive):
    assert len(default_archive) == 10
    assert type(default_archive[0]) == Instance
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
    filter_fn = lambda x: x.s >= empty_archive.threshold
    expected = len(list(filter(filter_fn, instances)))
    empty_archive.extend(instances, filter_fn=None)
    assert len(empty_archive) == expected


def test_archive_extend_with_s_and_p(empty_archive):
    new_threshold = 1.0
    empty_archive.threshold = new_threshold
    instances = [
        Instance([], fitness=0.0, p=np.random.randint(0, 100), s=np.random.random())
        for _ in range(10)
    ]
    filter_fn = lambda x: x.s >= empty_archive.threshold and x.p >= 50.0
    expected = len(list(filter(filter_fn, instances)))
    empty_archive.extend(instances, filter_fn=filter_fn)
    assert len(empty_archive) == expected


def test_archive_extend_with_s_p_and_fitness(empty_archive):
    new_threshold = 1.0
    empty_archive.threshold = new_threshold
    instances = [
        Instance(
            [],
            fitness=np.random.random(),
            p=np.random.randint(0, 100),
            s=np.random.random(),
        )
        for _ in range(10)
    ]
    filter_fn = (
        lambda x: x.s >= empty_archive.threshold and x.p >= 50.0 and x.fitness >= 0.5
    )
    expected = len(list(filter(filter_fn, instances)))
    empty_archive.extend(instances, filter_fn=filter_fn)
    assert len(empty_archive) == expected
