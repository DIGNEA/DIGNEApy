#!/usr/bin/env python

"""Tests for `digneapy` package."""

import pytest
import copy
import numpy as np
from digneapy.novelty_search import Archive
from digneapy.core import Instance


@pytest.fixture
def default_archive():
    instances = [Instance(variables=list(range(d, d + 5))) for d in range(10)]
    return Archive(instances)


@pytest.fixture
def empty_archive():
    return Archive()


def test_empty_archive(empty_archive):
    assert 0 == len(empty_archive)


def test_archive_to_array(default_archive):
    np_archive = np.array(default_archive)
    assert len(np_archive) == len(default_archive)
    assert isinstance(np_archive, np.ndarray)


def test_iterable_default_archive(default_archive):
    descriptors = [list(range(d, d + 5)) for d in range(10)]
    assert all(a == b for a, b in zip(descriptors, default_archive))


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
