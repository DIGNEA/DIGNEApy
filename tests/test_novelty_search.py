#!/usr/bin/env python

"""Tests for `digneapy` package."""

import pytest
import copy
import numpy as np
from digneapy.novelty_search import Archive


@pytest.fixture
def default_archive():
    descriptors = [list(range(d, d + 5)) for d in range(10)]
    return Archive(descriptors)


@pytest.fixture
def empty_archive():
    return Archive()


def test_empty_archive(empty_archive):
    assert 0 == len(empty_archive)


def test_archive_to_array(default_archive):
    np_archive = np.array(default_archive)
    assert len(np_archive) == len(default_archive)
    assert type(np_archive) == type(np.zeros(1))


def test_iterable_default_archive(default_archive):
    descriptors = [list(range(d, d + 5)) for d in range(10)]
    assert all(a == b for a, b in zip(descriptors, default_archive))


def test_not_equal_archives(default_archive, empty_archive):
    assert default_archive != empty_archive


def test_equal_archives(default_archive):
    a1 = copy.copy(default_archive)
    assert default_archive == a1


def test_append_descriptor(empty_archive):
    assert 0 == len(empty_archive)
    d = list(range(100))
    empty_archive.append(d)
    assert 1 == len(empty_archive)
    assert [d] == empty_archive.descriptors


def test_extend_iterable(empty_archive, default_archive):
    assert 0 == len(empty_archive)
    d = default_archive.descriptors
    empty_archive.extend(d)
    assert len(empty_archive) == len(default_archive)
    assert empty_archive == default_archive


def test_bool_on_empty_archive(empty_archive):
    assert not empty_archive


def test_bool_on_default_archive(default_archive):
    assert default_archive
