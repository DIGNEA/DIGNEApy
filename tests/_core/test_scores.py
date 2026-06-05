#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_scores.py
@Time    :   2026/05/12 14:36:38
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy import max_gap_target, runtime_score


def test_max_gap_target():
    instances = 10
    scores = np.arange(0, 10) + np.zeros((instances, 1))
    scores[:, 0] = 100.0
    expected = 100.0 - (instances - 1)
    assert_equal(max_gap_target(scores), expected)
    # Raises if the array is not a 2d matrix
    with pytest.raises(ValueError):
        not_valid = np.random.default_rng().integers(0, 10, size=instances)
        _ = max_gap_target(not_valid)


def test_max_gap_target_raises():
    # Raises if the array is not a 2d matrix
    with pytest.raises(ValueError):
        not_valid = np.random.default_rng().integers(0, 10, size=10)
        _ = max_gap_target(not_valid)


def test_runtime_target():
    instances = 10
    scores = np.arange(2, 10) + np.zeros((instances, 1))
    scores[:, 0] = 0.5
    expected = 3.0 - 0.5
    assert_equal(runtime_score(scores), expected)


def test_runtime_score_raises():
    # Raises if the array is not a 2d matrix
    with pytest.raises(ValueError):
        not_valid = np.random.default_rng().integers(0, 10, size=10)
        _ = runtime_score(not_valid)
