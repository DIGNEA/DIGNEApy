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

from digneapy import max_gap_target, runtime_score

INSTANCES = 10
SOLVERS = 10


def test_max_gap_target():
    scores = np.random.default_rng().integers(0, 10, size=(INSTANCES, SOLVERS))
    expected = scores[:, 0] - np.max(scores[:, 1:], axis=1)
    np.testing.assert_array_equal(max_gap_target(scores), expected)
    # Raises if the array is not a 2d matrix
    with pytest.raises(ValueError):
        not_valid = np.random.default_rng().integers(0, 10, size=INSTANCES)
        _ = max_gap_target(not_valid)


def test_runtime_target():
    scores = np.random.default_rng().integers(0, 10, size=(INSTANCES, SOLVERS))
    expected = np.min(scores[:, 1:], axis=1) - scores[:, 0]
    np.testing.assert_array_equal(runtime_score(scores), expected)
    # Raises if the array is not a 2d matrix
    with pytest.raises(ValueError):
        not_valid = np.random.default_rng().integers(0, 10, size=INSTANCES)
        _ = runtime_score(not_valid)
