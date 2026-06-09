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

from digneapy import maximise_perf_gap_easy, maximise_runtime_gap


def test_max_gap_target():
    scores = np.asarray([
        [10, 5, 6, 7],
        [20, 5, 6, 7],
        [30, 5, 6, 7],
        [40, 5, 6, 7],
        [50, 5, 6, 7],
        [60, 5, 6, 7],
        [70, 5, 6, 7],
    ])
    expected = np.asarray(list(i + 3 for i in range(0, 70, 10)))
    np.testing.assert_array_equal(maximise_perf_gap_easy(scores), expected)


def test_max_gap_easy_raises_not2d():
    # Raises if the array is not a 2d matrix
    with pytest.raises(ValueError):
        not_valid = np.random.default_rng().uniform(0, 10, size=10)
        _ = maximise_perf_gap_easy(not_valid)


def test_maximise_perf_gap_easy_batch():
    # 3d scores (N_instances, N_solvers, N_repetitions)
    scores = np.asarray([
        [[10, 10, 10], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[20, 20, 20], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[30, 30, 30], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[40, 40, 40], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[50, 50, 50], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[60, 60, 60], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[70, 70, 70], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
    ])
    # Expected return of aggregate
    # aggregated = [
    #     [10, 6, 7, 8],
    #     [20, 6, 7, 8],
    #     [30, 6, 7, 8],
    #     [40, 6, 7, 8],
    #     [50, 6, 7, 8],
    #     [60, 6, 7, 8],
    #     [70, 6, 7, 8],
    # ]
    expected = np.asarray(list(i + 2 for i in range(0, 70, 10)))
    np.testing.assert_array_equal(maximise_perf_gap_easy(scores), expected)


def test_maximise_perf_gap_easy_batch_with_median():
    # 3d scores (N_instances, N_solvers, N_repetitions)
    scores = np.asarray([
        [[10, 10, 10], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[20, 20, 20], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[30, 30, 30], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[40, 40, 40], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[50, 50, 50], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[60, 60, 60], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        [[70, 70, 70], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
    ])
    # Expected return of aggregate
    # aggregated = [
    #     [10, 6, 7, 8],
    #     [20, 6, 7, 8],
    #     [30, 6, 7, 8],
    #     [40, 6, 7, 8],
    #     [50, 6, 7, 8],
    #     [60, 6, 7, 8],
    #     [70, 6, 7, 8],
    # ]
    callable_aggr = np.median
    expected = np.asarray(list(i + 2 for i in range(0, 70, 10)))
    np.testing.assert_array_equal(
        maximise_perf_gap_easy(scores, aggregate_fn=callable_aggr), expected
    )


def test_runtime_target():
    scores = np.asarray([
        [1.0, 5.0, 10.0, 20.0],
        [2.0, 5.0, 10.0, 20.0],
        [3.0, 5.0, 10.0, 20.0],
        [4.0, 5.0, 10.0, 20.0],
        [5.0, 5.0, 10.0, 20.0],
        [6.0, 5.0, 10.0, 20.0],
    ])
    expected = np.asarray([4, 3, 2, 1, 0, -1])
    np.testing.assert_array_equal(maximise_runtime_gap(scores), expected)


def test_runtime_target_batch():
    scores = np.asarray([
        [[1, 1, 1], [10, 10, 10], [6, 7, 8], [7, 8, 9]],
        [[2, 2, 2], [20, 20, 20], [6, 7, 8], [7, 8, 9]],
        [[3, 3, 3], [30, 30, 30], [6, 7, 8], [7, 8, 9]],
        [[4, 4, 4], [40, 40, 40], [6, 7, 8], [7, 8, 9]],
        [[5, 5, 5], [50, 50, 50], [6, 7, 8], [7, 8, 9]],
        [[6, 6, 6], [60, 60, 60], [6, 7, 8], [7, 8, 9]],
        [[7, 7, 7], [70, 70, 70], [6, 7, 8], [7, 8, 9]],
        [[8, 8, 8], [70, 70, 70], [6, 7, 8], [7, 8, 9]],
        [[9, 9, 9], [70, 70, 70], [6, 7, 8], [7, 8, 9]],
    ])
    # aggregate_fn is max
    expected = np.asarray([7, 6, 5, 4, 3, 2, 1, 0, -1])
    np.testing.assert_array_equal(maximise_runtime_gap(scores), expected)


def test_runtime_raises_not2d():
    # Raises if the array is not a 2d matrix
    with pytest.raises(ValueError):
        not_valid = np.random.default_rng().integers(0, 10, size=10)
        _ = maximise_runtime_gap(not_valid)
