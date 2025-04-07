#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_metrics.py
@Time    :   2025/04/07 11:32:35
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
from digneapy import qd_score, qd_score_auc, Statistics
import pytest


def test_qd_score():
    data = np.arange(10)
    expected = 45
    assert np.isclose(qd_score(data), expected)


def test_qd_score_auc():
    data = np.arange(10)
    batch = 10
    expected = 450
    assert np.isclose(qd_score_auc(data, batch_size=batch), expected)


def test_statistics_raises():
    with pytest.raises(ValueError):
        Statistics()(population=[])

    with pytest.raises(TypeError):
        population = np.zeros((10, 10))
        Statistics()(population=population)
