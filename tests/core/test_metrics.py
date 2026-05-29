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
import polars as pl
import pytest
from deap import tools as deap_tools

from digneapy import Instance, Logbook, Statistics, qd_score, qd_score_auc


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


def test_logbook():
    log = Logbook()
    N_INSTANCES = 10
    DIMENSION = 10
    variables = np.random.default_rng().integers(
        0, 1_000, size=(N_INSTANCES, DIMENSION)
    )
    fitness = np.random.default_rng().random(N_INSTANCES)
    novelties = np.random.default_rng().random(N_INSTANCES)
    performances = np.random.default_rng().random(N_INSTANCES)
    instances = [
        Instance(
            variables=variables[i],
            fitness=fitness[i],
            p=performances[i],
            s=novelties[i],
        )
        for i in range(N_INSTANCES)
    ]
    assert len(log) == 0
    assert isinstance(log, deap_tools.Logbook)
    log.update(generation=0, population=instances)
    assert len(log) == 1
    df = log.to_df()
    assert len(df) == 1
    assert isinstance(df, pl.DataFrame)
    # Logbook doesn't accept negative generations
    with pytest.raises(ValueError):
        _ = log.update(-10, population=[])
