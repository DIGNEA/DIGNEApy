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

from digneapy.core import Instance, Logbook, Statistics, qd_score, qd_score_auc


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
    with pytest.raises(RuntimeError):
        Statistics()(instances=[])

    with pytest.raises(RuntimeError):
        instances = np.zeros((10, 10))
        Statistics()(instances=instances)


def test_logbook_creation():
    logbook = Logbook()
    assert len(logbook) == 0
    df = logbook.to_df()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 0


def test_logbook_several_generations():
    dimension = 10
    n_generations = 10
    n_instances = 10
    rng = np.random.default_rng()

    log = Logbook()
    assert len(log) == 0

    for i in range(n_generations):
        variables = rng.integers(0, 1_000, size=(n_instances, dimension))
        fitness = rng.uniform(low=0, high=100, size=n_instances)
        novelties = rng.uniform(low=0, high=100, size=n_instances)
        performances = rng.uniform(low=0, high=100, size=n_instances)
        instances = [
            Instance(
                variables=variables[i],
                fitness=fitness[i],
                performance_bias=performances[i],
                novelty=novelties[i],
            )
            for i in range(n_instances)
        ]
        log.update(generation=i, instances=instances, feedback=False)
        assert len(log) == i + 1

    assert "novelty" in log.chapters
    assert "performance_bias" in log.chapters
    assert "fitness" in log.chapters

    assert len(log.chapters["novelty"]) == n_generations
    assert len(log.chapters["performance_bias"]) == n_generations
    assert len(log.chapters["fitness"]) == n_generations


def test_logbook_to_df():
    dimension = 10
    n_generations = 10
    n_instances = 10
    rng = np.random.default_rng()

    log = Logbook()
    assert len(log) == 0

    for i in range(n_generations):
        variables = rng.integers(0, 1_000, size=(n_instances, dimension))
        fitness = rng.uniform(low=0, high=100, size=n_instances)
        novelties = rng.uniform(low=0, high=100, size=n_instances)
        performances = rng.uniform(low=0, high=100, size=n_instances)
        instances = [
            Instance(
                variables=variables[i],
                fitness=fitness[i],
                performance_bias=performances[i],
                novelty=novelties[i],
            )
            for i in range(n_instances)
        ]
        log.update(generation=i, instances=instances, feedback=False)
        assert len(log) == i + 1

    df = log.to_df()
    assert len(df) == n_generations
    assert isinstance(df, pl.DataFrame)


def test_logbook_raises_negative_generations():
    log = Logbook()
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4
    instances = [
        Instance(
            variables=list(range(dimension)),
            fitness=100.0,
            performance_bias=1.0,
            novelty=1.0,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        )
    ]
    # Logbook doesn't accept negative generations
    with pytest.raises(ValueError):
        _ = log.update(
            -10,
            instances,
        )


def test_logbook_raises_empty_population():
    log = Logbook()
    # Logbook doesn't accept negative generations
    with pytest.raises(RuntimeError):
        _ = log.update(1, instances=[])


def test_logbook_raises_not_all_instances():
    log = Logbook()
    dimension = 10
    descriptor_dim = 4
    portfolio_dim = 4
    instances = [
        Instance(
            variables=list(range(dimension)),
            fitness=100.0,
            performance_bias=1.0,
            novelty=1.0,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        ),
        list(range(dimension)),
    ]
    # Logbook doesn't accept negative generations
    with pytest.raises(RuntimeError):
        _ = log.update(
            1,
            instances,
        )
