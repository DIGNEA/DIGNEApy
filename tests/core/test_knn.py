#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_knn.py
@Time    :   2025/04/08 11:49:02
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import pytest
from digneapy._core import sparseness, sparseness_only_values
from digneapy import Instance
import numpy as np


@pytest.fixture
def random_population():
    rng = np.random.default_rng(seed=42)

    instances = [
        Instance(
            variables=rng.integers(low=0, high=100, size=10),
            descriptor=rng.integers(low=0, high=100, size=8),
        )
        for _ in range(10)
    ]
    return instances


def test_sparseness_updates(random_population):
    population = random_population[:5]
    archive = random_population[5:]
    assert all(np.isclose(instance.s, 0.0) for instance in population)

    result = sparseness(population, archive, k=2)
    assert len(result) == len(population)
    assert all(not np.isclose(instance.s, 0.0) for instance in population)
    assert all(np.isclose(instance.s, r) for instance, r in zip(population, result))


def test_sparseness_only_values(random_population):
    population = random_population[:5]
    archive = random_population[5:]
    assert all(np.isclose(instance.s, 0.0) for instance in population)

    result = sparseness_only_values(population, archive, k=2)
    assert len(result) == len(population)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(population),)
    # Does not updates the s
    assert all(np.isclose(instance.s, 0.0) for instance in population)
