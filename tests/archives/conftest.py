#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   conftest.py
@Time    :   2026/06/10 11:46:07
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest

from digneapy import Instance


@pytest.fixture
def benchmark_data_10k():
    """Provides a set of standardized benchmark data.

    Includes:
    - The number of values (10k)
    - 10k random solutions in the range (-1,1) in each dim
    - 10k random objective values drawn from the standard normal distribution
    - 10k random measures in the range (-1,1) in each dim
    """
    rng = np.random.default_rng(42)
    n_vals = 10_000
    solution_batch = rng.uniform(-1, 1, (n_vals, 10))
    objective_batch = rng.standard_normal(n_vals)
    measures_batch = rng.uniform(-1, 1, (n_vals, 2))
    return n_vals, solution_batch, objective_batch, measures_batch


def default_incremental_population(
    n_instances: int = 10,
    dimension: int = 100,
    descriptor_dim: int = 4,
    portfolio_dim: int = 4,
):
    return [
        Instance(
            variables=list(range(dimension)),
            fitness=100.0,
            performance_bias=1.0,
            novelty=1.0,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        )
        for _ in range(n_instances)
    ]


ARCHIVE_NAMES = [
    "UnstructuredArchive",
    "GridArchive",
    "CVTArchive",
]
