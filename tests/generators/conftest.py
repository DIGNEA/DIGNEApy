#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   conftest.py
@Time    :   2026/06/16 11:41:46
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence

import numpy as np

from digneapy.core import Instance


def generate_random_population(
    n_instances: int = 100,
    dimension: int = 101,
    descriptor_dim: int = 10,
    portfolio_dim: int = 4,
) -> Sequence[Instance]:

    rng = np.random.default_rng()
    descriptor = rng.uniform(low=0, high=1_000, size=(n_instances, descriptor_dim))
    performances = rng.random(size=(n_instances, portfolio_dim), dtype=np.float64)
    variables = rng.integers(low=0, high=100, size=(n_instances, dimension))
    instances = [
        Instance(
            variables=variables[i],
            descriptor=descriptor[i],
            portfolio_scores=performances[i],
            fitness=performances[i][0],
            performance_bias=performances[i][0],
        )
        for i in range(n_instances)
    ]
    return instances
