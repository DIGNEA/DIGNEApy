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

from digneapy import Instance


def default_incremental_population(
    n_instances: int = 10,
    dimension: int = 100,
    descriptor_dim: int = 4,
    portfolio_dim: int = 4,
    fitness: int = 100,
    novelty: float = 1.0,
    performance_bias: float = 1.0,
):
    return [
        Instance(
            variables=list(range(dimension)),
            fitness=fitness,
            performance_bias=performance_bias,
            novelty=novelty,
            descriptor=tuple(range(descriptor_dim)),
            portfolio_scores=tuple(range(portfolio_dim)),
        )
        for _ in range(n_instances)
    ]


def population_with_custom_descriptors(
    descriptors: np.ndarray,
    n_instances: int = 10,
    dimension: int = 100,
    portfolio_dim: int = 4,
    fitness: int = 100,
    novelty: float = 1.0,
    performance_bias: float = 1.0,
):

    return [
        Instance(
            variables=list(range(dimension)),
            fitness=fitness,
            performance_bias=performance_bias,
            novelty=novelty,
            descriptor=descriptors[i],
            portfolio_scores=tuple(range(portfolio_dim)),
        )
        for i in range(n_instances)
    ]


ARCHIVE_NAMES = [
    "UnstructuredArchive",
    "GridArchive",
    "CVTArchive",
]
