#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmar_dominated.py
@Time    :   2026/06/16 11:35:52
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import warnings

import numpy as np
import pytest

from digneapy import DescriptorPipeline, Instance
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import Dominated
from digneapy.generators._utils import extract_solvers_name
from digneapy.operators import (
    OPCX,
    UCX,
    BinarySelection,
    UMut,
)
from digneapy.solvers import (
    best_fit,
    default_kp,
    first_fit,
    greedy,
    map_kp,
    miw_kp,
    mpw_kp,
    next_fit,
    nneighbour,
    worst_fit,
)

DOMAIN_CONTEXT = [
    (KnapsackDomain, [default_kp, map_kp, miw_kp, mpw_kp], 8),
    (BPPDomain, [best_fit, first_fit, worst_fit, next_fit], 10),
    (TSPDomain, [nneighbour, greedy], 11),
]


@pytest.fixture
def random_population():
    N_INSTANCES = 100
    DIMENSION = 101
    N_FEATURES = 10

    rng = np.random.default_rng()
    features = rng.uniform(low=0, high=1_000, size=(N_INSTANCES, N_FEATURES))
    performances = rng.random(size=(N_INSTANCES, 4), dtype=np.float64)
    variables = rng.integers(low=0, high=100, size=(N_INSTANCES, DIMENSION))
    instances = [
        Instance(
            variables=variables[i],
            features=features[i],
            descriptor=features[i],
            portfolio_scores=performances[i],
            fitness=performances[i][0],
            performance_bias=performances[i][0],
        )
        for i in range(N_INSTANCES)
    ]
    return instances


@pytest.mark.parametrize("domain_cls, portfolio, feat_desc_n", DOMAIN_CONTEXT)
@pytest.mark.parametrize("descriptor", ["features", "performance", "instance"])
@pytest.mark.parametrize("crossover", (UCX, OPCX))
def test_dominated_generator(
    domain_cls,
    portfolio,
    feat_desc_n,
    descriptor,
    crossover,
):
    pop_size = 32
    k = 3
    dimension = 100
    domain = domain_cls(dimension=dimension)
    generations = 10
    selection = BinarySelection()
    mutation = UMut()
    descriptor_pipeline = DescriptorPipeline(descriptor)
    deig = Dominated(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        repetitions=1,
        k=k,
        descriptor_pipe=descriptor_pipeline,
        crossover=crossover(),
        mutation=mutation,
        selection=selection,
    )
    portfolio_names = tuple(extract_solvers_name(portfolio))

    result = deig()
    assert len(result.instances) == pop_size
    instances = result.instances
    # They could be empty
    assert isinstance(instances, list)
    assert all(len(s) == len(instances[0]) for s in instances[1:])
    # Fitness feasibility is not guaranted
    try:
        assert all(s.fitness >= 0.0 for s in instances)
    except AssertionError:
        warnings.warn(
            f"Fitness feasibility check failed - "
            f"Domain: {domain_cls.__name__}, Descriptor: {descriptor}, N: {dimension}, Portfolio: {portfolio_names}, K: {k}",
            UserWarning,
            2,
        )
    # It means that fitness is sorted
    fitness = [s.fitness for s in instances]
    sorted_fitness = sorted(fitness, reverse=True)
    assert fitness == sorted_fitness
    # Dominated always returns something. Even though they could be unfeasible
    desc_dimension = 0
    match descriptor:
        case "performance":
            desc_dimension = len(portfolio)
        case "instance":
            desc_dimension = len(instances[0])
        case "features":
            desc_dimension = feat_desc_n

    assert all(len(s.descriptor) == desc_dimension for s in instances)
    assert all(len(s.portfolio_scores) == len(portfolio) for s in instances)
    # DNS do not guarantees the biased aspect of the instances
    # but all scores of the target must be positive
    p_scores = [s.portfolio_scores for s in instances]
    assert all(p_scores[i][0] >= 0.0 for i in range(len(p_scores)))
