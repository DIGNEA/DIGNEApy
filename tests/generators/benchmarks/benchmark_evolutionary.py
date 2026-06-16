#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmark_evolutionary.py
@Time    :   2026/06/16 10:54:39
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import importlib

import pytest

from digneapy import DescriptorPipeline, GridArchive
from digneapy.archives import UnstructuredArchive
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import ES, Evolutionary
from digneapy.operators import (
    OPCX,
    UCX,
    BinarySelection,
    Elitist,
    Generational,
    GreedyReplacement,
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


@pytest.mark.parametrize("domain_cls, portfolio, feat_desc_n", DOMAIN_CONTEXT)
@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
@pytest.mark.parametrize("crossover", (UCX, OPCX))
@pytest.mark.parametrize("mutation", (UMut,))
@pytest.mark.parametrize("replacement", (Generational, GreedyReplacement, Elitist))
def test_evolutionary_generator_can_generate_instances(
    domain_cls,
    portfolio,
    feat_desc_n,
    descriptor,
    crossover,
    mutation,
    replacement,
):

    population_size = 100
    number_of_items = 100
    repetitions = 10
    generations = 100
    neighbours = 15
    threshold = 0.5
    selection = BinarySelection()

    descriptor_pipeline = DescriptorPipeline(descriptor)
    domain = domain_cls(number_of_items)
    generator = Evolutionary(
        pop_size=population_size,
        domain=domain,
        portfolio=portfolio,
        archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
        descriptor_pipe=descriptor_pipeline,
        repetitions=repetitions,
        generations=generations,
        crossover=crossover(),
        mutation=mutation(),
        selection=selection,
        replacement=replacement(),
    )

    result = generator()
    solution_set = result.instances
    # It could be empty
    assert isinstance(solution_set, UnstructuredArchive)
    if len(solution_set) != 0:
        desc_dimension = 0
        match descriptor:
            case "performance":
                desc_dimension = len(portfolio)
            case "instance":
                desc_dimension = len(solution_set[0])
            case "features":
                desc_dimension = feat_desc_n

        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.performance_bias >= 0 for s in solution_set)
        assert all(s.novelty >= 0 for s in solution_set)
        assert all(len(s.descriptor) == desc_dimension for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(
    "portfolio",
    [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ],
)
@pytest.mark.parametrize("descriptor", ("instance",))
def test_evolutionary_strategy(
    portfolio,
    descriptor,
):
    from digneapy.transformers.autoencoders import KPDecoder

    # It can be parametrised more, but it takes hours to run the test suit
    generations = 100
    generator_dimension = 2
    domain_dimension = 50
    k = 3
    descriptor_pipeline = DescriptorPipeline(descriptor, KPDecoder())
    domain = KnapsackDomain(number_of_items=domain_dimension)

    strategy = ES(
        generator_dimension=generator_dimension,
        domain=domain,
        portfolio=portfolio,
        lambda_=32,
        archives=[
            UnstructuredArchive(k=k, novelty_threshold=1.0),
            GridArchive(dimensions=(100,) * 2, ranges=[(-30.0, 10.0), (-30.0, 10.0)]),
        ],
        generations=generations,
        repetitions=1,
        descriptor_pipe=descriptor_pipeline,
    )
    builded_pipeline = strategy.descriptor_pipeline
    assert builded_pipeline._key == descriptor
    assert len(builded_pipeline._transformers) == 1
    result, archives = strategy()
    solution_set = result.instances
    # It could be empty
    for instance in solution_set:
        print(instance.descriptor.shape, len(instance.descriptor), instance.descriptor)
    assert isinstance(solution_set, GridArchive)
    assert id(solution_set) == id(archives[1])
    assert isinstance(archives[0], UnstructuredArchive)
    if len(solution_set) != 0:
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0 for s in solution_set)
        assert all(s.s >= 0 for s in solution_set)
        assert all(len(s.descriptor) == generator_dimension for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    # Test the creation of the evolution images
    log = strategy._logbook
    assert len(log) == strategy._generations
