#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_evolutionary.py
@Time    :   2026/04/21 14:03:12
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from digneapy import DescriptorPipeline, Domain, GridArchive, max_gap_target
from digneapy.archives import UnstructuredArchive
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import ES, Evolutionary
from digneapy.operators import (
    OPCX,
    UCX,
    BinarySelection,
    Crossover,
    Elitist,
    Generational,
    GreedyReplacement,
    Mutation,
    Replacement,
    Selection,
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
from digneapy.visualize import ea_generator_evolution_plot

DOMAIN_CONTEXT = [
    (KnapsackDomain, [default_kp, map_kp, miw_kp, mpw_kp], 8),
    (BPPDomain, [best_fit, first_fit, worst_fit, next_fit], 10),
    (TSPDomain, [nneighbour, greedy], 11),
]


def get_expected_descriptor_size(domain_cls: Domain, dimension: int):
    if domain_cls == TSPDomain:
        return dimension * 2
    if domain_cls == BPPDomain:
        return dimension + 1
    if domain_cls == KnapsackDomain:
        return (dimension * 2) + 1


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_default_generator(descriptor):
    descriptor_pipeline = DescriptorPipeline(descriptor)
    eig = Evolutionary(
        pop_size=100,
        domain=KnapsackDomain(),
        portfolio=[default_kp],
        archive=UnstructuredArchive(k=15, threshold=1.0),
        descriptor_pipe=descriptor_pipeline,
    )
    assert eig._pop_size == 100
    assert eig._generations == 1000
    builded_pipeline = eig.descriptor_pipeline
    assert builded_pipeline._key == descriptor
    assert len(builded_pipeline._transformers) == 0
    assert isinstance(eig._domain, KnapsackDomain)
    assert eig._portfolio == tuple([default_kp])
    assert eig._repetitions == 1
    assert np.isclose(eig.cxrate, 0.5)
    assert np.isclose(eig.mutrate, 0.8)
    assert isinstance(eig.crossover, Crossover)
    assert isinstance(eig.mutation, Mutation)
    assert isinstance(eig.selection, Selection)
    assert isinstance(eig.replacement, Replacement)
    assert np.isclose(eig.phi, 0.85)
    assert eig._performance_fn is not None
    assert eig._performance_fn == max_gap_target

    with pytest.raises(ValueError) as e:
        _ = Evolutionary(
            pop_size=100,
            domain=None,
            portfolio=[default_kp],
            archive=UnstructuredArchive(k=15, threshold=1.0),
            descriptor_pipe=descriptor_pipeline,
        )
    assert e.value.args[0] == "BaseGenerator: Invalid domain. Got None."

    with pytest.raises(ValueError) as e:
        _ = Evolutionary(
            pop_size=100,
            domain=KnapsackDomain(),
            portfolio=tuple(),
            archive=UnstructuredArchive(k=15, threshold=1.0),
            descriptor_pipe=descriptor_pipeline,
        )
    assert (
        e.value.args[0]
        == "BaseGenerator: the portfolio is empty or contains invalid solvers. ()"
    )

    with pytest.raises(ValueError) as e:
        _ = Evolutionary(
            pop_size=100,
            domain=KnapsackDomain(),
            portfolio=[default_kp],
            archive=UnstructuredArchive(k=15, threshold=1.0),
            phi=-1.0,
        )
    assert (
        e.value.args[0]
        == "Invalid parameters. cxrate, mutrate and phi must be a float number in the range [0.0-1.0]. Got: cxrate=0.5, mutrate=0.8, phi=-1.0."
    )
    with pytest.raises(ValueError) as e:
        _ = Evolutionary(
            pop_size=100,
            domain=KnapsackDomain(),
            portfolio=[default_kp],
            archive=UnstructuredArchive(k=15, threshold=1.0),
            phi="hello",
        )
    assert (
        e.value.args[0]
        == "Invalid parameters. cxrate, mutrate and phi must be a float number in the range [0.0-1.0]. Got: cxrate=0.5, mutrate=0.8, phi=hello."
    )

    with pytest.raises(TypeError) as e:
        _ = Evolutionary(
            domain=KnapsackDomain(),
            portfolio=[default_kp],
            pop_size=100,
            archive=tuple(),
        )
    assert "archive must be a subclass of Archive" in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = Evolutionary(
            domain=KnapsackDomain(),
            portfolio=[default_kp],
            pop_size=100,
            archive=KnapsackDomain(),
        )

    assert "archive must be a subclass of Archive" in str(e.value)


@pytest.mark.parametrize("domain_cls, portfolio, feat_desc_n", DOMAIN_CONTEXT)
@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
@pytest.mark.parametrize("crossover", (UCX, OPCX))
@pytest.mark.parametrize("mutation", (UMut,))
@pytest.mark.parametrize("replacement", (Generational, GreedyReplacement, Elitist))
def test_evolutionary_generator(
    domain_cls,
    portfolio,
    feat_desc_n,
    descriptor,
    crossover,
    mutation,
    replacement,
):
    # It can be parametrised more, but it takes hours to run the test suit
    generations = 10
    k = 3
    dimension = 100

    descriptor_pipeline = DescriptorPipeline(descriptor)
    domain = domain_cls(dimension=dimension)

    selection = BinarySelection()
    eig = Evolutionary(
        pop_size=32,
        generations=generations,
        domain=domain,
        archive=UnstructuredArchive(k=k, threshold=3),
        solution_set=UnstructuredArchive(k=k, threshold=3),
        portfolio=portfolio,
        repetitions=1,
        descriptor_pipe=descriptor_pipeline,
        crossover=crossover(),
        mutation=mutation(),
        selection=selection,
        replacement=replacement(),
    )
    builded_pipeline = eig.descriptor_pipeline
    assert builded_pipeline._key == descriptor
    assert len(builded_pipeline._transformers) == 0
    result = eig()
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
        assert all(s.p >= 0 for s in solution_set)
        assert all(s.s >= 0 for s in solution_set)
        assert all(
            len(s.descriptor) == desc_dimension for s in solution_set
        )  # Based on descriptor
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    # Test the creation of the evolution images
    log = eig._logbook
    assert len(log) == eig._generations
    filename = Path("test_evolution.png")
    ea_generator_evolution_plot(log, filename=filename)
    assert filename.exists()
    filename.unlink()


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
    domain = KnapsackDomain(dimension=domain_dimension)

    strategy = ES(
        generator_dimension=generator_dimension,
        domain=domain,
        portfolio=portfolio,
        lambda_=32,
        archives=[
            UnstructuredArchive(k=k, threshold=1.0),
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
