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

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy import (
    DescriptorPipeline,
    Domain,
    Instance,
    maximise_perf_gap_easy,
)
from digneapy.archives import UnstructuredArchive
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import Evolutionary
from digneapy.operators import (
    Crossover,
    Mutation,
    Replacement,
    Selection,
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


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
@pytest.mark.parametrize("domain_cls, portfolio, n_features", DOMAIN_CONTEXT)
def test_evoluationary_generator_attrs(descriptor, domain_cls, portfolio, n_features):
    population_size = 100
    number_of_items = 100
    repetitions = 10
    generations = 100
    neighbours = 15
    threshold = 0.5

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
    )

    assert generator.descriptor_pipeline is descriptor_pipeline
    assert generator.descriptor_pipeline._key == descriptor

    # Testing protected/private attributes
    assert generator._pop_size == population_size
    assert generator._repetitions == repetitions
    assert generator._generations == generations
    assert_equal(generator._population, [])
    assert isinstance(generator._domain, Domain)
    assert_equal(generator._portfolio, portfolio)
    assert_equal(generator.cxrate, 0.5)
    assert_equal(generator.mutrate, 0.8)
    assert_equal(generator.phi, 0.85)
    assert isinstance(generator.crossover, Crossover)
    assert isinstance(generator.mutation, Mutation)
    assert isinstance(generator.selection, Selection)
    assert isinstance(generator.replacement, Replacement)
    assert generator._performance_fn is maximise_perf_gap_easy


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_evolutionary_generator_raises_wrong_args(descriptor):
    population_size = 100
    repetitions = 10
    generations = 100
    neighbours = 15
    threshold = 0.5
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    descriptor_pipeline = DescriptorPipeline(descriptor)
    # Raises because the pop_size is negative
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=-population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
        )
    # Raises because the generations is negative
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=-generations,
        )
    # Raises because the repetitions is negative
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=-repetitions,
            generations=generations,
        )
    # Raises because Domain is not a valid domain
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=None,
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
        )
    # Raises because the portfolio is empty
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=tuple(),
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
        )
    # Raises because the phi is negative
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
            phi=-1.0,
        )
    # Raises because the cxrate is negative
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
            cxrate=-1.0,
        )
    # Raises because the mutrate is negative
    with pytest.raises(ValueError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
            mutrate=-1.0,
        )

    # Raises because the arhive is not a valid Archive
    with pytest.raises(TypeError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=[],
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
        )
    # Raises because the solution_set is not a valid Archive
    with pytest.raises(TypeError):
        _ = Evolutionary(
            pop_size=population_size,
            domain=KnapsackDomain(),
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            solution_set=[],
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
        )


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_evolutionary_generator_can_generate_instances(
    descriptor,
):

    population_size = 10
    number_of_items = 25
    repetitions = 1
    generations = 64
    neighbours = 3
    threshold = 0.5
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    domain = KnapsackDomain(number_of_items=number_of_items)

    descriptor_pipeline = DescriptorPipeline(descriptor)
    generator = Evolutionary(
        pop_size=population_size,
        domain=domain,
        portfolio=portfolio,
        archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
        descriptor_pipe=descriptor_pipeline,
        repetitions=repetitions,
        generations=generations,
    )

    result = generator()
    archive = result.instances
    # The archive must not be empty
    # this archive allows unfeasible solutions
    assert isinstance(archive, UnstructuredArchive)
    assert len(archive) > 0
    assert all(isinstance(x, Instance) for x in archive)
    assert all(len(s.descriptor) > 0 for s in archive)
    assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)

    # Logbook can be plotted
    logbook = generator.log
    filename = Path("evolutionary_generator_logbook.png")
    ea_generator_evolution_plot(logbook, filename)
    assert filename.exists()
    filename.unlink()


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_evolutionary_generator_can_generate_instances_with_solution_set(
    descriptor,
):
    population_size = 10
    number_of_items = 25
    repetitions = 1
    generations = 100
    neighbours = 3
    threshold = 0.5
    portfolio = [mpw_kp, map_kp, miw_kp, default_kp]
    descriptor_pipeline = DescriptorPipeline(descriptor)
    domain = KnapsackDomain(number_of_items=number_of_items)
    generator = Evolutionary(
        pop_size=population_size,
        domain=domain,
        portfolio=portfolio,
        archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
        solution_set=UnstructuredArchive(k=1, novelty_threshold=0.0001),
        descriptor_pipe=descriptor_pipeline,
        repetitions=repetitions,
        generations=generations,
    )

    result = generator()
    # Logbook can be plotted
    logbook = generator.log
    filename = Path("evolutionary_generator_logbook.png")
    ea_generator_evolution_plot(logbook, filename)
    assert filename.exists()
    filename.unlink()

    solution_set = result.instances
    # The solution_set can  be empty
    # this archive does NOT allow unfeasible solutions
    assert isinstance(solution_set, UnstructuredArchive)
    assert len(solution_set) > 0

    for instance in solution_set:
        assert isinstance(instance, Instance)
        assert instance.fitness >= 0.0
        assert instance.performance_bias >= 0.0
        assert instance.novelty >= 0.0
        assert len(instance.descriptor) >= 0
        assert len(instance.portfolio_scores) == len(portfolio)
        assert instance.portfolio_scores[0] >= np.max(instance.portfolio_scores)


# @pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
# @pytest.mark.parametrize(
#     "portfolio",
#     [
#         [default_kp, map_kp, miw_kp, mpw_kp],
#         [map_kp, default_kp, miw_kp, mpw_kp],
#         [miw_kp, default_kp, map_kp, mpw_kp],
#         [mpw_kp, default_kp, map_kp, miw_kp],
#     ],
# )
# @pytest.mark.parametrize("descriptor", ("instance",))
# def test_evolutionary_strategy(
#     portfolio,
#     descriptor,
# ):
#     from digneapy.transformers.autoencoders import KPDecoder

#     # It can be parametrised more, but it takes hours to run the test suit
#     generations = 100
#     generator_dimension = 2
#     domain_dimension = 50
#     k = 3
#     descriptor_pipeline = DescriptorPipeline(descriptor, KPDecoder())
#     domain = KnapsackDomain(number_of_items=domain_dimension)

#     strategy = ES(
#         generator_dimension=generator_dimension,
#         domain=domain,
#         portfolio=portfolio,
#         lambda_=32,
#         archives=[
#             UnstructuredArchive(k=k, novelty_threshold=1.0),
#             GridArchive(dimensions=(100,) * 2, ranges=[(-30.0, 10.0), (-30.0, 10.0)]),
#         ],
#         generations=generations,
#         repetitions=1,
#         descriptor_pipe=descriptor_pipeline,
#     )
#     builded_pipeline = strategy.descriptor_pipeline
#     assert builded_pipeline._key == descriptor
#     assert len(builded_pipeline._transformers) == 1
#     result, archives = strategy()
#     solution_set = result.instances
#     # It could be empty
#     for instance in solution_set:
#         print(instance.descriptor.shape, len(instance.descriptor), instance.descriptor)
#     assert isinstance(solution_set, GridArchive)
#     assert id(solution_set) == id(archives[1])
#     assert isinstance(archives[0], UnstructuredArchive)
#     if len(solution_set) != 0:
#         assert all(s.fitness >= 0.0 for s in solution_set)
#         assert all(s.performance_bias >= 0 for s in solution_set)
#         assert all(s.novelty >= 0 for s in solution_set)
#         assert all(len(s.descriptor) == generator_dimension for s in solution_set)
#         assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
#         p_scores = [s.portfolio_scores for s in solution_set]
#         assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

#     # Test the creation of the evolution images
#     log = strategy._logbook
#     assert len(log) == strategy._generations
