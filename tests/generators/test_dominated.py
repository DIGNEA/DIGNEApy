#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_dominated.py
@Time    :   2026/05/22 12:55:05
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy.core import DescriptorPipeline, maximise_perf_gap_easy
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import DNSResult, Dominated, dominated_novelty_search
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

from .conftest import generate_random_population

DOMAIN_CONTEXT = [
    (KnapsackDomain, [default_kp, map_kp, miw_kp, mpw_kp], 8),
    (BPPDomain, [best_fit, first_fit, worst_fit, next_fit], 10),
    (TSPDomain, [nneighbour, greedy], 11),
]


@pytest.mark.parametrize("k", [3, 15, 30])
def test_dominated_novelty_search_different_ks(k):
    population = generate_random_population()
    descriptors = np.asarray([instance.descriptor for instance in population])
    performances = np.asarray([instance.performance_bias for instance in population])

    assert all(len(d) != 0 for d in descriptors)
    result: DNSResult = dominated_novelty_search(
        descriptors, performances=performances, k=k
    )

    assert result.descriptors.shape == descriptors.shape
    assert result.performances.shape == performances.shape
    assert result.competition_fitness.shape == performances.shape
    assert all(f != 0.0 for f in result.competition_fitness)
    # Check that the function can be sorted
    sorted_indices = np.argsort(-result.competition_fitness)
    sorted_comp_fitness = result.competition_fitness[sorted_indices]
    assert not np.array_equal(result.competition_fitness, sorted_comp_fitness)


def test_dominated_novelty_search_edge_cases():
    # If len(performacnes) != len(descriptors)
    with pytest.raises(ValueError) as mismatch_error:
        dominated_novelty_search(
            descriptors=np.arange(20), performances=np.empty(17), k=15
        )
    assert (
        "Array mismatch between performances and descriptors. len(performance) = 17 != 20 len(descriptors)"
        in str(mismatch_error.value)
    )

    # If len(pop) < k it should return zeros
    result = dominated_novelty_search(
        descriptors=np.arange(0), performances=np.empty(0), k=15
    )
    np.testing.assert_equal(result.competition_fitness, np.zeros(0))


@pytest.mark.parametrize("descriptor", ["features", "performance", "instance"])
def test_dominated_generator_attrs(
    descriptor,
):
    population_size = 32
    generations = 100
    k = 3
    number_of_items = 100
    repetitions = 1
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    domain = KnapsackDomain(number_of_items=number_of_items)
    descriptor_pipeline = DescriptorPipeline(descriptor)
    generator = Dominated(
        pop_size=population_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        repetitions=repetitions,
        k=k,
        descriptor_pipe=descriptor_pipeline,
    )
    assert generator.descriptor_pipeline is descriptor_pipeline
    assert generator.descriptor_pipeline._key == descriptor

    # Testing protected/private attributes
    assert generator._pop_size == population_size
    assert generator._repetitions == repetitions
    assert generator._generations == generations
    assert_equal(generator._population, [])
    assert isinstance(generator._domain, KnapsackDomain)
    assert_equal(generator._portfolio, portfolio)
    assert_equal(generator.cxrate, 0.5)
    assert_equal(generator.mutrate, 0.8)
    assert_equal(generator.phi, 0.85)
    assert isinstance(generator.crossover, Crossover)
    assert isinstance(generator.mutation, Mutation)
    assert isinstance(generator.selection, Selection)
    assert isinstance(generator.replacement, Replacement)
    assert generator._performance_fn is maximise_perf_gap_easy


@pytest.mark.parametrize("descriptor", ["features", "performance", "instance"])
def test_dominated_generator_can_generate_instances(descriptor):
    population_size = 32
    generations = 100
    k = 3
    number_of_items = 50
    repetitions = 1
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    domain = KnapsackDomain(number_of_items=number_of_items)
    descriptor_pipeline = DescriptorPipeline(descriptor)
    generator = Dominated(
        pop_size=population_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        repetitions=repetitions,
        k=k,
        descriptor_pipe=descriptor_pipeline,
    )

    result = generator()

    assert len(result.instances) == population_size
    instances = result.instances
    assert isinstance(instances, list)
    assert all(len(s) == len(instances[0]) for s in instances[1:])
    # Fitness feasibility is not guaranted
    try:
        assert all(s.fitness >= 0.0 for s in instances)
    except AssertionError:
        warnings.warn(
            f"Fitness feasibility check failed - Descriptor: {descriptor}.",
            UserWarning,
            2,
        )
    # It means that fitness is sorted
    fitness = [s.fitness for s in instances]
    sorted_fitness = sorted(fitness, reverse=True)
    assert fitness == sorted_fitness
    # Dominated always returns something. Even though they could be unfeasible
    assert all(len(s.descriptor) > 0 for s in instances)
    assert all(len(s.portfolio_scores) == len(portfolio) for s in instances)
    # DNS do not guarantees the biased aspect of the instances
    # but all scores of the target must be positive
    p_scores = [s.portfolio_scores for s in instances]
    assert all(p_scores[i][0] >= 0.0 for i in range(len(p_scores)))
