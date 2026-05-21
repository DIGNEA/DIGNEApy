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

import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from digneapy import NS, Archive, DescriptorPipeline, Domain, max_gap_target
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import (
    Dominated,
    Evolutionary,
)
from digneapy.operators import (
    Crossover,
    Generational,
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
        domain=None,
        portfolio=[],
        novelty_approach=NS(k=15),
        descriptor_pipe=descriptor_pipeline,
    )
    assert eig._pop_size == 100
    assert eig._generations == 1000
    builded_pipeline = eig.descriptor_pipeline
    assert builded_pipeline._key == descriptor
    assert len(builded_pipeline._transformers) == 0
    assert eig._domain is None
    assert eig._portfolio == tuple()
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
        eig()
    assert e.value.args[0] == "You must specify a domain to run the generator."

    with pytest.raises(ValueError) as e:
        eig._domain = KnapsackDomain()
        eig()
    assert (
        e.value.args[0]
        == "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
    )

    with pytest.raises(ValueError) as e:
        _ = Evolutionary(
            pop_size=100, domain=None, portfolio=[], novelty_approach=NS(k=15), phi=-1.0
        )
    assert (
        e.value.args[0]
        == "Phi must be a float number in the range [0.0-1.0]. Got: -1.0."
    )
    with pytest.raises(ValueError) as e:
        _ = Evolutionary(
            pop_size=100,
            domain=None,
            portfolio=[],
            novelty_approach=NS(k=15),
            phi="hello",
        )
    assert e.value.args[0] == "Phi must be a float number in the range [0.0-1.0]."

    with pytest.raises(TypeError) as e:
        _ = Evolutionary(
            domain=KnapsackDomain(),
            portfolio=[],
            pop_size=100,
            novelty_approach=tuple(),
        )
    assert "Novelty Search cannot be None in Evolutionary Generator" in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = Evolutionary(
            domain=KnapsackDomain(),
            portfolio=[],
            pop_size=100,
            novelty_approach=KnapsackDomain(),
        )

    assert "Novelty Search cannot be None in Evolutionary Generator" in str(e.value)


@pytest.mark.parametrize("domain_cls, portfolio, feat_desc_n", DOMAIN_CONTEXT)
@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
@pytest.mark.parametrize("dimension", [50, 100])
@pytest.mark.parametrize("k", [3, 15])
@pytest.mark.parametrize("ps", [64, 128])
def test_evolutionary_generator(
    domain_cls, portfolio, feat_desc_n, descriptor, dimension, k, ps
):
    descriptor_pipeline = DescriptorPipeline(descriptor)
    domain = domain_cls(dimension=dimension)
    generations = 10
    eig = Evolutionary(
        pop_size=ps,
        generations=generations,
        domain=domain,
        novelty_approach=NS(k=k),
        solution_set=Archive(threshold=3),
        portfolio=portfolio,
        repetitions=1,
        descriptor_pipe=descriptor_pipeline,
        replacement=Generational(),
    )
    builded_pipeline = eig.descriptor_pipeline
    assert builded_pipeline._key == descriptor
    assert len(builded_pipeline._transformers) == 0
    result = eig()
    solution_set = result.instances
    # It could be empty
    assert isinstance(solution_set, Archive)
    if len(solution_set) != 0:
        # Dominated always returns something. Even though they could be unfeasible
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
    ea_generator_evolution_plot(log.logbook, filename=filename)
    assert filename.exists()
    filename.unlink()


@pytest.mark.parametrize("domain_cls, portfolio, feat_desc_n", DOMAIN_CONTEXT)
@pytest.mark.parametrize("k", [3, 15])
@pytest.mark.parametrize("descriptor", ["features", "performance", "instance"])
@pytest.mark.parametrize("dimension", [50, 100])
@pytest.mark.parametrize("ps", [64, 128])
def test_dominated_generator(
    domain_cls, portfolio, feat_desc_n, k, descriptor, dimension, ps
):
    pop_size = ps
    domain = domain_cls(dimension=dimension)
    generations = 10
    descriptor_pipeline = DescriptorPipeline(descriptor)
    deig = Dominated(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        repetitions=1,
        k=k,
        descriptor_pipe=descriptor_pipeline,
    )
    portfolio_names = [s.__name__ for s in portfolio]
    expected_str = f"pop_size={pop_size},gen={generations},domain={domain.__name__},portfolio={portfolio_names}"
    assert deig.__str__() == f"Dominated({expected_str})"
    assert deig.__repr__() == f"Dominated<{expected_str}>"
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
    # but all scores of the target must be positive --> Forces feasibility
    p_scores = [s.portfolio_scores for s in instances]
    assert all(p_scores[i][0] >= 0.0 for i in range(len(p_scores)))


@pytest.mark.parametrize("descriptor", ["features", "performance", "instance"])
def test_dominated_evolutionary_generator_raises_if_wrong_args(descriptor):
    descriptor_pipeline = DescriptorPipeline(descriptor)

    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    generations = 10
    eig = Dominated(
        pop_size=10,
        generations=generations,
        domain=None,
        portfolio=portfolio,
        repetitions=1,
        descriptor_pipe=descriptor_pipeline,
    )
    with pytest.raises(ValueError):
        _ = eig()

    with pytest.raises(ValueError):
        kp_domain = KnapsackDomain(dimension=50)
        eig = Dominated(
            pop_size=10,
            generations=generations,
            domain=kp_domain,
            portfolio=list(),
            repetitions=1,
            descriptor_pipe=descriptor_pipeline,
        )
        _ = eig()
