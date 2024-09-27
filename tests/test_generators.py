#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_generator.py
@Time    :   2024/04/15 12:02:25
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import os
from collections import deque

import pytest

from digneapy import (
    Archive,
    CVTArchive,
    GridArchive,
    Instance,
    max_gap_target,
    runtime_score,
)
from digneapy.domains import BPPDomain, KnapsackDomain
from digneapy.generators import EAGenerator, MapElitesGenerator
from digneapy.operators import (
    binary_tournament_selection,
    generational_replacement,
    uniform_crossover,
    uniform_one_mutation,
)
from digneapy.solvers import (
    best_fit,
    default_kp,
    first_fit,
    map_kp,
    miw_kp,
    mpw_kp,
    worst_fit,
)
from digneapy.solvers.pisinger import combo, expknap, minknap
from digneapy.visualize import ea_generator_evolution_plot, map_elites_evolution_plot


def test_default_generator():
    eig = EAGenerator(domain=None, portfolio=[])
    assert eig.pop_size == 100
    assert eig.generations == 1000
    assert eig.k == 15
    assert eig._describe_by == "features"
    assert eig._transformer is None
    assert eig.domain is None
    assert eig.portfolio == tuple()
    assert eig.repetitions == 1
    assert eig.cxrate == 0.5
    assert eig.mutrate == 0.8
    assert eig.crossover == uniform_crossover
    assert eig.mutation == uniform_one_mutation
    assert eig.selection == binary_tournament_selection
    assert eig.replacement == generational_replacement
    assert eig.phi == 0.85
    assert eig.performance_function is not None
    assert eig.performance_function == max_gap_target

    assert (
        eig.__str__()
        == "EAGenerator(pop_size=100,gen=1000,domain=None,portfolio=[],NS(descriptor=features,k=15,A=(),S_S=()))"
    )

    assert (
        eig.__repr__()
        == "EAGenerator<pop_size=100,gen=1000,domain=None,portfolio=[],NS<descriptor=features,k=15,A=(),S_S=()>>"
    )

    with pytest.raises(ValueError) as e:
        eig()
    assert e.value.args[0] == "You must specify a domain to run the generator."

    eig.domain = KnapsackDomain()
    with pytest.raises(ValueError) as e:
        eig()
    assert (
        e.value.args[0]
        == "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
    )

    with pytest.raises(ValueError) as e:
        eig = EAGenerator(domain=None, portfolio=[], phi=-1.0)
    assert (
        e.value.args[0]
        == "Phi must be a float number in the range [0.0-1.0]. Got: -1.0."
    )
    with pytest.raises(ValueError) as e:
        eig = EAGenerator(domain=None, portfolio=[], phi="hello")
    assert e.value.args[0] == "Phi must be a float number in the range [0.0-1.0]."


def test_eig_gen_kp_perf_descriptor():
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    generations = 100
    k = 3
    eig = EAGenerator(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="performance",
        replacement=generational_replacement,
    )
    archive, solution_set = eig()
    # They could be empty
    assert isinstance(archive, Archive)
    assert isinstance(solution_set, Archive)
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(len(s.descriptor) == len(portfolio) for s in archive)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s.portfolio_scores for s in archive]
        # The instances are biased to the performance of the target
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.descriptor) == len(portfolio) for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))


def test_eig_gen_kp_feat_descriptor():
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    generations = 100
    k = 3
    eig = EAGenerator(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="features",
        replacement=generational_replacement,
    )
    archive, solution_set = eig()
    # They could be empty
    assert isinstance(archive, Archive)
    assert isinstance(solution_set, Archive)
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(len(s.descriptor) == 8 for s in archive)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s.portfolio_scores for s in archive]
        # The instances are biased to the performance of the target
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.descriptor) == 8 for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    # Test the creation of the evolution images
    log = eig._logbook
    assert len(log) == eig.generations
    filename = "test_evolution.png"
    ea_generator_evolution_plot(log, filename=filename)
    assert os.path.exists(filename)
    os.remove(filename)


def test_eig_gen_kp_inst_descriptor():
    portfolio = deque([map_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    generations = 100
    k = 3
    eig = EAGenerator(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="instance",
        replacement=generational_replacement,
    )
    archive, solution_set = eig()
    # They could be empty
    assert isinstance(archive, Archive)
    assert isinstance(solution_set, Archive)
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(
            len(s.descriptor) == len(s) for s in archive
        )  # Because we do not calculate features in this case
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s.portfolio_scores for s in archive]
        # The instances are biased to the performance of the target
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.descriptor) == len(s) for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))


def test_eig_gen_kp_perf_descriptor_with_pisinger():
    portfolio = deque([combo, minknap, expknap])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    generations = 100
    k = 3
    eig = EAGenerator(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="performance",
        replacement=generational_replacement,
        performance_function=runtime_score,
    )
    archive, solution_set = eig()
    # They could be empty
    assert isinstance(archive, Archive)
    assert isinstance(solution_set, Archive)
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(len(s.descriptor) == len(portfolio) for s in archive)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s.portfolio_scores for s in archive]
        # The instances are biased to the performance of the target
        # in this case, the performance score is the minimum because
        # we are measuring running time
        assert all(min(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.descriptor) == len(portfolio) for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(min(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))


test_data = [
    (
        KnapsackDomain,
        [default_kp, map_kp, miw_kp],
        8,
        "MapElites(descriptor=features,pop_size=32,gen=1000,domain=KP,portfolio=['default_kp', 'map_kp', 'miw_kp'])",
        "MapElites<descriptor=features,pop_size=32,gen=1000,domain=KP,portfolio=['default_kp', 'map_kp', 'miw_kp']>",
    ),
    (
        BPPDomain,
        [best_fit, first_fit, worst_fit],
        10,
        "MapElites(descriptor=features,pop_size=32,gen=1000,domain=BPP,portfolio=['best_fit', 'first_fit', 'worst_fit'])",
        "MapElites<descriptor=features,pop_size=32,gen=1000,domain=BPP,portfolio=['best_fit', 'first_fit', 'worst_fit']>",
    ),
]


@pytest.mark.parametrize(
    "domain_cls, portfolio, desc_size, expected_str, expected_repr", test_data
)
def test_map_elites_domain_grid(
    domain_cls, portfolio, desc_size, expected_str, expected_repr
):
    dimension = 50
    archive = GridArchive(dimensions=(10,) * desc_size, ranges=[(0, 1e4)] * desc_size)
    domain = domain_cls(dimension=dimension)
    assert domain.dimension == dimension

    map_elites = MapElitesGenerator(
        domain,
        portfolio=portfolio,
        archive=archive,
        initial_pop_size=32,
        mutation=uniform_one_mutation,
        generations=1000,
        descriptor="features",
        repetitions=1,
    )
    archive = map_elites()
    assert map_elites.__str__() == expected_str
    assert map_elites.__repr__() == expected_repr

    assert len(archive) != 0
    assert isinstance(archive, GridArchive)
    assert all(isinstance(i, Instance) for i in archive)
    assert len(map_elites.log) == 1001

    # Is able to print the log
    log = map_elites.log
    map_elites_evolution_plot(log, "example.png")
    assert os.path.exists("example.png")
    os.remove("example.png")


test_data_cvt = [
    (
        KnapsackDomain,
        [map_kp, default_kp, miw_kp],
        [(1.0, 10_000), *[(1.0, 1_000) for _ in range(100)]],
    ),
    (
        BPPDomain,
        [best_fit, first_fit, worst_fit],
        [(1.0, 10_000), *[(1.0, 1_000) for _ in range(50)]],
    ),
]


@pytest.mark.parametrize("domain_cls, portfolio, ranges", test_data_cvt)
def test_map_elites_domain_cvt(domain_cls, portfolio, ranges):
    dimension = 50
    archive = CVTArchive(k=1000, ranges=ranges, n_samples=10_000)
    domain = domain_cls(dimension=dimension)
    assert domain.dimension == dimension

    map_elites = MapElitesGenerator(
        domain,
        portfolio=portfolio,
        archive=archive,
        initial_pop_size=32,
        mutation=uniform_one_mutation,
        generations=1000,
        descriptor="instance",
        repetitions=1,
    )
    archive = map_elites()

    assert len(archive) != 0
    assert all(i.p >= 0 for i in archive)
    assert all(i.s == 0 for i in archive)
    assert isinstance(archive, CVTArchive)
    assert all(isinstance(i, Instance) for i in archive)
    assert len(map_elites.log) == 1001

    # Is able to print the log
    log = map_elites.log
    map_elites_evolution_plot(log, "example.png")
    assert os.path.exists("example.png")
    os.remove("example.png")
