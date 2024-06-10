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

import pytest
from digneapy.archives import Archive
from digneapy.generators import EIG
from digneapy.generators import def_perf_metric, pis_perf_metric
from digneapy.generators.utils import plot_generator_logbook
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)
from digneapy.solvers.pisinger import combo, minknap, expknap
from digneapy.domains.knapsack import KPDomain
from digneapy.operators import crossover, selection, mutation, replacement
from collections import deque
import os


def test_default_generator():
    eig = EIG(domain=None, portfolio=[])
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
    assert eig.crossover == crossover.uniform_crossover
    assert eig.mutation == mutation.uniform_one_mutation
    assert eig.selection == selection.binary_tournament_selection
    assert eig.replacement == replacement.generational
    assert eig.phi == 0.85
    assert eig.performance_function is not None
    assert eig.performance_function == def_perf_metric

    assert (
        eig.__str__()
        == f"EIG(pop_size=100,gen=1000,domain=None,portfolio=[],NS(descriptor=features,k=15,A=(),S_S=()))"
    )

    assert (
        eig.__repr__()
        == f"EIG<pop_size=100,gen=1000,domain=None,portfolio=[],NS<descriptor=features,k=15,A=(),S_S=()>>"
    )

    with pytest.raises(AttributeError) as e:
        eig()
    assert e.value.args[0] == "You must specify a domain to run the generator."

    eig.domain = KPDomain()
    with pytest.raises(AttributeError) as e:
        eig()
    assert (
        e.value.args[0]
        == "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
    )

    with pytest.raises(AttributeError) as e:
        eig = EIG(domain=None, portfolio=[], phi=-1.0)
    assert (
        e.value.args[0]
        == f"Phi must be a float number in the range [0.0-1.0]. Got: -1.0."
    )
    with pytest.raises(AttributeError) as e:
        eig = EIG(domain=None, portfolio=[], phi="hello")
    assert e.value.args[0] == f"Phi must be a float number in the range [0.0-1.0]."


def test_eig_gen_kp_perf_descriptor():
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KPDomain(dimension=50, capacity_approach="evolved")
    generations = 1000
    k = 3
    eig = EIG(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="performance",
        replacement=replacement.generational,
    )
    archive, solution_set = eig()
    # They could be empty
    assert type(archive) == Archive
    assert type(solution_set) == Archive
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(len(s.features) == 0 for s in archive)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s._portfolio_m for s in archive]
        # The instances are biased to the performance of the target
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.features) == 0 for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s._portfolio_m for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    # Check it does not insert any when list is empty
    current_len = len(eig.archive)
    eig._update_archive(list())
    assert current_len == len(eig.archive)

    current_len = len(eig.solution_set)
    eig._update_solution_set(list())
    assert current_len == len(eig.solution_set)


def test_eig_gen_kp_feat_descriptor():
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KPDomain(dimension=50, capacity_approach="evolved")
    generations = 1000
    k = 3
    eig = EIG(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="features",
        replacement=replacement.generational,
    )
    archive, solution_set = eig()
    # They could be empty
    assert type(archive) == Archive
    assert type(solution_set) == Archive
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(len(s.features) == 8 for s in archive)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s._portfolio_m for s in archive]
        # The instances are biased to the performance of the target
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.features) == 8 for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s._portfolio_m for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    # Check it does not insert any when list is empty
    current_len = len(eig.archive)
    eig._update_archive(list())
    assert current_len == len(eig.archive)

    current_len = len(eig.solution_set)
    eig._update_solution_set(list())
    assert current_len == len(eig.solution_set)

    # Test the creation of the evolution images
    log = eig.logbook
    assert len(log) == eig.generations
    filename = "test_evolution.png"
    plot_generator_logbook(log, filename=filename)
    assert os.path.exists(filename)
    os.remove(filename)


def test_eig_gen_kp_inst_descriptor():
    portfolio = deque([map_kp, mpw_kp])
    kp_domain = KPDomain(dimension=50, capacity_approach="evolved")
    generations = 1000
    k = 3
    eig = EIG(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="instance",
        replacement=replacement.generational,
    )
    archive, solution_set = eig()
    # They could be empty
    assert type(archive) == Archive
    assert type(solution_set) == Archive
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(
            len(s.features) == 0 for s in archive
        )  # Because we do not calculate features in this case
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s._portfolio_m for s in archive]
        # The instances are biased to the performance of the target
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.features) == 0 for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s._portfolio_m for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    # Check it does not insert any when list is empty
    current_len = len(eig.archive)
    eig._update_archive(list())
    assert current_len == len(eig.archive)

    current_len = len(eig.solution_set)
    eig._update_solution_set(list())
    assert current_len == len(eig.solution_set)


def test_eig_gen_kp_perf_descriptor_with_pisinger():
    portfolio = deque([combo, minknap, expknap])
    kp_domain = KPDomain(dimension=50, capacity_approach="evolved")
    generations = 1000
    k = 3
    eig = EIG(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor="performance",
        replacement=replacement.generational,
        performance_function=pis_perf_metric,
    )
    archive, solution_set = eig()
    # They could be empty
    assert type(archive) == Archive
    assert type(solution_set) == Archive
    # If they're not empty
    if len(archive) != 0:
        assert all(len(s) == 101 for s in archive)
        assert all(s.fitness >= 0.0 for s in archive)
        assert all(s.p >= 0.0 for s in archive)
        assert all(s.s >= 0.0 for s in archive)
        assert all(len(s.features) == 0 for s in archive)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in archive)
        p_scores = [s._portfolio_m for s in archive]
        # The instances are biased to the performance of the target
        # in this case, the performance score is the minimum because
        # we are measuring running time
        assert all(min(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.features) == 0 for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s._portfolio_m for s in solution_set]
        assert all(min(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))

    # Check it does not insert any when list is empty
    current_len = len(eig.archive)
    eig._update_archive(list())
    assert current_len == len(eig.archive)

    current_len = len(eig.solution_set)
    eig._update_solution_set(list())
    assert current_len == len(eig.solution_set)
