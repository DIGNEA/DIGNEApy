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

import os
from collections import deque

import numpy as np
import pytest

from digneapy import (
    NS,
    Archive,
    max_gap_target,
)
from digneapy.domains import KnapsackDomain
from digneapy.generators import (
    Dominated,
    Evolutionary,
)
from digneapy.operators import (
    binary_tournament_selection,
    generational_replacement,
    uniform_crossover,
    uniform_one_mutation,
)
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)
from digneapy.visualize import ea_generator_evolution_plot


def test_default_generator():
    eig = Evolutionary(
        pop_size=100, domain=None, portfolio=[], novelty_approach=NS(k=15)
    )
    assert eig._pop_size == 100
    assert eig._generations == 1000
    assert eig._novelty_search.k == 15
    assert eig._describe_by == "features"
    assert eig._transformer is None
    assert eig._domain is None
    assert eig._portfolio == tuple()
    assert eig._repetitions == 1
    assert np.isclose(eig.cxrate, 0.5)
    assert np.isclose(eig.mutrate, 0.8)
    assert eig.crossover == uniform_crossover
    assert eig.mutation == uniform_one_mutation
    assert eig.selection == binary_tournament_selection
    assert eig.replacement == generational_replacement
    assert np.isclose(eig.phi, 0.85)
    assert eig._performance_fn is not None
    assert eig._performance_fn == max_gap_target

    assert (
        eig.__str__() == "Evolutionary(pop_size=100,gen=1000,domain=None,portfolio=[])"
    )

    assert (
        eig.__repr__() == "Evolutionary<pop_size=100,gen=1000,domain=None,portfolio=[]>"
    )

    with pytest.raises(ValueError) as e:
        eig()
    assert e.value.args[0] == "You must specify a domain to run the generator."

    eig._domain = KnapsackDomain()
    with pytest.raises(ValueError) as e:
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


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_evo_instance_generator_for_KP_with_performance_descriptor(capacity_approach):
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach=capacity_approach)
    generations = 10
    k = 3
    eig = Evolutionary(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        novelty_approach=NS(k=k),
        solution_set=Archive(threshold=3),
        portfolio=portfolio,
        repetitions=1,
        describe_by="performance",
        replacement=generational_replacement,
    )
    result = eig()
    solution_set = result.instances
    # They could be empty
    assert isinstance(solution_set, Archive)

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0 for s in solution_set)
        assert all(s.s >= 0 for s in solution_set)
        assert all(len(s.descriptor) == len(portfolio) for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))


@pytest.mark.parametrize("capacity_approach", ("fixed", "evolved", "percentage"))
def test_evo_instance_generator_for_KP_with_features_descriptor(capacity_approach):
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach=capacity_approach)
    generations = 10
    k = 3
    eig = Evolutionary(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        novelty_approach=NS(k=k),
        solution_set=Archive(threshold=3),
        repetitions=1,
        describe_by="features",
        replacement=generational_replacement,
    )
    result = eig()
    solution_set = result.instances
    # They could be empty
    assert isinstance(solution_set, Archive)

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
    assert len(log) == eig._generations
    filename = "test_evolution.png"
    ea_generator_evolution_plot(log.logbook, filename=filename)
    assert os.path.exists(filename)
    os.remove(filename)


def test_eig_gen_kp_inst_descriptor():
    portfolio = deque([map_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    generations = 10
    k = 3
    eig = Evolutionary(
        pop_size=10,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        novelty_approach=NS(k=k),
        solution_set=Archive(threshold=3),
        repetitions=1,
        describe_by="instance",
        replacement=generational_replacement,
    )
    result = eig()
    solution_set = result.instances
    # They could be empty
    assert isinstance(solution_set, Archive)

    if len(solution_set) != 0:
        assert all(len(s) == 101 for s in solution_set)
        assert all(s.fitness >= 0.0 for s in solution_set)
        assert all(s.p >= 0.0 for s in solution_set)
        assert all(s.s >= 0.0 for s in solution_set)
        assert all(len(s.descriptor) == len(s) for s in solution_set)
        assert all(len(s.portfolio_scores) == len(portfolio) for s in solution_set)
        p_scores = [s.portfolio_scores for s in solution_set]
        assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))


@pytest.mark.parametrize("k", [3, 15, 30])
@pytest.mark.parametrize("descriptor", ["features", "performance", "instance"])
def test_dominated_evolutionary_generator_with_k_and_descriptor(k, descriptor):
    portfolio = [map_kp, miw_kp, mpw_kp]
    pop_size = 128
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    generations = 10
    deig = Dominated(
        pop_size=pop_size,
        generations=generations,
        domain=kp_domain,
        portfolio=portfolio,
        repetitions=1,
        k=k,
        describe_by=descriptor,
    )
    portfolio_names = [s.__name__ for s in portfolio]
    expected_str = f"pop_size={pop_size},gen={generations},domain={kp_domain.name},portfolio={portfolio_names}"
    assert deig.__str__() == f"Dominated({expected_str})"
    assert deig.__repr__() == f"Dominated<{expected_str}>"
    result = deig()
    assert len(result.instances) == pop_size
    instances = result.instances
    # They could be empty
    assert isinstance(instances, list)
    assert all(len(s) == 101 for s in instances)
    assert all(s.fitness >= 0.0 for s in instances)
    fitness = [s.fitness for s in instances]
    sorted_fitness = sorted(fitness, reverse=True)
    assert fitness == sorted_fitness
    if descriptor == "features":
        assert all(len(s.descriptor) == 8 for s in instances)
        assert all((s.descriptor == s.features).all() for s in instances)
    elif descriptor == "performance":
        assert all(len(s.descriptor) == 3 for s in instances)
    else:
        assert all(len(s.descriptor) == 101 for s in instances)

    assert all(len(s.portfolio_scores) == len(portfolio) for s in instances)
    # TODO: DNS do not guarantees the biased aspect of the instances
    # p_scores = [s.portfolio_scores for s in instances]
    # assert all(max(p_scores[i]) == p_scores[i][0] for i in range(len(p_scores)))


def test_dominated_evolutionary_generator_raises_if_wrong_args():
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    generations = 10
    eig = Dominated(
        pop_size=10,
        generations=generations,
        domain=None,
        portfolio=portfolio,
        repetitions=1,
        describe_by="features",
    )
    with pytest.raises(ValueError):
        _ = eig()

    with pytest.raises(ValueError):
        eig = Dominated(
            pop_size=10,
            generations=generations,
            domain=kp_domain,
            portfolio=list(),
            repetitions=1,
            describe_by="features",
        )
        _ = eig()
