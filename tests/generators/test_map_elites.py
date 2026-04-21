#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_map_elites.py
@Time    :   2026/04/21 14:04:19
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import os

import pytest

from digneapy import (
    CVTArchive,
    GridArchive,
    Instance,
)
from digneapy.domains import BPPDomain, KnapsackDomain
from digneapy.generators import (
    MapElites,
)
from digneapy.operators import (
    uniform_one_mutation,
)
from digneapy.solvers import (
    best_fit,
    default_kp,
    first_fit,
    map_kp,
    miw_kp,
    worst_fit,
)
from digneapy.visualize import map_elites_evolution_plot

test_data = [
    (
        KnapsackDomain,
        [default_kp, map_kp, miw_kp],
        8,
        "MapElites(pop_size=32,gen=10,domain=KP,portfolio=['default_kp', 'map_kp', 'miw_kp'])",
        "MapElites<pop_size=32,gen=10,domain=KP,portfolio=['default_kp', 'map_kp', 'miw_kp']>",
    ),
    (
        BPPDomain,
        [best_fit, first_fit, worst_fit],
        10,
        "MapElites(pop_size=32,gen=10,domain=BPP,portfolio=['best_fit', 'first_fit', 'worst_fit'])",
        "MapElites<pop_size=32,gen=10,domain=BPP,portfolio=['best_fit', 'first_fit', 'worst_fit']>",
    ),
]


@pytest.mark.parametrize(
    "domain_cls, portfolio, desc_size, expected_str, expected_repr", test_data
)
def test_map_elites_domain_grid(
    domain_cls, portfolio, desc_size, expected_str, expected_repr
):
    dimension = 50
    generations = 10
    archive = GridArchive(dimensions=(10,) * desc_size, ranges=[(0, 1e4)] * desc_size)
    domain = domain_cls(dimension=dimension)
    assert domain.dimension == dimension

    map_elites = MapElites(
        domain,
        portfolio=portfolio,
        archive=archive,
        pop_size=32,
        mutation=uniform_one_mutation,
        generations=generations,
        describe_by="features",
        repetitions=1,
    )
    result = map_elites()
    archive = result.instances
    assert map_elites.__str__() == expected_str
    assert map_elites.__repr__() == expected_repr

    assert len(archive) != 0
    assert isinstance(archive, GridArchive)
    assert all(isinstance(i, Instance) for i in archive)
    assert len(map_elites.log) == generations + 1

    # Is able to print the log
    log = map_elites.log
    map_elites_evolution_plot(log.logbook, "example.png")
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
    generations = 10
    archive = CVTArchive(k=1000, ranges=ranges, n_samples=10_000)
    domain = domain_cls(dimension=dimension)
    assert domain.dimension == dimension

    map_elites = MapElites(
        domain,
        portfolio=portfolio,
        archive=archive,
        pop_size=32,
        mutation=uniform_one_mutation,
        generations=generations,
        describe_by="instance",
        repetitions=1,
    )
    result = map_elites()
    archive = result.instances
    assert len(archive) != 0
    assert all(i.p >= 0 for i in archive)
    assert all(i.s == 0 for i in archive)
    assert isinstance(archive, CVTArchive)
    assert all(isinstance(i, Instance) for i in archive)
    assert len(map_elites.log) == generations + 1

    # Is able to print the log
    log = map_elites.log
    map_elites_evolution_plot(log.logbook, "example.png")
    assert os.path.exists("example.png")
    os.remove("example.png")
