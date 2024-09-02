#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_map_elites.py
@Time    :   2024/06/19 13:11:37
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import os

import pytest

from digneapy.archives import GridArchive
from digneapy.core import Instance
from digneapy.domains.bin_packing import BPPDomain
from digneapy.domains.knapsack import KPDomain
from digneapy.generators import MElitGen, plot_map_elites_logbook
from digneapy.operators.mutation import uniform_one_mutation
from digneapy.solvers import best_fit, default_kp, first_fit, map_kp, miw_kp, worst_fit

test_data = [
    (
        KPDomain,
        [default_kp, map_kp, miw_kp],
        8,
        "MapElites(descriptor=features,pop_size=5,gen=1000,domain=KP,portfolio=['default_kp', 'map_kp', 'miw_kp'])",
        "MapElites<descriptor=features,pop_size=5,gen=1000,domain=KP,portfolio=['default_kp', 'map_kp', 'miw_kp']>",
    ),
    (
        BPPDomain,
        [best_fit, first_fit, worst_fit],
        10,
        "MapElites(descriptor=features,pop_size=5,gen=1000,domain=BPP,portfolio=['best_fit', 'first_fit', 'worst_fit'])",
        "MapElites<descriptor=features,pop_size=5,gen=1000,domain=BPP,portfolio=['best_fit', 'first_fit', 'worst_fit']>",
    ),
]


@pytest.mark.parametrize(
    "domain_cls, portfolio, desc_size, expected_str, expected_repr", test_data
)
def test_map_elites_domain(
    domain_cls, portfolio, desc_size, expected_str, expected_repr
):
    dimension = 100
    archive = GridArchive(dimensions=(10,) * desc_size, ranges=[(0, 1e4)] * desc_size)
    domain = domain_cls(dimension=dimension)
    assert domain.dimension == dimension

    map_elites = MElitGen(
        domain,
        portfolio=portfolio,
        archive=archive,
        initial_pop_size=5,
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
    plot_map_elites_logbook(log, "example.png")
    assert os.path.exists("example.png")
    os.remove("example.png")
