#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_domain_map_elites.py
@Time    :   2024/06/17 10:46:48
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from digneapy.archives import GridArchive
from digneapy.domains.knapsack import KPDomain
from digneapy.generators import MElitGen
from digneapy.operators.mutation import uniform_one_mutation
from digneapy.solvers import default_kp, map_kp, miw_kp


def map_elites_knapsack():
    archive = GridArchive(
        dimensions=(10,) * 8,
        ranges=[
            (1, 1e4),
            (890, 1000),
            (860, 1000.0),
            (1.0, 200),
            (1.0, 230.0),
            (0.10, 12.0),
            (400, 610),
            (240, 330),
        ],
    )
    domain = KPDomain(dimension=50, capacity_approach="percentage")
    map_elites = MElitGen(
        domain,
        portfolio=[map_kp, default_kp, miw_kp],
        archive=archive,
        initial_pop_size=10,
        mutation=uniform_one_mutation,
        generations=1000,
        strategy="features",
        repetitions=1,
    )
    archive = map_elites(verbose=True)
    log = map_elites.log
    print(log)
    print(archive.coverage)


if __name__ == "__main__":
    map_elites_knapsack()
