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

from digneapy import GridArchive
from digneapy.domains import KnapsackDomain
from digneapy.generators import MapElitesGenerator
from digneapy.operators import uniform_one_mutation
from digneapy.solvers import default_kp, map_kp, miw_kp
from digneapy.visualize import map_elites_evolution_plot


def map_elites_knapsack():
    archive = GridArchive(
        dimensions=(10,) * 101,
        ranges=[(0.0, 10000), *[(1.0, 1000) for _ in range(100)]],
    )

    domain = KnapsackDomain(dimension=50, capacity_approach="evolved")
    map_elites = MapElitesGenerator(
        domain,
        portfolio=[map_kp, default_kp, miw_kp],
        archive=archive,
        initial_pop_size=128,
        mutation=uniform_one_mutation,
        generations=100,
        descriptor="instance",
        repetitions=1,
    )

    archive = map_elites(verbose=True)
    log = map_elites.log
    print(archive.coverage, archive.n_cells, len(archive))

    map_elites_evolution_plot(log, "example.png")


if __name__ == "__main__":
    map_elites_knapsack()
