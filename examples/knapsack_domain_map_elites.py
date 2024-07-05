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
from digneapy.generators import MElitGen, plot_map_elites_logbook
from digneapy.operators.mutation import uniform_one_mutation
from digneapy.solvers import default_kp, map_kp, miw_kp


def map_elites_knapsack():
    archive = GridArchive(
        dimensions=(2,) * 8,
        ranges=[
            (3.0, 188190307.0),
            (10.0, 100099.0),
            (300.0, 188190307.0),
            (0.0, 100005.0),
            (0.0, 1000.0),
            (0.024096446510133377, 91415),
            (63.30284857571215, 149746.5024986119),
            (41.07985038925373, 4314244.913937186),
        ],
    )

    domain = KPDomain(dimension=10, capacity_approach="percentage")
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
    assert archive.filled_cells == len(archive.instances)
    archive = map_elites(verbose=True)
    log = map_elites.log
    print(archive.coverage, archive.filled_cells, archive.n_cells)

    plot_map_elites_logbook(log, "example.png")


if __name__ == "__main__":
    map_elites_knapsack()
