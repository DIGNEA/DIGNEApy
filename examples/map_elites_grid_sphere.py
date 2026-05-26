#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   map_elites_grid_sphere.py
@Time    :   2026/05/21 10:53:18
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Sequence

import numpy as np

from digneapy import DescriptorPipeline, GridArchive, Instance, Solver
from digneapy.domains import SphereDomain
from digneapy.generators import MapElites, PlottedMapElites
from digneapy.operators import BatchUMut
from digneapy.solvers import random_solver
from digneapy.visualize import ArchivePlotter


def generate_instances(
    portfolio: Sequence[Solver],
    dimension: int,
    pop_size: int,
    generations: int,
    resolution: int = 100,
    verbose: bool = False,
):
    archive = GridArchive(
        dimensions=[resolution, resolution],
        ranges=[(-5.12, 5.12), (-5.12, 5.12)],
    )
    domain = SphereDomain(dimension=dimension)
    map_elites = MapElites(
        domain,
        portfolio=portfolio,
        archive=archive,
        pop_size=pop_size,
        mutation=BatchUMut(seed=32),
        generations=generations,
        describe_pipe=DescriptorPipeline("instance"),
        repetitions=1,
    )
    plotted_map_elites = PlottedMapElites(map_elites)

    return plotted_map_elites(verbose=verbose)


if __name__ == "__main__":
    portfolio = [random_solver, random_solver, random_solver]
    dimension = 2
    pop_size = 16
    generations = 10_000
    results = generate_instances(
        portfolio=portfolio,
        dimension=dimension,
        pop_size=pop_size,
        generations=generations,
    )
    print(results)
    archive = GridArchive(
        dimensions=[10, 10],
        ranges=[(-5.12, 5.12), (-5.12, 5.12)],
    )
    plotter = ArchivePlotter(archive)
    sequence = np.random.SeedSequence()
    GENERATIONS = 1_000
    INSTANCES = 10
    seed = sequence.spawn(1)
    rng = np.random.default_rng(seed[0])
    domain = SphereDomain(dimension=2)
    lbs, ubs = domain.lbs, domain.ubs
    for g in range(GENERATIONS):
        variables = np.random.uniform(low=lbs, high=ubs, size=(INSTANCES, 2))
        new_instances = [
            Instance(
                variables=variables[i],
                descriptor=variables[i],
                fitness=np.float64(1e5),
            )
            for i in range(INSTANCES)
        ]
        archive.extend(new_instances)
        plotter.update(generation=g)
        print(archive.coverage)
