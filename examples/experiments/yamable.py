#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   yamable.py
@Time    :   2026/06/24 14:01:47
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from digneapy.archives import CVTArchive, UnstructuredArchive
from digneapy.core import DescriptorPipeline
from digneapy.domains import KnapsackDomain
from digneapy.generators import Dominated, Evolutionary, MapElites
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)


def main():

    population_size = 32
    generations = 100
    k = 3
    number_of_items = 100
    repetitions = 1
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    domain = KnapsackDomain(number_of_items=number_of_items)
    descriptor_pipeline = DescriptorPipeline("features")
    generator = Dominated(
        pop_size=population_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        repetitions=repetitions,
        k=k,
        descriptor_pipe=descriptor_pipeline,
    )
    with open("dns_representation.yaml", "w") as f:
        f.write(generator.__repr__())

    archive = CVTArchive(dimensions=100, centroids=10, ranges=[(0.0, 1e5)] * 100)
    domain = KnapsackDomain(number_of_items=number_of_items)
    descriptor_pipeline = DescriptorPipeline(key="features")
    generator = MapElites(
        domain=domain,
        portfolio=portfolio,
        pop_size=population_size,
        archive=archive,
        repetitions=repetitions,
        generations=generations,
        descriptor_pipe=descriptor_pipeline,
    )
    with open("map_elites_representation.yaml", "w") as f:
        f.write(generator.__repr__())

    population_size = 100
    number_of_items = 100
    repetitions = 10
    generations = 100
    neighbours = 15
    threshold = 0.5

    descriptor_pipeline = DescriptorPipeline("performance")
    generator = Evolutionary(
        pop_size=population_size,
        domain=KnapsackDomain(number_of_items=1000, capacity_approach="percentage"),
        portfolio=portfolio,
        archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
        descriptor_pipe=descriptor_pipeline,
        repetitions=repetitions,
        generations=generations,
    )

    with open("evolutionary_representation.yaml", "w") as f:
        f.write(generator.__repr__())


if __name__ == "__main__":
    main()
