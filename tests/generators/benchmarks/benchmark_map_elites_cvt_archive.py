#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmark_map_elites_cvt_archive.py
@Time    :   2026/06/18 14:49:29
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy import CVTArchive, DescriptorPipeline
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import (
    MapElites,
)
from digneapy.operators import BatchUMut
from digneapy.solvers import (
    best_fit,
    default_kp,
    first_fit,
    greedy,
    map_kp,
    miw_kp,
    mpw_kp,
    next_fit,
    nneighbour,
    worst_fit,
)


def build_ranges(domain_cls, descriptor, dimension, n_solvers: int):
    if domain_cls == KnapsackDomain:
        if descriptor == "features":
            return [(1.0, 100_000), *[(1.0, 1_000) for _ in range(7)]]
        elif descriptor == "performance":
            return [(1.0, 800_000) for _ in range(n_solvers)]
        else:  # case instance
            return [(1.0, 100_000), *[(1.0, 1_000) for _ in range(dimension * 2)]]
    elif domain_cls == BPPDomain:
        if descriptor == "features":
            return [
                (20, 100),
                (0, 100.0),
                (20, 100),
                (20, 100),
                (20, 100),
                *[(0.0, 1.0) for _ in range(5)],
            ]
        elif descriptor == "performance":
            return [(0.0, 1.0) for _ in range(n_solvers)]
        else:  # case instance
            return [(150, 150), *[(20, 100) for _ in range(dimension)]]
    elif domain_cls == TSPDomain:
        if descriptor == "features":
            return [(dimension * 2, dimension * 2), *[(0.0, 1_000) for _ in range(10)]]
        elif descriptor == "performance":
            return [(0.0, 1.0) for _ in range(n_solvers)]
        else:  # case instance
            return [(0, 1_000) for _ in range(dimension * 2)]


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def benchmark_map_elites_domain_cvt_for_knapsack(descriptor, benchmark):
    def setup():
        number_of_items = 100
        generations = 1000
        pop_size = 32
        ranges = build_ranges(KnapsackDomain, descriptor, number_of_items, 4)
        archive = CVTArchive(dimensions=len(ranges), centroids=1000, ranges=ranges)
        domain = KnapsackDomain(number_of_items=number_of_items)
        portfolio = [map_kp, default_kp, miw_kp, mpw_kp]
        map_elites = MapElites(
            domain,
            portfolio=portfolio,
            archive=archive,
            pop_size=pop_size,
            mutation=BatchUMut(),
            generations=generations,
            descriptor_pipe=DescriptorPipeline(key=descriptor),
            repetitions=1,
        )
        return map_elites()

    def generate_for_knapsack(generator):
        result = generator()

    benchmark.pedantic(generate_for_knapsack, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def benchmark_map_elites_domain_cvt_for_tsp(descriptor, benchmark):
    def setup():
        number_of_nodes = 50
        generations = 1000
        pop_size = 32
        ranges = build_ranges(TSPDomain, descriptor, number_of_nodes, 2)
        archive = CVTArchive(dimensions=len(ranges), centroids=1000, ranges=ranges)
        domain = TSPDomain(number_of_nodes=number_of_nodes)
        portfolio = [nneighbour, greedy]
        map_elites = MapElites(
            domain,
            portfolio=portfolio,
            archive=archive,
            pop_size=pop_size,
            mutation=BatchUMut(),
            generations=generations,
            descriptor_pipe=DescriptorPipeline(key=descriptor),
            repetitions=1,
        )
        return map_elites()

    def generate_for_tsp(generator):
        result = generator()

    benchmark.pedantic(generate_for_tsp, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def benchmark_map_elites_domain_cvt_for_bin_packing(descriptor, benchmark):
    def setup():
        number_of_items = 120
        generations = 1000
        pop_size = 32
        ranges = build_ranges(BPPDomain, descriptor, number_of_items, 4)
        archive = CVTArchive(dimensions=len(ranges), centroids=1000, ranges=ranges)
        domain = BPPDomain(number_of_items=number_of_items)
        portfolio = [best_fit, first_fit, worst_fit, next_fit]
        map_elites = MapElites(
            domain,
            portfolio=portfolio,
            archive=archive,
            pop_size=pop_size,
            mutation=BatchUMut(),
            generations=generations,
            descriptor_pipe=DescriptorPipeline(key=descriptor),
            repetitions=1,
        )
        return map_elites()

    def generate_for_bin_packing(generator):
        result = generator()

    benchmark.pedantic(generate_for_bin_packing, setup=setup, rounds=5)
