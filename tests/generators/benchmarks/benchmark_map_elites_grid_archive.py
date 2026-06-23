#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmark_map_elites_grid_archive.py
@Time    :   2026/06/18 14:49:41
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy import DescriptorPipeline, GridArchive
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import (
    MapElites,
)
from digneapy.operators import BatchUMut
from digneapy.solvers import (
    best_fit,
    default_kp,
    first_fit,
    map_kp,
    miw_kp,
    mpw_kp,
    nearest_neighbour,
    next_fit,
    shortest_edge,
    worst_fit,
)


@pytest.mark.parametrize("descriptor", ("features", "performance"))
def benchmark_map_elites_with_grid_archive_for_knapsack(descriptor, benchmark):
    def setup():
        pop_size = 32
        descriptor_pipeline = DescriptorPipeline(descriptor)
        number_of_items = 50
        generations = 100
        portfolio = [map_kp, default_kp, miw_kp, mpw_kp]

        desc_dimension = 8 if descriptor == "features" else len(portfolio)
        archive = GridArchive(
            dimensions=(10,) * desc_dimension, ranges=[(0, 1e5)] * desc_dimension
        )
        domain = KnapsackDomain(number_of_items=number_of_items)
        map_elites = MapElites(
            domain,
            portfolio=portfolio,
            archive=archive,
            pop_size=pop_size,
            mutation=BatchUMut(),
            generations=generations,
            descriptor_pipe=descriptor_pipeline,
            repetitions=1,
        )
        return (map_elites,), {}

    def generate_for_knapsack(generator):
        _ = generator()

    benchmark.pedantic(generate_for_knapsack, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ("features", "performance"))
def benchmark_map_elites_with_grid_archive_for_tsp(descriptor, benchmark):
    def setup():
        pop_size = 32
        descriptor_pipeline = DescriptorPipeline(descriptor)
        number_of_nodes = 50
        generations = 100
        portfolio = [nearest_neighbour, shortest_edge]

        desc_dimension = 11 if descriptor == "features" else len(portfolio)
        archive = GridArchive(
            dimensions=(10,) * desc_dimension, ranges=[(0, 10.0)] * desc_dimension
        )
        domain = TSPDomain(number_of_nodes=number_of_nodes)
        map_elites = MapElites(
            domain,
            portfolio=portfolio,
            archive=archive,
            pop_size=pop_size,
            mutation=BatchUMut(),
            generations=generations,
            descriptor_pipe=descriptor_pipeline,
            repetitions=1,
        )
        return (map_elites,), {}

    def generate_for_tsp(generator):
        _ = generator()

    benchmark.pedantic(generate_for_tsp, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ("features", "performance"))
def benchmark_map_elites_with_grid_archive_for_bin_packing(descriptor, benchmark):
    def setup():
        pop_size = 32
        descriptor_pipeline = DescriptorPipeline(descriptor)
        number_of_items = 50
        generations = 100
        portfolio = [best_fit, first_fit, worst_fit, next_fit]

        desc_dimension = 10 if descriptor == "features" else len(portfolio)
        archive = GridArchive(
            dimensions=(10,) * desc_dimension, ranges=[(0, 1.0)] * desc_dimension
        )
        domain = BPPDomain(number_of_items=number_of_items)
        map_elites = MapElites(
            domain,
            portfolio=portfolio,
            archive=archive,
            pop_size=pop_size,
            mutation=BatchUMut(),
            generations=generations,
            descriptor_pipe=descriptor_pipeline,
            repetitions=1,
        )
        return (map_elites,), {}

    def generate_for_bin_packing(generator):
        _ = generator()

    benchmark.pedantic(generate_for_bin_packing, setup=setup, rounds=5)
