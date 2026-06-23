#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmark_evolutionary.py
@Time    :   2026/06/16 10:54:39
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy import DescriptorPipeline
from digneapy.archives import UnstructuredArchive
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import Evolutionary
from digneapy.operators import (
    OPCX,
    UCX,
    BinarySelection,
    Elitist,
    Generational,
    GreedyReplacement,
    UMut,
)
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


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
@pytest.mark.parametrize("crossover", (UCX, OPCX))
@pytest.mark.parametrize("replacement", (Generational, GreedyReplacement, Elitist))
def benchmark_evolutionary_generator_for_knapsack(
    descriptor, crossover, replacement, benchmark
):
    def setup():
        population_size = 32
        number_of_items = 50
        repetitions = 1
        generations = 100
        neighbours = 3
        threshold = 0.05

        selection = BinarySelection()
        mutation = UMut()
        descriptor_pipeline = DescriptorPipeline(descriptor)
        domain = KnapsackDomain(number_of_items=number_of_items)
        portfolio = [map_kp, default_kp, miw_kp, mpw_kp]
        generator = Evolutionary(
            pop_size=population_size,
            domain=domain,
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
            crossover=crossover(),
            mutation=mutation,
            selection=selection,
            replacement=replacement(),
        )
        return (generator,), {}

    def generate_for_knapsack(generator):
        result = generator()
        assert len(result.instances) >= 0

    benchmark.pedantic(generate_for_knapsack, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def benchmark_evolutionary_generator_for_tsp(descriptor, benchmark):
    def setup():
        population_size = 32
        number_of_nodes = 50
        repetitions = 1
        generations = 100
        neighbours = 3
        threshold = 0.05

        selection = BinarySelection()
        mutation = UMut()
        descriptor_pipeline = DescriptorPipeline(descriptor)
        domain = TSPDomain(number_of_nodes=number_of_nodes)
        portfolio = [nearest_neighbour, shortest_edge]
        generator = Evolutionary(
            pop_size=population_size,
            domain=domain,
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
            crossover=UCX(),
            mutation=mutation,
            selection=selection,
            replacement=Generational(),
        )
        return (generator,), {}

    def generate_for_tsp(generator):
        result = generator()
        assert len(result.instances) >= 0

    benchmark.pedantic(generate_for_tsp, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
@pytest.mark.parametrize("crossover", (UCX, OPCX))
def benchmark_evolutionary_generator_for_bin_packing(descriptor, crossover, benchmark):
    def setup():
        population_size = 32
        number_of_items = 50
        repetitions = 1
        generations = 100
        neighbours = 3
        threshold = 0.05

        selection = BinarySelection()
        mutation = UMut()
        descriptor_pipeline = DescriptorPipeline(descriptor)
        domain = BPPDomain(number_of_items=number_of_items)
        portfolio = [best_fit, first_fit, worst_fit, next_fit]
        generator = Evolutionary(
            pop_size=population_size,
            domain=domain,
            portfolio=portfolio,
            archive=UnstructuredArchive(k=neighbours, novelty_threshold=threshold),
            descriptor_pipe=descriptor_pipeline,
            repetitions=repetitions,
            generations=generations,
            crossover=crossover(),
            mutation=mutation,
            selection=selection,
            replacement=Generational(),
        )
        return (generator,), {}

    def generate_for_bin_packing(generator):
        result = generator()
        assert len(result.instances) >= 0

    benchmark.pedantic(generate_for_bin_packing, setup=setup, rounds=5)
