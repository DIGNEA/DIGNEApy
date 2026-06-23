#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmark_dominated.py
@Time    :   2026/06/16 11:35:52
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy import DescriptorPipeline
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import Dominated
from digneapy.operators import (
    OPCX,
    UCX,
    BinarySelection,
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


@pytest.mark.parametrize("descriptor", ["performance", "instance", "features"])
@pytest.mark.parametrize("cx_cls", (UCX, OPCX))
def benchmark_dominated_generator_knapsack(descriptor, cx_cls, benchmark):
    def setup():
        pop_size = 32
        k = 3
        number_of_items = 50
        domain = KnapsackDomain(number_of_items=number_of_items)
        generations = 100
        selection = BinarySelection()
        mutation = UMut()
        crossover = cx_cls()
        descriptor_pipeline = DescriptorPipeline(descriptor)
        portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
        deig = Dominated(
            pop_size=pop_size,
            generations=generations,
            domain=domain,
            portfolio=portfolio,
            repetitions=1,
            k=k,
            descriptor_pipe=descriptor_pipeline,
            crossover=crossover,
            mutation=mutation,
            selection=selection,
        )
        return (deig, pop_size), {}

    def generate_for_knapsack(generator, population_size):
        result = generator()
        assert len(result.instances) == population_size
        instances = result.instances
        # They could be empty
        assert isinstance(instances, list)
        assert all(len(s) == len(instances[0]) for s in instances[1:])

    benchmark.pedantic(generate_for_knapsack, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ["performance", "instance", "features"])
@pytest.mark.parametrize("cx_cls", (UCX, OPCX))
def benchmark_dominated_generator_bin_packing(descriptor, cx_cls, benchmark):
    def setup():
        pop_size = 32
        k = 3
        number_of_items = 50
        domain = BPPDomain(number_of_items=number_of_items)
        generations = 100
        selection = BinarySelection()
        mutation = UMut()
        crossover = cx_cls()
        descriptor_pipeline = DescriptorPipeline(descriptor)
        portfolio = [best_fit, first_fit, worst_fit, next_fit]
        deig = Dominated(
            pop_size=pop_size,
            generations=generations,
            domain=domain,
            portfolio=portfolio,
            repetitions=1,
            k=k,
            descriptor_pipe=descriptor_pipeline,
            crossover=crossover,
            mutation=mutation,
            selection=selection,
        )
        return (deig, pop_size), {}

    def generate_for_bin_packing(generator, population_size):
        result = generator()
        assert len(result.instances) == population_size
        instances = result.instances
        # They could be empty
        assert isinstance(instances, list)
        assert all(len(s) == len(instances[0]) for s in instances[1:])

    benchmark.pedantic(generate_for_bin_packing, setup=setup, rounds=5)


@pytest.mark.parametrize("descriptor", ["performance", "instance", "features"])
@pytest.mark.parametrize("cx_cls", (UCX, OPCX))
def benchmark_dominated_generator_tsp(descriptor, cx_cls, benchmark):
    def setup():
        pop_size = 32
        k = 3
        number_of_nodes = 50
        domain = TSPDomain(number_of_nodes=number_of_nodes)
        generations = 100
        selection = BinarySelection()
        mutation = UMut()
        crossover = cx_cls()
        descriptor_pipeline = DescriptorPipeline(descriptor)
        portfolio = [nearest_neighbour, shortest_edge]
        deig = Dominated(
            pop_size=pop_size,
            generations=generations,
            domain=domain,
            portfolio=portfolio,
            repetitions=1,
            k=k,
            descriptor_pipe=descriptor_pipeline,
            crossover=crossover,
            mutation=mutation,
            selection=selection,
        )
        return (deig, pop_size), {}

    def generate_for_tsp(generator, population_size):
        result = generator()
        assert len(result.instances) == population_size
        instances = result.instances
        # They could be empty
        assert isinstance(instances, list)
        assert all(len(s) == len(instances[0]) for s in instances[1:])

    benchmark.pedantic(generate_for_tsp, setup=setup, rounds=5)
