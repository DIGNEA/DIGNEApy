#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmark_tsp_solvers.py
@Time    :   2026/06/19 15:21:05
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from digneapy.domains import TSPDomain
from digneapy.solvers.tsp import nearest_neighbour, shortest_edge, two_opt


def benchmark_tsp_shortest_edge(benchmark):

    def setup():
        seed = np.random.SeedSequence(13)
        number_of_nodes = 100
        domain = TSPDomain(number_of_nodes=number_of_nodes, seed=seed)
        instances = domain.generate_instances(1)
        problem = domain.generate_problems_from_instances(instances)[0]
        return (problem,), {}

    def tsp_shortest_edge(problem):
        solution = shortest_edge(problem)

    benchmark.pedantic(tsp_shortest_edge, setup=setup, rounds=10, iterations=1)


def benchmark_tsp_nearest_neighbour(benchmark):

    def setup():
        seed = np.random.SeedSequence(13)
        number_of_nodes = 100
        domain = TSPDomain(number_of_nodes=number_of_nodes, seed=seed)
        instances = domain.generate_instances(1)
        problem = domain.generate_problems_from_instances(instances)[0]
        return (problem,), {}

    def tsp_nearest_neighbour(problem):
        solution = nearest_neighbour(problem)

    benchmark.pedantic(tsp_nearest_neighbour, setup=setup, rounds=10, iterations=1)


def benchmark_tsp_2_Opt(benchmark):

    def setup():
        seed = np.random.SeedSequence(13)
        number_of_nodes = 100
        domain = TSPDomain(number_of_nodes=number_of_nodes, seed=seed)
        instances = domain.generate_instances(1)
        problem = domain.generate_problems_from_instances(instances)[0]
        return (problem,), {}

    def tsp_2_Opt(problem):
        solution = two_opt(problem)

    benchmark.pedantic(tsp_2_Opt, setup=setup, rounds=10, iterations=1)
