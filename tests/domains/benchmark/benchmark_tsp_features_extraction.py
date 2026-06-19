#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   benchmark_tsp_features_extraction.py
@Time    :   2026/06/19 10:37:50
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from digneapy.domains import TSPDomain


def benchmark_tsp_feature_extraction(benchmark):

    def setup():
        seed = np.random.SeedSequence(None)
        number_of_nodes = 50
        number_of_instances = 128
        domain = TSPDomain(number_of_nodes=number_of_nodes, seed=seed)
        instances = domain.generate_instances(number_of_instances)

        return (domain, instances), {}

    def tsp_feature_extraction(domain, instances):
        features = domain.extract_features(instances)

    benchmark.pedantic(tsp_feature_extraction, setup=setup, rounds=10, iterations=1)
