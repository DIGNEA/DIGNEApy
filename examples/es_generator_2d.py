#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   es_generator_2d.py
@Time    :   2026/05/25 11:16:12
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from digneapy import DescriptorPipeline
from digneapy.archives import GridArchive, UnstructuredArchive
from digneapy.domains import KnapsackDomain
from digneapy.generators import ES
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)
from digneapy.transformers.autoencoders import KPDecoder


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    verbose: bool = True,
):

    domain = KnapsackDomain(dimension)
    encoder = KPDecoder()
    descriptor_pipeline = DescriptorPipeline("instance", encoder)
    eig = ES(
        generator_dimension=2,
        lambda_=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        keep_only_feasible=False,
        archives=[
            UnstructuredArchive(k=15, novelty_threshold=0.1),
            UnstructuredArchive(k=1, novelty_threshold=0.01),
            GridArchive(
                dimensions=(100,) * 2,
                ranges=[(-30.0, 10.0), (-30.0, 10.0)],
            ),
        ],
        repetitions=1,
        descriptor_pipe=descriptor_pipeline,
    )

    result, archives = eig(verbose=verbose)
    print(portfolio[0].__name__, len(result.instances))
    print(archives[-1], archives[-1].coverage)


if __name__ == "__main__":
    descriptor = "instance"
    dimension = 50
    portfolios = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]
    for portfolio in portfolios:
        results = generate_instances(
            portfolio, dimension=dimension, pop_size=128, generations=1000
        )
