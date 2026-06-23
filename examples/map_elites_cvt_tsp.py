#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   map_elites_cvt_tsp.py
@Time    :   2026/04/07 10:04:25
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools
from functools import partial
from multiprocessing import Pool, current_process

import numpy as np

from digneapy.archives import CVTArchive
from digneapy.core import DescriptorKey, DescriptorPipeline
from digneapy.domains import TSPDomain
from digneapy.generators import MapElites
from digneapy.operators import BatchUMut
from digneapy.solvers import nearest_neighbour, shortest_edge, two_opt
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    number_of_nodes: int,
    pop_size: int,
    generations: int,
    descriptor: DescriptorKey,
    seeds: list[np.random.SeedSequence],
    verbose: bool,
):
    domain = TSPDomain(number_of_nodes=number_of_nodes)
    seed = seeds[current_process()._identity[0]]
    # Create an empty archive with the previous centroids and samples
    ranges = []
    if descriptor == "features":
        ranges = [
            (number_of_nodes * 2, number_of_nodes * 2),
            *[(0.0, 1_000) for _ in range(10)],
        ]
    elif descriptor == "performance":
        ranges = [(0.0, 1.0) for _ in range(len(portfolio))]
    else:  # case instance
        ranges = [(0, 1_000) for _ in range(number_of_nodes * 2)]
    cvt_archive = CVTArchive(
        dimensions=len(ranges),
        centroids=1_000,
        ranges=ranges,
        seed=seed,
    )
    map_elites = MapElites(
        domain=domain,
        portfolio=portfolio,
        archive=cvt_archive,
        pop_size=pop_size,
        mutation=BatchUMut(),
        generations=generations,
        descriptor_pipe=DescriptorPipeline(descriptor),
        repetitions=1,
        seed=seed,
    )

    result = map_elites(verbose=verbose)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the Travelling Salesman Problem with different solvers using CVT-MapElites."
    )
    parser.add_argument(
        "-n",
        "-number_of_nodes",
        type=int,
        required=True,
        help="number of nodes.",
        default=50,
    )
    parser.add_argument(
        "-d", "--descriptor", type=str, required=True, help="Descriptor to use."
    )

    parser.add_argument(
        "-p",
        "--population_size",
        default=128,
        type=int,
        required=True,
        help="Number of instances to evolve.",
    )
    parser.add_argument(
        "-g",
        "--generations",
        default=1000,
        type=int,
        required=True,
        help="Number of generations to perform.",
    )
    parser.add_argument(
        "-r", "--repetition", type=int, required=True, help="Repetition index."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Print the evolution logbook.",
    )
    args = parser.parse_args()
    descriptor = args.descriptor
    generations = args.generations
    population_size = args.population_size
    number_of_nodes = args.n
    rep = args.repetition
    verbose = args.verbose
    portfolios = [
        [shortest_edge, nearest_neighbour, two_opt],
        [nearest_neighbour, shortest_edge, two_opt],
        [two_opt, shortest_edge, nearest_neighbour],
    ]
    root_sequence = np.random.SeedSequence(1342)
    N_WORKERS = len(portfolios)
    workers_seeds = root_sequence.spawn(N_WORKERS + 1)
    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                number_of_nodes=number_of_nodes,
                pop_size=population_size,
                generations=generations,
                descriptor=descriptor,
                seeds=workers_seeds,
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    vars_names = list(
        itertools.chain.from_iterable([
            (f"x_{i}", f"y_{i}") for i in range(number_of_nodes)
        ])
    )
    features_names = TSPDomain().features_names if descriptor == "features" else None

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"map_elites_cvt_{descriptor}_tsp_N_{number_of_nodes}_target_{result.target}_rep_{rep}",
            result=result,
            variables_names=vars_names,
            descriptor_names=features_names,
            only_instances=True,
        )
