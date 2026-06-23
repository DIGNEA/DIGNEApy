#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   novelty_search_tsp.py
@Time    :   2025/04/22 09:49:19
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools
from functools import partial
from multiprocessing import Pool, current_process

import numpy as np

from digneapy.archives import UnstructuredArchive
from digneapy.core import DescriptorKey, DescriptorPipeline
from digneapy.domains import TSPDomain
from digneapy.generators import Evolutionary
from digneapy.operators import Generational
from digneapy.solvers import nearest_neighbour, shortest_edge, two_opt
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    number_of_nodes: int,
    pop_size: int,
    generations: int,
    archive_threshold: float,
    ss_threshold: float,
    k: int,
    descriptor: DescriptorKey,
    seeds: list[np.random.SeedSequence],
    verbose,
):
    seed = seeds[current_process()._identity[0]]
    domain = TSPDomain(number_of_nodes=number_of_nodes)
    eig = Evolutionary(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        archive=UnstructuredArchive(novelty_threshold=archive_threshold, k=k),
        solution_set=UnstructuredArchive(novelty_threshold=ss_threshold, k=1),
        repetitions=1,
        descriptor_pipe=DescriptorPipeline(descriptor),
        seed=seed,
        replacement=Generational(),
    )

    result = eig()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the Travelling Salesman Problem with different solvers."
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
        "-k",
        type=int,
        required=True,
        help="Number of neighbors to use for the NS.",
        default=3,
    )
    parser.add_argument(
        "-a",
        "--archive_threshold",
        default=1e-7,
        type=float,
        help="Threshold for the Archive.",
    )
    parser.add_argument(
        "-s",
        "--solution_set_threshold",
        default=1e-10,
        type=float,
        help="Threshold for the Archive.",
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
    archive_threshold = args.archive_threshold
    solution_set_threshold = args.solution_set_threshold
    number_of_nodes = args.n
    k = args.k
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
    with Pool(N_WORKERS) as pool:
        results = pool.map(
            partial(
                generate_instances,
                number_of_nodes=number_of_nodes,
                pop_size=population_size,
                generations=generations,
                archive_threshold=archive_threshold,
                ss_threshold=solution_set_threshold,
                k=k,
                descriptor=descriptor,
                seeds=workers_seeds,
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    features_names = TSPDomain().features_names if descriptor == "features" else None
    vars_names = list(
        itertools.chain.from_iterable([
            (f"x_{i}", f"y_{i}") for i in range(number_of_nodes)
        ])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"ns_{descriptor}_N_{number_of_nodes}_target_{result.target}_rep_{rep}",
            result=result,
            variables_names=vars_names,
            descriptor_names=features_names,
            only_instances=True,
        )
