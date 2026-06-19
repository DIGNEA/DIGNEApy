#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   dominated_ns_tsp.py
@Time    :   2026/05/15 14:45:24
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

from digneapy.core import DescriptorKey, DescriptorPipeline
from digneapy.domains import TSPDomain
from digneapy.generators import Dominated
from digneapy.solvers import greedy, nneighbour, two_opt
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    number_of_nodes: int,
    pop_size: int,
    generations: int,
    k: int,
    descriptor: DescriptorKey,
    seeds: list[np.random.SeedSequence],
    verbose,
):
    seed = seeds[current_process()._identity[0]]
    domain = TSPDomain(number_of_nodes=number_of_nodes)
    deig = Dominated(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor_pipe=DescriptorPipeline(descriptor),
        seed=seed,
    )
    result = deig()

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
    k = args.k
    rep = args.repetition
    verbose = args.verbose
    portfolios = [
        [greedy, nneighbour, two_opt],
        [nneighbour, greedy, two_opt],
        [two_opt, greedy, nneighbour],
    ]
    print(
        "Running with parameters:\n"
        f"\t - number_of_nodes={number_of_nodes}"
        f"\t - k={k}"
        f"\t - population_size={population_size}"
        f"\t - generations={generations}"
        f"\t - descriptor={descriptor}"
    )
    root_sequence = np.random.SeedSequence(4213)
    workers_seeds = root_sequence.spawn(len(portfolios) + 1)
    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                number_of_nodes=number_of_nodes,
                pop_size=population_size,
                generations=generations,
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
            f"dns_{descriptor}_N_{number_of_nodes}_target_{result.target}_rep_{rep}",
            result=result,
            variables_names=vars_names,
            descriptor_names=features_names,
            only_instances=True,
        )
