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

from digneapy import DescriptorKey, DescriptorPipeline
from digneapy.domains import TSPDomain
from digneapy.generators import Dominated
from digneapy.solvers import greedy, nneighbour, two_opt
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    k: int,
    descriptor: DescriptorKey,
    seed: int | np.random.SeedSequence,
    verbose,
):
    domain = TSPDomain(dimension=dimension)
    deig = Dominated(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor_pipe=DescriptorPipeline(descriptor),
    )
    result = deig()
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the Travelling Salesman Problem with different solvers."
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
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
    dimension = args.n
    k = args.k
    rep = args.repetition
    verbose = args.verbose
    portfolios = [
        [greedy, nneighbour, two_opt],
        [nneighbour, greedy, two_opt],
        [two_opt, greedy, nneighbour],
    ]
    print(
        f"Running with parameters:\ndimension={dimension}, k={k}, population_size={population_size}, generations={generations}, descriptor={descriptor}, verbose={verbose}"
    )
    root_sequence = np.random.SeedSequence(4123)
    workers_seed = root_sequence.spawn(4)
    with Pool(4) as pool:
        index = current_process()._identity[0]
        results = pool.map(
            partial(
                generate_instances,
                dimension=dimension,
                pop_size=population_size,
                generations=generations,
                k=k,
                descriptor=descriptor,
                seed=workers_seed[index],
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    features_names = TSPDomain().feat_names if descriptor == "features" else None
    vars_names = list(
        itertools.chain.from_iterable([(f"x_{i}", f"y_{i}") for i in range(dimension)])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"dns_{descriptor}_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            solvers_names,
            features_names,
            vars_names,
        )
