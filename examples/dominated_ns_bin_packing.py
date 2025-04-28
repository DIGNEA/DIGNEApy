#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   dominated_ns_bin_packing.py
@Time    :   2025/04/21 11:05:57
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import argparse
from digneapy.domains import BPPDomain
from digneapy.domains.bpp import BPP
from digneapy.utils import save_results_to_files
import itertools
from multiprocessing.pool import Pool
from functools import partial
from digneapy.generators import DEAGenerator
from digneapy.solvers import best_fit, worst_fit, next_fit, first_fit


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    k: int,
    descriptor: str,
    verbose,
):
    domain = BPPDomain(
        dimension=dimension,
        min_i=20,
        max_i=100,
        max_capacity=150,
        capacity_approach="fixed",
    )
    deig = DEAGenerator(
        pop_size=pop_size,
        offspring_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        k=k,
        repetitions=1,
        descriptor_strategy=descriptor,
    )
    result = deig()
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the BP problem with different solvers using Dominated Novelty Search."
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
        type=int,
        required=True,
        help="Size of the BPP problem.",
        default=120,
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
        [best_fit, first_fit, next_fit, worst_fit],
        [first_fit, best_fit, next_fit, worst_fit],
        [next_fit, best_fit, first_fit, worst_fit],
        [worst_fit, best_fit, first_fit, next_fit],
    ]

    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                dimension=dimension,
                pop_size=population_size,
                generations=generations,
                k=k,
                descriptor=descriptor,
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    features_names = BPPDomain().feat_names if descriptor == "features" else None
    vars_names = ["Q", *[f"w_{i}" for i in range(dimension)]]

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"dns_{descriptor}_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            solvers_names,
            features_names,
            vars_names,
        )
