#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   nn_transformer_gecco_23.py
@Time    :   2023/11/10 14:09:41
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools
from functools import partial
from multiprocessing import Pool, current_process

import numpy as np

from digneapy import DescriptorKey, DescriptorPipeline
from digneapy.domains import KnapsackDomain
from digneapy.generators import Dominated
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    k: int,
    descriptor: DescriptorKey,
    seeds: list[np.random.SeedSequence],
    verbose,
):
    domain = KnapsackDomain(dimension, capacity_approach="evolved")
    seed = seeds[current_process()._identity[0]]
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
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the knapsack problem with different solvers using Dominated Novelty Search."
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
        type=int,
        required=True,
        help="Size of the knapsack problem.",
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
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]
    root_sequence = np.random.SeedSequence(4213)
    workers_seeds = root_sequence.spawn(len(portfolios) + 1)

    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                dimension=dimension,
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
    features_names = KnapsackDomain().feat_names if descriptor == "features" else None
    vars_names = ["Q"] + list(
        itertools.chain.from_iterable([(f"w_{i}", f"p_{i}") for i in range(dimension)])
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
