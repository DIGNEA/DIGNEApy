#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   generate_instancess.py
@Time    :   2024/09/30 09:19:06
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools
from functools import partial
from multiprocessing import Pool, current_process

import numpy as np

from digneapy import CVTArchive, DescriptorKey, DescriptorPipeline
from digneapy.domains import KnapsackDomain
from digneapy.generators import MapElites
from digneapy.operators import BatchUMut
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    descriptor: DescriptorKey,
    seeds: list[np.random.SeedSequence],
    verbose: bool,
):
    domain = KnapsackDomain(dimension=dimension)
    seed = seeds[current_process()._identity[0]]
    # Create an empty archive with the previous centroids and samples
    ranges = []
    if descriptor == "features":
        ranges = [(1.0, 100_000), *[(1.0, 1_000) for _ in range(7)]]
    elif descriptor == "performance":
        ranges = [(1.0, 800_000) for _ in range(len(portfolio))]
    else:  # case instance
        ranges = [(1.0, 100_000), *[(1.0, 1_000) for _ in range(dimension * 2)]]
    cvt_archive = CVTArchive(
        k=1_000,
        ranges=ranges,
        n_samples=100_000,
        seed=seed,
    )
    map_elites = MapElites(
        domain=domain,
        portfolio=portfolio,
        archive=cvt_archive,
        pop_size=pop_size,
        mutation=BatchUMut(),
        generations=generations,
        describe_pipe=DescriptorPipeline(descriptor),
        repetitions=1,
        seed=seed,
    )

    result = map_elites(verbose=verbose)
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the knapsack problem with different solvers using CVT-MapElites."
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
        "-p",
        "--population_size",
        default=128,
        type=int,
        required=True,
        help="Number of instances to evolve.",
    )
    parser.add_argument(
        "-d",
        "--descriptor",
        type=str,
        required=True,
        help="Descriptor to use",
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
    generations = args.generations
    population_size = args.population_size
    dimension = args.n
    descriptor = args.descriptor
    rep = args.repetition
    verbose = args.verbose

    portfolios = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]
    root_sequence = np.random.SeedSequence(1342)
    N_WORKERS = len(portfolios)
    workers_seeds = root_sequence.spawn(N_WORKERS + 1)
    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                dimension=dimension,
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
    vars_names = ["capacity"] + list(
        itertools.chain.from_iterable([(f"w_{i}", f"p_{i}") for i in range(dimension)])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"map_elites_cvt_{descriptor}_knapsack_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            only_genotypes=True,
            only_instances=True,
            solvers_names=solvers_names,
            vars_names=vars_names,
            files_format="parquet",
        )
