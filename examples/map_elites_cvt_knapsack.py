#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   generate_instances.py
@Time    :   2024/09/30 09:19:06
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import argparse
from digneapy import CVTArchive
from digneapy.domains import KnapsackDomain
from digneapy.generators import MapElitesGenerator
from digneapy.operators import uniform_one_mutation
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.utils import save_results_to_files
import itertools
from multiprocessing.pool import Pool
from functools import partial


def generate_instances(
    portfolio,
    cvt_archive: CVTArchive,
    domain: KnapsackDomain,
    pop_size: int,
    generations: int,
    verbose: bool,
):
    map_elites = MapElitesGenerator(
        domain,
        portfolio=portfolio,
        archive=cvt_archive,
        initial_pop_size=pop_size,
        mutation=uniform_one_mutation,
        generations=generations,
        descriptor="instance",
        repetitions=1,
    )
    result = map_elites()
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
    rep = args.repetition
    verbose = args.verbose

    domain = KnapsackDomain(dimension=dimension)
    # Create an empty archive with the previous centroids and samples
    cvt_archive = CVTArchive(
        k=1_000,
        ranges=[(1.0, 10_000), *[(1.0, 1_000) for _ in range(dimension * 2)]],
        n_samples=100000,
    )
    portfolios = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]

    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                domain=domain,
                pop_size=population_size,
                generations=generations,
                cvt_archive=cvt_archive,
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    vars_names = ["Q"] + list(
        itertools.chain.from_iterable([(f"w_{i}", f"p_{i}") for i in range(dimension)])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"nsf_N_{dimension}_target_{result.target}_rep_{rep}",
            result=result,
            solvers_names=solvers_names,
            vars_names=vars_names,
            features_names=None,
        )
