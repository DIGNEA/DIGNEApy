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
from functools import partial
from multiprocessing.pool import Pool

from digneapy import CVTArchive
from digneapy.domains import BPPDomain
from digneapy.generators import MapElitesGenerator
from digneapy.operators import uniform_one_mutation
from digneapy.solvers import best_fit, first_fit, next_fit, worst_fit
from digneapy.utils import save_results_to_files


def generate_instancess(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    verbose: bool,
):
    domain = BPPDomain(
        dimension=dimension,
        min_i=20,
        max_i=100,
        max_capacity=150,
        capacity_approach="fixed",
    )

    # Create an empty archive with the previous centroids and samples
    # The genotype of the BPP is [capacity, w_i x N]
    # The ranges must be [150, (20, 100)]
    cvt_archive = CVTArchive(
        k=1_000,
        ranges=[(150, 150), *[(20, 100) for _ in range(dimension)]],
        n_samples=100000,
    )
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
        prog="map_elites_cvt_bin_packing",
        description="CVTMAP-Elites for Bin Packing instances",
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
        type=int,
        required=True,
        help="Size of the Bin Packing Problem.",
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

    portfolios = [
        [best_fit, first_fit, next_fit, worst_fit],
        [first_fit, best_fit, next_fit, worst_fit],
        [next_fit, best_fit, first_fit, worst_fit],
        [worst_fit, best_fit, first_fit, next_fit],
    ]

    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instancess,
                dimension=dimension,
                pop_size=population_size,
                generations=generations,
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"map_elites_cvt_bin_packing_N_{dimension}_target_{result.target}_rep_{rep}",
            result=result,
            solvers_names=solvers_names,
            vars_names=["capacity", *[f"w_{i}" for i in range(dimension)]],
            features_names=None,
        )
