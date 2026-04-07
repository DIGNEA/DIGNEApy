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
from multiprocessing.pool import Pool

from digneapy.archives import CVTArchive
from digneapy.domains import TSPDomain
from digneapy.generators import MapElitesGenerator
from digneapy.operators import uniform_one_mutation
from digneapy.solvers import greedy, nneighbour, two_opt
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    descriptor: str,
    verbose: bool,
):
    domain = TSPDomain(dimension=dimension)
    # Create an empty archive with the previous centroids and samples
    ranges = []
    if descriptor == "features":
        ranges = [(dimension * 2, dimension * 2), *[(0.0, 1_000) for _ in range(10)]]
    elif descriptor == "performance":
        ranges = [(0.0, 1.0) for _ in range(len(portfolio))]
    else:  # case instance
        ranges = [(0, 1_000) for _ in range(dimension)]
    cvt_archive = CVTArchive(
        k=1_000,
        ranges=ranges,
        n_samples=100_000,
    )
    map_elites = MapElitesGenerator(
        domain=domain,
        portfolio=portfolio,
        archive=cvt_archive,
        initial_pop_size=pop_size,
        mutation=uniform_one_mutation,
        generations=generations,
        descriptor=descriptor,
        repetitions=1,
    )

    result = map_elites(verbose=verbose)
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the Travelling Salesman Problem with different solvers using CVT-MapElites."
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
    rep = args.repetition
    verbose = args.verbose
    portfolios = [
        [greedy, nneighbour, two_opt],
        [nneighbour, greedy, two_opt],
        [two_opt, greedy, nneighbour],
    ]

    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                dimension=dimension,
                pop_size=population_size,
                generations=generations,
                descriptor=descriptor,
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    vars_names = list(
        itertools.chain.from_iterable([(f"x_{i}", f"y_{i}") for i in range(dimension)])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"map_elites_cvt_{descriptor}_tsp_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            only_genotypes=True,
            only_instances=True,
            solvers_names=solvers_names,
            vars_names=vars_names,
            files_format="parquet",
        )
