#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bin_packing_novelty_search.py
@Time    :   2025/04/02 15:40:48
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import argparse
from functools import partial
from multiprocessing.pool import Pool

from digneapy import NS, Archive
from digneapy.domains import BPPDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import best_fit, first_fit, next_fit, worst_fit
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: int,
    descriptor: str,
    generations: int = 1000,
    population_size: int = 128,
    k: int = 15,
    archive_threshold: float = 1e-7,
    ss_threshold: float = 1e-7,
    verbose: bool = False,
):
    domain = BPPDomain(
        dimension=dimension,
        min_i=20,
        max_i=100,
        max_capacity=150,
        capacity_approach="fixed",
    )

    eig = EAGenerator(
        pop_size=population_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        novelty_approach=NS(Archive(threshold=archive_threshold), k=k),
        solution_set=Archive(threshold=ss_threshold),
        repetitions=1,
        descriptor_strategy=descriptor,
        replacement=generational_replacement,
    )
    result = eig()
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    expected_dimensions = (120, 240, 560, 1080)
    parser = argparse.ArgumentParser(
        prog="novelty_search_bin_packing",
        description="Bin-Packing Problem instance generator using NS",
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
        type=int,
        help="Size of the BP problem.",
        default=120,
    )
    parser.add_argument(
        "-d", "--descriptor", type=str, default="features", help="Descriptor to use."
    )
    parser.add_argument(
        "-k",
        type=int,
        help="Number of neighbors to use for the NS.",
        default=15,
    )

    parser.add_argument(
        "-p",
        "--population_size",
        default=128,
        type=int,
        help="Number of instances to evolve.",
    )
    parser.add_argument(
        "-g",
        "--generations",
        default=1000,
        type=int,
        help="Number of generations to perform.",
    )
    parser.add_argument(
        "-a",
        "--archive_threshold",
        default=0.489739445237057,
        type=float,
        help="Threshold for the Archive.",
    )
    parser.add_argument(
        "-s",
        "--solution_set_threshold",
        default=0.040663809390192,
        type=float,
        help="Threshold for the Archive.",
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

    portfolios = [
        [best_fit, first_fit, next_fit, worst_fit],
        [first_fit, best_fit, next_fit, worst_fit],
        [next_fit, best_fit, first_fit, worst_fit],
        [worst_fit, best_fit, first_fit, next_fit],
    ]
    args = parser.parse_args()
    descriptor = args.descriptor
    generations = args.generations
    population_size = args.population_size
    dimension = args.n
    k = args.k
    rep = args.repetition
    verbose = args.verbose
    archive_threshold = args.archive_threshold
    solution_set_threshold = args.solution_set_threshold
    pool = Pool(4)
    print(f"Running with {len(portfolios)} portfolios and rep {rep}.")
    
    results = pool.map(
        partial(
            generate_instances,
            dimension=dimension,
            descriptor=descriptor,
            generations=generations,
            population_size=population_size,
            k=k,
            archive_threshold=archive_threshold,
            ss_threshold=solution_set_threshold,
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
            f"best_irace_ns_{descriptor}_bin_packing_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            solvers_names,
            features_names,
            vars_names,
        )
