#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   novelty_search_knapsack.py
@Time    :   2023/11/10 14:09:41
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import argparse
import concurrent.futures
import itertools

import numpy as np

from digneapy import DescriptorKey, DescriptorPipeline, UnstructuredArchive
from digneapy.domains import KnapsackDomain
from digneapy.generators import Evolutionary
from digneapy.operators import UCX, Generational, UMut
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: np.uint32,
    pop_size: np.uint32,
    generations: np.uint32,
    archive_threshold: float,
    ss_threshold: float,
    k: np.uint32,
    descriptor: DescriptorKey,
    verbose: bool,
):

    domain = KnapsackDomain(dimension)
    eig = Evolutionary(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        archive=UnstructuredArchive(threshold=archive_threshold, k=k),
        solution_set=UnstructuredArchive(threshold=ss_threshold, k=np.uint32(1)),
        repetitions=np.uint16(1),
        descriptor_pipe=DescriptorPipeline(descriptor),
        replacement=Generational(),
        crossover=UCX(),
        mutation=UMut(),
    )
    result = eig(verbose=verbose)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the knapsack problem with different solvers."
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
        type=np.uint32,
        required=True,
        help="Size of the knapsack problem.",
        default=50,
    )
    parser.add_argument(
        "-d",
        "--descriptor",
        type=str,
        required=True,
        help="Descriptor to use.",
    )
    parser.add_argument(
        "-k",
        type=np.uint32,
        required=True,
        help="Number of neighbors to use for the NS.",
        default=3,
    )
    parser.add_argument(
        "-a",
        "--archive_threshold",
        default=3.0,
        type=float,
        required=True,
        help="Threshold for the Archive.",
    )
    parser.add_argument(
        "-s",
        "--solution_set_threshold",
        default=3.0,
        type=float,
        required=True,
        help="Threshold for the Archive.",
    )
    parser.add_argument(
        "-p",
        "--population_size",
        default=128,
        type=np.uint32,
        required=True,
        help="Number of instances to evolve.",
    )
    parser.add_argument(
        "-g",
        "--generations",
        default=1000,
        type=np.uint32,
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

    features_names = KnapsackDomain().feat_names if descriptor == "features" else None
    vars_names = ["capacity"] + list(
        itertools.chain.from_iterable([(f"w_{i}", f"p_{i}") for i in range(dimension)])
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(len(portfolios)):
            f = executor.submit(
                generate_instances,
                portfolio=portfolios[i],
                dimension=dimension,
                pop_size=population_size,
                generations=generations,
                archive_threshold=archive_threshold,
                ss_threshold=solution_set_threshold,
                k=k,
                descriptor=descriptor,
                verbose=verbose,
            )
            futures.append(f)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                save_results_to_files(
                    f"ns_{descriptor}_N_{dimension}_target_{result.solvers[0]}_rep_{rep}",
                    result,
                    variables_names=vars_names,
                    files_format="csv",
                )
            except Exception as e:
                print(e)
