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

from digneapy.archives import UnstructuredArchive
from digneapy.core import DescriptorKey, DescriptorPipeline
from digneapy.domains import KnapsackDomain
from digneapy.generators import Evolutionary
from digneapy.operators import UCX, BinarySelection, Generational, UMut
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    number_of_items: np.uint32,
    pop_size: np.uint32,
    generations: np.uint32,
    archive_threshold: float,
    ss_threshold: float,
    k: np.uint32,
    descriptor: DescriptorKey,
    seed: np.random.SeedSequence,
):

    # Seed here is the master seed for this job
    # We need to generate several need seeds for the
    # components of the experiment: domain, EA, operators, etc.
    domain_seed, generator_seed, cx_seed, mut_seed, sel_seed, rep_seed = seed.spawn(6)
    domain = KnapsackDomain(number_of_items=number_of_items, seed=domain_seed)
    generator = Evolutionary(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        archive=UnstructuredArchive(novelty_threshold=archive_threshold, k=k),
        solution_set=UnstructuredArchive(novelty_threshold=ss_threshold, k=1),
        repetitions=1,  # Portfolio of determinist heuristics
        descriptor_pipe=DescriptorPipeline(descriptor),
        cxrate=0.85,
        mutrate=(1.0 / number_of_items),
        selection=BinarySelection(seed=sel_seed),
        crossover=UCX(seed=cx_seed),
        mutation=UMut(seed=mut_seed),
        replacement=Generational(seed=rep_seed),
        seed=generator_seed,
    )
    result = generator()
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
        "--seed",
        type=int,
        default=13,
        help="Seed for random number generation",
    )
    args = parser.parse_args()
    descriptor = args.descriptor
    generations = args.generations
    population_size = args.population_size
    archive_threshold = args.archive_threshold
    solution_set_threshold = args.solution_set_threshold
    number_of_items = args.n
    k = args.k
    master_seed = args.seed
    portfolios = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]

    features_names = (
        KnapsackDomain().features_names if descriptor == "features" else None
    )
    vars_names = ["capacity"] + list(
        itertools.chain.from_iterable([
            (f"w_{i}", f"p_{i}") for i in range(number_of_items)
        ])
    )
    max_workers = len(portfolios)
    master_seed = np.random.SeedSequence(entropy=master_seed)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for portfolio in portfolios:
            fut = executor.submit(
                generate_instances,
                portfolio=portfolio,
                number_of_items=number_of_items,
                pop_size=population_size,
                generations=generations,
                archive_threshold=archive_threshold,
                ss_threshold=solution_set_threshold,
                k=k,
                descriptor=descriptor,
                seed=master_seed,
            )
            futures[fut] = portfolio[0].__name__

        for fut in concurrent.futures.as_completed(futures):
            try:
                target = futures[fut]
                result = fut.result()
                save_results_to_files(
                    filename_pattern=f"novelty_search_{descriptor}_N_{number_of_items}_target_{target}_seed_{master_seed.entropy}",
                    result=result,
                    variables_names=vars_names,
                    descriptor_names=features_names,
                    only_instances=True,
                    lazily=True,
                )
            except Exception as e:
                print(e)
