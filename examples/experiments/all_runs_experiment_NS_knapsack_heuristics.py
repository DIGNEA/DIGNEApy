#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   experiment_NS_knapsack_heuristics.py
@Time    :   2026/06/17 12:21:22
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import argparse
import concurrent.futures
import itertools
from typing import Sequence

import numpy as np
from utils import make_seed_sequences

from digneapy import DescriptorKey, DescriptorPipeline, Solver, UnstructuredArchive
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
    portfolio: Sequence[Solver],
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
        "-r",
        "--repetitions",
        type=int,
        required=True,
        help="Number of repetitions to perform.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        required=True,
        help="Number of workers to launch at the same time.",
    )
    parser.add_argument(
        "seed",
        type=int,
        default=None,
        help="Master seed to generate others",
    )

    args = parser.parse_args()
    descriptor = args.descriptor
    generations = args.generations
    population_size = args.population_size
    archive_threshold = args.archive_threshold
    solution_set_threshold = args.solution_set_threshold
    number_of_items = args.n
    k_neighbours = args.k
    repetitions = args.repetitions
    n_workers = args.workers
    master_seed = args.seed

    # The solvers are fixed to Knapsack Heuristics available
    portfolios: list[list[Solver]] = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]

    # Combinations of solvers / repetitions to perform
    combinations = itertools.product(range(repetitions), enumerate(portfolios))
    seed_sequences = make_seed_sequences(
        master_seed=master_seed, n_repetitions=repetitions, n_solvers=len(portfolios)
    )
    features_names = (
        KnapsackDomain().features_names if descriptor == "features" else None
    )
    vars_names = ["capacity"] + list(
        itertools.chain.from_iterable([
            (f"w_{i}", f"p_{i}") for i in range(number_of_items)
        ])
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for repetition, (portfolio_idx, portfolio) in combinations:
            # Extract the seed for this repetition/target solver combination
            seed = seed_sequences[(repetition, portfolio_idx)]
            fut = executor.submit(
                generate_instances,
                portfolio=portfolio,
                number_of_items=number_of_items,
                pop_size=population_size,
                generations=generations,
                archive_threshold=archive_threshold,
                ss_threshold=solution_set_threshold,
                k=k_neighbours,
                descriptor=descriptor,
                seed=seed,
            )
            futures[fut] = (repetition, portfolio_idx)
            print(f"Combination {portfolio[0].__name__}/{repetition} submitted.")

        for fut in concurrent.futures.as_completed(futures):
            try:
                repetition, portfolio_index = futures[fut]
                result = fut.result()
                save_results_to_files(
                    f"novelty_search_{descriptor}_N_{number_of_items}_target_{result.solvers[0]}_rep_{repetition}_master_seed_{master_seed}",
                    result=result,
                    variables_names=vars_names,
                    descriptor_names=features_names,
                    only_instances=True,
                )
            except Exception as e:
                print(e)
