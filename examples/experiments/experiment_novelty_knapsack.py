#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   experiment_novelty_knapsack.py
@Time    :   2026/07/10 13:00:01
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools
from pathlib import Path
from typing import Sequence

import numpy as np
from utils import make_seed_sequences

from digneapy.archives import UnstructuredArchive
from digneapy.core import DescriptorKey, DescriptorPipeline, Solver
from digneapy.domains import KnapsackDomain
from digneapy.generators import Evolutionary
from digneapy.lab import GenerationExperiment, RunConfig
from digneapy.operators import UCX, BinarySelection, Generational, UMut
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)


def run(
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
    # Each run has its independent domain and generators
    # to avoid random generation issues

    # Seed here is the root seed for this job
    # We need to generate several seeds for the
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
        cxrate=0.5,
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
        "-x",
        "--seed",
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
    root_seed = args.seed

    # The solvers are fixed to Knapsack Heuristics available
    portfolios: list[list[Solver]] = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]
    if root_seed is None:
        entropy = np.random.SeedSequence(None).entropy
        root_seed = np.random.SeedSequence(entropy=entropy)

    # Combinations of solvers / repetitions to perform
    combinations = itertools.product(range(repetitions), enumerate(portfolios))
    seed_sequences = make_seed_sequences(
        root_seed=root_seed, n_repetitions=repetitions, n_solvers=len(portfolios)
    )
    runs_to_do = []
    experiment_name = "NoveltySearch_Knapsack_Sample"
    base_dir = Path(__file__).parent
    features_names = (
        KnapsackDomain().features_names if descriptor == "features" else None
    )
    vars_names = ["capacity"] + list(
        itertools.chain.from_iterable([
            (f"w_{i}", f"p_{i}") for i in range(number_of_items)
        ])
    )
    for repetition, (portfolio_idx, portfolio) in combinations:
        seed = seed_sequences[(repetition, portfolio_idx)]
        run_name = f"{portfolio[0].__name__}_rep_{repetition}"
        runs_to_do.append(
            RunConfig(
                call_fn=run,
                name=run_name,
                kwargs={
                    "portfolio": portfolio,
                    "number_of_items": number_of_items,
                    "pop_size": population_size,
                    "generations": generations,
                    "archive_threshold": archive_threshold,
                    "ss_threshold": solution_set_threshold,
                    "k": k_neighbours,
                    "descriptor": descriptor,
                    "seed": seed,
                },
                save_kwargs={
                    "files_format": "parquet",
                    "variables_names": vars_names,
                    "descriptor_names": features_names,
                    "only_instances": True,
                    "lazily": True,
                },
            )
        )
    experiment = GenerationExperiment(
        experiment_name=experiment_name,
        base_dir=base_dir,
        runs_to_do=runs_to_do,
        max_workers=n_workers,
        root_seed=root_seed,
    )
    results = experiment()
    print(results)
