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
import concurrent.futures
import itertools
from typing import Optional

import numpy as np

from digneapy import DescriptorKey, DescriptorPipeline
from digneapy.domains import KnapsackDomain
from digneapy.generators import Dominated
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    number_of_items: np.uint32,
    pop_size: np.uint32,
    generations: np.uint32,
    k: np.uint32,
    descriptor: DescriptorKey,
    seed: Optional[int | np.random.SeedSequence],
    verbose,
):
    domain = KnapsackDomain(
        number_of_items=number_of_items, capacity_approach="evolved"
    )
    deig = Dominated(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        k=k,
        repetitions=np.uint16(1),
        descriptor_pipe=DescriptorPipeline(descriptor),
        seed=seed,
    )
    result = deig()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the knapsack problem with different solvers using Dominated Novelty Search."
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
        "-d", "--descriptor", type=str, required=True, help="Descriptor to use."
    )
    parser.add_argument(
        "-k",
        type=np.uint32,
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
        type=np.uint32,
        required=True,
        help="Number of generations to perform.",
    )
    parser.add_argument(
        "-r", "--repetition", type=np.uint32, required=True, help="Repetition index."
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
    number_of_items = args.n
    k = args.k
    rep = args.repetition
    verbose = args.verbose
    portfolios = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]
    features_names = (
        KnapsackDomain().features_names if descriptor == "features" else None
    )
    vars_names = ["Q"] + list(
        itertools.chain.from_iterable([
            (f"w_{i}", f"p_{i}") for i in range(number_of_items)
        ])
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for portfolio in portfolios:
            futures.append(
                executor.submit(
                    generate_instances,
                    portfolio=portfolio,
                    number_of_items=number_of_items,
                    pop_size=population_size,
                    generations=generations,
                    k=k,
                    descriptor=descriptor,
                    seed=None,
                    verbose=verbose,
                )
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                save_results_to_files(
                    f"dns_{descriptor}_N_{number_of_items}_target_{result.solvers[0]}_rep_{rep}",
                    result=result,
                    variables_names=vars_names,
                    descriptor_names=features_names,
                    only_instances=True,
                )
            except Exception as exc:
                print(f"Exception generated: {exc}")
