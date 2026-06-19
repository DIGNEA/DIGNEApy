#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_domain_map_elites.py
@Time    :   2024/06/17 10:46:48
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools
from functools import partial
from multiprocessing.pool import Pool

from digneapy.archives import GridArchive
from digneapy.core import DescriptorPipeline
from digneapy.domains import KnapsackDomain
from digneapy.generators import MapElites
from digneapy.operators import BatchUMut
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    number_of_items: int,
    pop_size: int,
    generations: int,
    verbose,
):
    archive = GridArchive(
        dimensions=(10,) * 101,
        ranges=[(0.0, 10000), *[(1.0, 1000) for _ in range(number_of_items * 2)]],
    )

    domain = KnapsackDomain(
        number_of_items=number_of_items, capacity_approach="evolved"
    )
    map_elites = MapElites(
        domain,
        portfolio=portfolio,
        archive=archive,
        pop_size=pop_size,
        mutation=BatchUMut(),
        generations=generations,
        descriptor_pipe=DescriptorPipeline("instance"),
        repetitions=1,
    )

    result = map_elites(verbose=verbose)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the knapsack problem with different solvers using MapElites."
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
    number_of_items = args.n
    rep = args.repetition
    verbose = args.verbose

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
                number_of_items=number_of_items,
                pop_size=population_size,
                generations=generations,
                verbose=verbose,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    vars_names = ["Q"] + list(
        itertools.chain.from_iterable([
            (f"w_{i}", f"p_{i}") for i in range(number_of_items)
        ])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"mapelites_grid_N_{number_of_items}_target_{result.target}_rep_{rep}",
            result=result,
            variables_names=vars_names,
            descriptor_names=None,
            only_instances=True,
        )
