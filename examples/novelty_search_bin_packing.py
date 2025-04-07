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


def save_instances(filename, results, dimension):
    bpp_features = ",mean,std,median,max,min,tiny,small,medium,large,huge"
    header = "target,N,capacity,"
    header += ",".join([f"w_{i}" for i in range(dimension)])
    if descriptor == "features":
        header += bpp_features

    with open(filename, "w") as file:
        file.write(f"{header}\n")
        for solver, instances in results:
            for instance in instances:
                assert len(instance) == dimension + 1
                vars = ",".join(str(x) for x in instance)
                if descriptor == "features":
                    features = ",".join(str(f) for f in instance.features)
                    file.write(f"{solver},{dimension},{vars},{features}\n")
                else:
                    file.write(f"{solver},{dimension},{vars}\n")


def generate_instances_for_target(
    portfolio,
    dimension: int,
    descriptor: str,
    generations: int = 1000,
    population_size: int = 128,
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
        novelty_approach=NS(Archive(threshold=1e-7), k=15),
        solution_set=Archive(threshold=1e-7),
        repetitions=1,
        descriptor_strategy=descriptor,
        replacement=generational_replacement,
    )
    _, solution_set = eig()
    print(f"Solver: {portfolio[0].__name__} finised")
    return (portfolio[0].__name__, solution_set)


if __name__ == "__main__":
    expected_dimensions = (120, 240, 560, 1080)
    parser = argparse.ArgumentParser(
        prog="bin_packing_ns",
        description="Bin-Packing Problem instance generator using NS",
    )
    parser.add_argument(
        "dimension",
        choices=expected_dimensions,
        type=int,
        help="dimension of the Bin Packing instances",
    )
    parser.add_argument(
        "repetition",
        choices=list(range(10)),
        type=int,
        help="Number of the run to append in the final CSV file",
    )
    parser.add_argument(
        "descriptor",
        choices=("features", "performance", "instance"),
        type=str,
        help="Descriptor for the NS",
    )
    args = parser.parse_args()
    n_run = args.repetition
    dimension = args.dimension
    descriptor = args.descriptor

    portfolios = [
        [best_fit, first_fit, next_fit, worst_fit],
        [first_fit, best_fit, next_fit, worst_fit],
        [next_fit, best_fit, first_fit, worst_fit],
        [worst_fit, best_fit, first_fit, next_fit],
    ]

    print(
        f"Running experiment #{n_run + 1} for dimension: {dimension} with descriptor: {descriptor}"
    )
    pool = Pool(4)
    results = pool.map(
        partial(
            generate_instances_for_target,
            dimension=dimension,
            descriptor=descriptor,
            generations=1000,
            population_size=128,
        ),
        portfolios,
    )
    pool.close()
    pool.join()
    save_instances(
        f"bpp_ns{descriptor[0]}_N_{dimension}_{n_run}.csv",
        results=results,
        dimension=dimension,
    )
