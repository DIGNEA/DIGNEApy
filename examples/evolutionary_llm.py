#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   evolutionary_llm.py
@Time    :   2026/05/11 14:27:48
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import argparse

from digneapy import DESCRIPTORS, NS, Archive
from digneapy.domains import TSPDomain
from digneapy.generators import LLMEvolutionary
from digneapy.operators import generational_replacement
from digneapy.solvers import greedy, nneighbour, two_opt


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    archive_threshold: float,
    ss_threshold: float,
    k: int,
    descriptor: DESCRIPTORS,
    verbose,
):
    domain = TSPDomain(dimension=dimension)
    eig = LLMEvolutionary(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        novelty_approach=NS(Archive(threshold=archive_threshold), k=k),
        solution_set=Archive(threshold=ss_threshold),
        repetitions=1,
        describe_by=descriptor,
        replacement=generational_replacement,
        model="qwen/qwen3.6-35b-a3b",
    )

    result = eig()
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the Travelling Salesman Problem with different solvers."
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
        "-k",
        type=int,
        required=True,
        help="Number of neighbors to use for the NS.",
        default=3,
    )
    parser.add_argument(
        "-a",
        "--archive_threshold",
        default=1e-7,
        type=float,
        help="Threshold for the Archive.",
    )
    parser.add_argument(
        "-s",
        "--solution_set_threshold",
        default=1e-10,
        type=float,
        help="Threshold for the Archive.",
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
    archive_threshold = args.archive_threshold
    solution_set_threshold = args.solution_set_threshold
    dimension = args.n
    k = args.k
    rep = args.repetition
    verbose = args.verbose
    portfolios = [
        [greedy, nneighbour, two_opt],
        [nneighbour, greedy, two_opt],
        [two_opt, greedy, nneighbour],
    ]
    print(
        f"Running with parameters:\ndimension={dimension}, k={k}, archive_threshold={archive_threshold}, solution_set_threshold={solution_set_threshold}, population_size={population_size}, generations={generations}, descriptor={descriptor}, verbose={verbose}"
    )

    results = generate_instances(
        portfolios[0],
        50,
        128,
        1000,
        archive_threshold,
        solution_set_threshold,
        k,
        descriptor="instance",
        verbose=True,
    )
    print(results)
    # with Pool(4) as pool:
    #     results = pool.map(
    #         partial(
    #             generate_instances,
    #             dimension=dimension,
    #             pop_size=population_size,
    #             generations=generations,
    #             archive_threshold=archive_threshold,
    #             ss_threshold=solution_set_threshold,
    #             k=k,
    #             descriptor=descriptor,
    #             verbose=verbose,
    #         ),
    #         portfolios,
    #     )

    # pool.close()
    # pool.join()
    # features_names = TSPDomain().feat_names if descriptor == "features" else None
    # vars_names = list(
    #     itertools.chain.from_iterable([(f"x_{i}", f"y_{i}") for i in range(dimension)])
    # )

    # for i, result in enumerate(results):
    #     solvers_names = [p.__name__ for p in portfolios[i]]

    #     save_results_to_files(
    #         f"ns_{descriptor}_N_{dimension}_target_{result.target}_rep_{rep}",
    #         result,
    #         solvers_names,
    #         features_names,
    #         vars_names,
    #     )
