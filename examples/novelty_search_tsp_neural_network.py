#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   novelty_search_tsp_neural_network.py
@Time    :   2025/04/29 14:59:33
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools
import multiprocessing as mp
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

from digneapy import NS, Archive
from digneapy.domains import TSPDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import greedy, nneighbour, two_opt
from digneapy.transformers.neural import KerasNN
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    archive_threshold: float,
    ss_threshold: float,
    k: int,
    verbose,
):
    nn = KerasNN(
        name="NN_transformer_TSP.keras",
        input_shape=[11],
        shape=(5, 2),
        activations=("relu", None),
        scale=True,
    )
    best_weights = np.load(Path(__file__).with_name("tsp_NN_weights_N_50_2D_best.npy"))
    nn.update_weights(best_weights)
    domain = TSPDomain(dimension=dimension)
    eig = EAGenerator(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        novelty_approach=NS(Archive(threshold=archive_threshold), k=k),
        solution_set=Archive(threshold=ss_threshold),
        repetitions=1,
        descriptor_strategy="features",
        transformer=nn,
        replacement=generational_replacement,
    )

    result = eig()
    if verbose:
        print(f"Target: {result.target} completed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances for the TSP with different solvers using a Neural Network as Transformer."
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
        type=int,
        help="Size of the TSP problem.",
        default=50,
    )

    parser.add_argument(
        "-k",
        type=int,
        help="Number of neighbors to use for the NS.",
        default=15,
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
        default=1e-7,
        type=float,
        help="Threshold for the Archive.",
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
    mp.set_start_method("spawn", force=True)

    with Pool(4) as pool:
        results = pool.map(
            partial(
                generate_instances,
                dimension=dimension,
                pop_size=population_size,
                generations=generations,
                archive_threshold=archive_threshold,
                ss_threshold=solution_set_threshold,
                k=k,
                verbose=True,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    features_names = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius".split(
        ","
    )
    vars_names = list(
        itertools.chain.from_iterable([(f"x_{i}", f"y_{i}") for i in range(dimension)])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"TSP_ns_NN_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            solvers_names,
            features_names,
            vars_names,
        )
