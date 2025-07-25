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

from digneapy.domains import TSPDomain
from digneapy.generators import DEAGenerator
from digneapy.solvers import greedy, nneighbour, two_opt
from digneapy.transformers.neural import KerasNN
from digneapy.utils import save_results_to_files


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
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
    eig = DEAGenerator(
        pop_size=pop_size,
        offspring_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        repetitions=1,
        k=k,
        descriptor_strategy="features",
        transformer=nn,
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
            f"TSP_dns_NN_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            solvers_names,
            features_names,
            vars_names,
        )
