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
from digneapy.domains import KnapsackDomain
from digneapy.generators import DEAGenerator
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.utils import save_results_to_files
from digneapy.transformers.neural import KerasNN
import itertools
from multiprocessing.pool import Pool
from functools import partial


def generate_instances(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    k: int,
    verbose,
):
    nn = KerasNN(
        name="NN_transformer_knapsack_domain.keras",
        input_shape=[8],
        shape=(4, 2),
        activations=("relu", None),
        scale=True,
    )
    best_weights = [
        8.46536379866679,
        -7.99534033843463,
        -8.184847366906643,
        -3.6560089672227667,
        -1.6256989543323064,
        -1.184628183269614,
        4.300459150985919,
        -3.6615440892278106,
        -9.669970623875736,
        0.9050152099383979,
        4.824863370241699,
        1.5813289313001333,
        -7.528445500205142,
        -7.434397360115661,
        10.120540012463628,
        2.1120050886615704,
        -11.53730853954561,
        4.546137811050492,
        3.1273592046562158,
        0.6539418604333902,
        6.428198395023706,
        8.081255495437807,
        -7.008946067401477,
        -5.448684569848762,
        1.1877065401955407,
        -0.169390977414007,
        3.7600475572075815,
        -3.118874809383546,
        6.10554459479039,
        -0.035288482418961764,
        1.474049596115826,
        -10.783566230103878,
        16.558651814850613,
        -4.8688398608283725,
        3.0112256185926305,
        0.8507824147793978,
        0.8160047368887384,
        -0.6662407975202618,
        -3.1076433082018604,
        2.650233293599235,
        2.745332952821439,
        -9.285699735622021,
        -7.038376498451783,
        4.057540049804168,
        2.653990161334333,
        13.515364805172545,
    ]
    nn.update_weights(best_weights)

    domain = KnapsackDomain(dimension=dimension, capacity_approach="percentage")
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
        description="Generate instances for the knapsack problem with different solvers using a Neural Network as Transformer."
    )
    parser.add_argument(
        "-n",
        "-number_of_items",
        type=int,
        help="Size of the knapsack problem.",
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
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]
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
    vars_names = ["capacity"] + list(
        itertools.chain.from_iterable([(f"w_{i}", f"p_{i}") for i in range(dimension)])
    )

    for i, result in enumerate(results):
        solvers_names = [p.__name__ for p in portfolios[i]]

        save_results_to_files(
            f"dnsf_NN_N_{dimension}_target_{result.target}_rep_{rep}",
            result,
            solvers_names,
            features_names=None,
            vars_names=vars_names,
        )
