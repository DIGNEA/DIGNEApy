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

import copy
from collections import deque

import pandas as pd

from digneapy import NS, Archive, Direction, GridArchive
from digneapy.domains import KnapsackDomain
from digneapy.generators import EAGenerator
from digneapy.operators import first_improve_replacement
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.transformers.neural import TorchNN
from digneapy.transformers.tuner import NNEATuner


def save_best_nn_results(filename, best_nn):
    """Writes the fitness and weights of the best NN found by CMA-ES

    Args:
        filename (str): Filename to sotre the informatio
        best_nn (NN): Best NN found by CMA-ES
    """
    with open(filename, "w") as f:
        chromosome = ",".join((str(w) for w in best_nn))
        fitness = best_nn.fitness.values[0]
        f.write(f"{fitness}\n")
        f.write(f"{chromosome}")


class NSEval:
    """Experiment Code for the Novelty Search with NN transformed space paper for GECCO 2024.
    It receives a iterable of features tuples with the minimum and maximum values for each, an a
    resolution value R to define how many bins per feature we'll be creating.
    This must be called for each transformed at every generation of the CMA-ES algorithm.
    """

    def __init__(self, features_info, resolution: int = 10):
        self.resolution = resolution
        self.features_info = features_info
        self.kp_domain = KnapsackDomain(dimension=50, capacity_approach="percentage")
        self.portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])

    def __call__(self, transformer: TorchNN):
        """This method runs the Novelty Search using a NN as a transformer
        for searching novelty. It generates KP instances for each of the solvers in
        the portfolio [Default, MaP, MiW, MPW] and calculates how many bins of the
        8D-feature hypercube are occupied.

        Args:
            transformer (TorchNN): Transformer to reduce a 8D feature vector into a 2D vector.

        Returns:
            int: Number of bins occupied. The maximum value if 8 x R.
        """
        gen_instances = {
            s.__name__: GridArchive(
                dimensions=(self.resolution,) * 8, ranges=self.features_info
            )
            for s in self.portfolio
        }
        for i in range(len(self.portfolio)):
            self.portfolio.rotate(i)  # This allow us to change the target on the fly
            eig = EAGenerator(
                pop_size=10,
                generations=1000,
                domain=self.kp_domain,
                portfolio=self.portfolio,
                novelty_approach=NS(Archive(threshold=0.5), k=3),
                solution_set=NS(Archive(threshold=0.05), k=1),
                repetitions=1,
                descriptor_strategy="features",
                replacement=first_improve_replacement,
                transformer=transformer,
            )
            _, solution_set = eig()
            print(solution_set)
            if solution_set is not None and len(solution_set) != 0:
                gen_instances[self.portfolio[0].__name__].extend(
                    copy.deepcopy(solution_set)
                )

        for solver, hypercube in gen_instances.items():
            print(f"Solver {solver} -> {hypercube.coverage}")

        combined_coverage = [
            list(hypercube.filled_cells) for hypercube in gen_instances.values()
        ]

        print(combined_coverage)
        return 0


def main():
    R = 10  # Resolution/Number of bins for each of the 8 features
    dimension = 46  # Number of weights of the NN for KP
    nn = TorchNN(
        name="NN_transformer_kp_domain.keras",
        input_size=8,
        shape=(4,),
        output_size=2,
    )

    # KP Features information extracted from previously generated instances
    # features_info = [
    #     (711, 30000),
    #     (890, 1000),
    #     (860, 1000.0),
    #     (1.0, 200),
    #     (1.0, 230.0),
    #     (0.10, 12.0),
    #     (400, 610),
    #     (240, 330),
    # ]
    features_info = [
        (1, 30000),  # Capacity
        (1, 1000),  # Max Profit
        (1, 1000),  # Max Weight
        (1, 1000),  # Min Profit
        (1, 1000),  # Min Weight
        (0.0, 100.0),  # Average Efficiency
        (0.0, 1000),  # Mean of vars
        (0.0, 500.0),  # Std
    ]
    # NSEval is the evaluation/fitness function used to measure the NNs in CMA-Es
    ns_eval = NSEval(features_info, resolution=R)
    # Custom CMA-ES derived from DEAP to evolve NNs weights
    cma_es = NNEATuner(
        dimension=dimension,
        direction=Direction.MAXIMISE,
        lambda_=64,
        generations=250,
        transformer=nn,
        eval_fn=ns_eval,
    )
    best_nn, population, logbook = cma_es()
    # Save the scores and the weights
    save_best_nn_results("NN_best_score_and_weights.csv", best_nn)
    # Save the model itself
    nn.update_weights(best_nn)
    nn.save("NN_best_transformer_found_kp_domain.keras")
    for i, ind in enumerate(population):
        nn.update_weights(ind)
        nn.save(f"NN_final_population_{i}_transformer_kp_domain.keras")

    # Saving logbook to CSV
    df = pd.DataFrame(logbook)
    df.to_csv("CMAES_logbook_for_NN_transformers.csv", index=False)


if __name__ == "__main__":
    main()
