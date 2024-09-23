#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   evolve_nn_full_instances.py
@Time    :   2024/07/24 10:34:49
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import copy

import pandas as pd

from digneapy import Archive, Direction, GridArchive
from digneapy.domains import KnapsackDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import default_kp, map_kp, miw_kp
from digneapy.transformers.neural import KerasNN
from digneapy.transformers.tuner import NNTuner


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


class EvalNN:
    def __init__(self, ranges, resolution: int = 20):
        self.resolution = resolution
        self.ranges = ranges

        self.kp_domain = KnapsackDomain(dimension=50, capacity_approach="percentage")

    def __call__(self, transformer: KerasNN):
        """This method runs the Novelty Search using a KerasNN as a transformer
        for searching novelty. It generates KP instances for each of the solvers in
        the portfolio [Default, MaP, MiW].
        Evolves the instances using the FULL instance as its own descriptor and then
        calculate the transformed descriptor using the ``transformer'' KerasNN object
        which takes a 101 array of numbers and produces a 2D floating point number descriptor.
        The coverage is measure into a 2D Hypercube defined by the AE from our previous work.

        Args:
            transformer (KerasNN): Transformer to reduce a 8D feature vector into a 2D vector.

        Returns:
            int: Number of cells occupied
        """
        hypercube = GridArchive(dimensions=(self.resolution,) * 8, ranges=self.ranges)
        portfolios = [
            [default_kp, map_kp, miw_kp],
            [map_kp, default_kp, miw_kp],
            [miw_kp, default_kp, map_kp],
        ]
        for portfolio in portfolios:
            eig = EAGenerator(
                pop_size=10,
                generations=1000,
                domain=self.kp_domain,
                portfolio=portfolio,
                archive=Archive(threshold=0.5),
                s_set=Archive(threshold=0.05),
                k=3,
                repetitions=1,
                descriptor="instance",
                replacement=generational_replacement,
                transformer=transformer,
            )
            _, solution_set = eig()
            if len(solution_set) != 0:
                hypercube.extend(copy.deepcopy(solution_set))

        return len(hypercube)


def main():
    R = 20  # Resolution/Number of bins for each of the 8 features
    dimension = 5202  # Number of weights of the NN for this architecture
    nn = KerasNN(
        name="NN_transformer_for_N_50_to_2D_kp_domain.keras",
        input_shape=(101,),
        shape=(50, 2),
        activations=("relu", "relu", None),
    )
    # Hypercube boundaries based on the results from AE
    # https://colab.research.google.com/drive/1b392jz3syTJiehSCt_Tf72_D0OkmVVe5?hl=es
    ranges = [(0.0, 25.0), (0.0, 60.0)]
    # EvalNN is the evaluation/fitness function used to measure the NNs in CMA-Es
    ns_eval = EvalNN(ranges, resolution=R)
    # Custom CMA-ES derived from DEAP to evolve NNs weights
    cma_es = NNTuner(
        dimension=dimension,
        direction=Direction.MAXIMISE,
        lambda_=64,
        generations=1000,
        transformer=nn,
        eval_fn=ns_eval,
    )
    best_nn, population, logbook = cma_es()
    # Save the scores and the weights
    save_best_nn_results(
        "NN_transformer_for_N_50_to_2D_kp_domain_best_score_and_weights.csv", best_nn
    )
    # Save the model itself
    nn.update_weights(best_nn)
    nn.save("NN_best_transformer_found_for_N_50_to_2D_kp_domain.keras")
    for i, ind in enumerate(population):
        nn.update_weights(ind)
        nn.save(
            f"NN_final_population_{i}_transformer_found_for_N_50_to_2D_kp_domain.keras"
        )

    # Saving logbook to CSV
    df = pd.DataFrame(logbook)
    df.to_csv(
        "CMAES_logbook_for_NN_transformers_for_N_50_to_2D_kp_domain.csv", index=False
    )


if __name__ == "__main__":
    main()
