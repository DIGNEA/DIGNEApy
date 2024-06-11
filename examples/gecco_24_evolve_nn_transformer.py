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

from collections import deque

import numpy as np
import pandas as pd

from digneapy.archives import Archive
from digneapy.domains.knapsack import KPDomain
from digneapy.generators import EIG
from digneapy.operators.replacement import first_improve_replacement
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.transformers import KerasNN, NNTuner


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

    def __init__(self, features_info, resolution: int = 20):
        self.resolution = resolution
        self.features_info = features_info
        self.hypercube = [
            np.linspace(start, stop, self.resolution) for start, stop in features_info
        ]
        self.kp_domain = KPDomain(dimension=50, capacity_approach="percentage")
        self.portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])

    def __save_instances(self, filename, generated_instances):
        """Writes the generated instances into a CSV file

        Args:
            filename (str): Filename
            generated_instances (iterable): Iterable of instances
        """
        features = [
            "target",
            "capacity",
            "max_p",
            "max_w",
            "min_p",
            "min_w",
            "avg_eff",
            "mean",
            "std",
        ]
        with open(filename, "w") as file:
            file.write(",".join(features) + "\n")
            for solver, descriptors in generated_instances.items():
                for desc in descriptors:
                    content = solver + "," + ",".join(str(f) for f in desc) + "\n"
                    file.write(content)

    def __call__(self, transformer: KerasNN, filename: str = None):
        """This method runs the Novelty Search using a KerasNN as a transformer
        for searching novelty. It generates KP instances for each of the solvers in
        the portfolio [Default, MaP, MiW, MPW] and calculates how many bins of the
        8D-feature hypercube are occupied.

        Args:
            transformer (KerasNN): Transformer to reduce a 8D feature vector into a 2D vector.
            filename (str, optional): Filename to store the instances. Defaults to None.

        Returns:
            int: Number of bins occupied. The maximum value if 8 x R.
        """
        gen_instances = {s.__name__: [] for s in self.portfolio}
        for i in range(len(self.portfolio)):
            self.portfolio.rotate(i)  # This allow us to change the target on the fly
            eig = EIG(
                pop_size=10,
                generations=1000,
                domain=self.kp_domain,
                portfolio=self.portfolio,
                archive=Archive(threshold=0.5),
                s_set=Archive(threshold=0.05),
                k=3,
                repetitions=1,
                descriptor="features",
                replacement=first_improve_replacement,
                transformer=transformer,
            )
            archive, solution_set = eig()
            descriptors = [list(i.descriptor) for i in solution_set]
            gen_instances[self.portfolio[0].__name__].extend(descriptors)
        if any(len(sequence) != 0 for sequence in gen_instances.values()):
            self.__save_instances(filename, gen_instances)

        # Here we gather all the instances together to calculate the metric
        coverage = {k: set() for k in range(8)}
        for (
            solver,
            descriptors,
        ) in gen_instances.items():  # For each set of instances
            for desc in descriptors:  # For each descriptor in the set
                for i, f in enumerate(desc):  # Location of the ith feature
                    coverage[i].add(np.digitize(f, self.hypercube[i]))

        f = sum(len(s) for s in coverage.values())
        return f


def main():
    R = 20  # Resolution/Number of bins for each of the 8 features
    dimension = 118  # Number of weights of the NN for KP
    nn = KerasNN(
        name="NN_transformer_kp_domain.keras",
        input_shape=[8],
        shape=(8, 4, 2),
        activations=("relu", "relu", None),
    )
    # KP Features information extracted from previously generated instances
    features_info = [
        (711, 30000),
        (890, 1000),
        (860, 1000.0),
        (1.0, 200),
        (1.0, 230.0),
        (0.10, 12.0),
        (400, 610),
        (240, 330),
    ]
    # NSEval is the evaluation/fitness function used to measure the NNs in CMA-Es
    ns_eval = NSEval(features_info, resolution=R)
    # Custom CMA-ES derived from DEAP to evolve NNs weights
    cma_es = NNTuner(
        dimension=dimension,
        direction="maximise",
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
