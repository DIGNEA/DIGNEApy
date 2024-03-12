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
from typing import Dict
from collections import deque
from digneapy.transformers import NN
from digneapy.generator import EIG
from digneapy.solvers.heuristics import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.domains.knapsack import KPDomain
from digneapy.operators.replacement import first_improve_replacement
import numpy as np
import copy


class NSEval:
    def __init__(
        self, features_info: Dict, resolution: int = 20, output_dir: str = None
    ):
        self.resolution = resolution
        self.features_info = features_info
        self.hypercube = [
            np.linspace(start, stop, self.resolution) for start, stop in features_info
        ]
        self.kp_domain = KPDomain(dimension=50, capacity_approach="percentage")
        self.portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
        self.out_dir = output_dir

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
            for solver, instances in generated_instances.items():
                for inst in instances:
                    content = (
                        solver + "," + ",".join(str(f) for f in inst.features) + "\n"
                    )
                    file.write(content)

    def __call__(self, transformer: NN, filename: str = None):
        """This method runs the Novelty Search using a NN as a transformer
        for searching novelty. It generates KP instances for each of the solvers in
        the portfolio [Default, MaP, MiW, MPW] and calculates how many bins of the
        8D-feature hypercube are occupied.

        Args:
            transformer (NN): Transformer to reduce a 8D feature vector into a 2D vector.
            filename (str, optional): Filename to store the instances. Defaults to None.

        Returns:
            int: Number of bins occupied. The maximum value if 8 x R.
        """
        coverage = [set() for _ in range(8)]
        instances = {}
        for i in range(len(self.portfolio)):
            self.portfolio.rotate(i)  # This allow us to change the target on the fly
            eig = EIG(
                10,
                1000,
                domain=self.kp_domain,
                portfolio=self.portfolio,
                t_a=0.5,
                t_ss=0.05,
                k=3,
                repetitions=1,
                descriptor="features",
                replacement=first_improve_replacement,
                transformer=transformer,
            )
            _, solution_set = eig()
            instances[self.portfolio[0].__name__] = copy.copy(solution_set)

            for instance in solution_set:  # For each set of instances
                for i, f in enumerate(instance.features):
                    coverage[i].add(np.digitize(f, self.hypercube[i]))

        f = sum(len(s) for s in coverage)
        self.__save_instances(filename, instances)
        return f


def main():
    R = 20  # Resolution/Number of bins for each of the 8 features
    nn = NN(
        name="NN_transformer_kp_domain.keras",
        input_shape=[8],
        shape=(4, 2),
        activations=("relu", None),
        scale=True,
    )
    weights_113 = [
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
    nn.update_weights(weights_113)
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

    for r in range(10):
        exp_filename = f"instances_nn_best_found_113_cells_rep_{r}.csv"
        print(f"Running repetition: {exp_filename}")
        ns_eval(nn, exp_filename)


if __name__ == "__main__":
    main()
