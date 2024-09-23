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
import itertools
import sys

from digneapy import Archive
from digneapy.domains import KnapsackDomain
from digneapy.generators import EAGenerator
from digneapy.operators import first_improve_replacement
from digneapy.solvers import default_kp, map_kp, miw_kp
from digneapy.transformers.neural import KerasNN


def save_instances(filename, generated_instances):
    """Writes the generated instances into a CSV file

    Args:
        filename (str): Filename
        generated_instances (iterable): Iterable of instances
    """
    features = [
        "capacity",
        "max_p",
        "max_w",
        "min_p",
        "min_w",
        "avg_eff",
        "mean",
        "std",
    ]
    header = (
        ["target"]
        + ["x_0", "x_1"]
        + features
        + list(itertools.chain.from_iterable([(f"w_{i}", f"p_{i}") for i in range(50)]))
    )

    with open(filename, "w") as file:
        file.write(",".join(header) + "\n")
        for solver, instances in generated_instances.items():
            for inst in instances:
                content = [solver, inst.descriptor, inst.features, inst[1:]]
                file.write(
                    ",".join(str(x) for x in itertools.chain.from_iterable(content))
                    + "\n"
                )


def generate_instances(transformer: KerasNN):
    """This method runs the Novelty Search using a NN as a transformer
    for searching novelty. It generates KP instances for each of the solvers in
    the portfolio [Default, MaP, MiW, MPW] and calculates how many bins of the
    8D-feature hypercube are occupied.

    Args:
        transformer (KerasNN): Transformer to reduce a 8D feature vector into a 2D vector.
        filename (str, optional): Filename to store the instances. Defaults to None.

    Returns:
        int: Number of bins occupied. The maximum value if 8 x R.
    """
    kp_domain = KnapsackDomain(dimension=50, capacity_approach="percentage")
    portfolios = [
        [default_kp, map_kp, miw_kp],
        [map_kp, default_kp, miw_kp],
        [miw_kp, default_kp, map_kp],
    ]
    instances = {}
    for portfolio in portfolios:
        eig = EAGenerator(
            pop_size=10,
            generations=1000,
            domain=kp_domain,
            portfolio=portfolio,
            archive=Archive(threshold=0.5),
            s_set=Archive(threshold=0.05),
            k=3,
            repetitions=1,
            descriptor="features",
            replacement=first_improve_replacement,
            transformer=transformer,
        )
        _, solution_set = eig()
        instances[portfolio[0].__name__] = copy.copy(solution_set)

    return instances


def main(repetition: int = 0):
    nn = KerasNN(
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

    exp_filename = f"instances_best_NN_gecco_24_f_and_e_rep_{repetition}.csv"
    print(f"Running repetition: {repetition}")
    instances = generate_instances(nn)
    save_instances(exp_filename, instances)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            f"Error expected a repetition number.\n\tpython3 gecco_24_gen_best_nn.py <repetition_idx>"
        )
    rep = int(sys.argv[1])
    main(rep)
