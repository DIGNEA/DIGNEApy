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

from digneapy.domains import KnapsackDomain
from digneapy.generators import DEAGenerator
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp


def save_instances(filename, generated_instances, dimension: int):
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
    header = ["target", *features] + list(
        itertools.chain.from_iterable([(f"w_{i}", f"p_{i}") for i in range(dimension)])
    )

    with open(filename, "w") as file:
        file.write(",".join(header) + "\n")
        for solver, instances in generated_instances.items():
            for instance in instances:
                vars = ",".join(str(x) for x in instance[1:])
                features = ",".join(str(f) for f in instance.features)
                content = solver + "," + features + "," + vars + "\n"
                file.write(content)


def generate_instances(dimension: int, repetition: int = 0):
    kp_domain = KnapsackDomain(dimension, capacity_approach="percentage")
    portfolios = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]
    instances = {}
    for portfolio in portfolios:
        p_names = [s.__name__ for s in portfolio]

        eig = DEAGenerator(
            pop_size=128,
            offspring_size=128,
            generations=1000,
            domain=kp_domain,
            portfolio=portfolio,
            k=15,
            repetitions=1,
            descriptor_strategy="features",
        )
        population, _ = eig(verbose=False)
        df = eig.log.to_df()
        df.to_csv(
            f"dominated_nsf_generator_{portfolio[0].__name__}_log_run_{repetition}.csv",
            index=False,
        )
        instances[portfolio[0].__name__] = copy.deepcopy(population)

        status = f"\rRunning portfolio: {p_names} completed ✅"
        print(status, end="")
    # When completed clear the terminal
    blank = " " * 80
    print(f"\r{blank}\r", end="")
    return instances


def main(dimension: int, repetition: int = 0):
    exp_filename = f"instances_dominated_nsf_N_{dimension}_{repetition}.csv"
    print(
        f"Running experiment for dimension: {dimension} and repetition: {repetition} 🚀"
    )
    instances = generate_instances(dimension, repetition=repetition)
    save_instances(exp_filename, instances, dimension=dimension)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Error expected a dimension and a repetition number.\n\tpython3 ns_variable_size.py <size> <repetition_idx>"
        )
    dimension = int(sys.argv[1])
    rep = int(sys.argv[2])
    main(dimension, rep)
