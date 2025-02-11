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

from digneapy import Archive, NS
from digneapy.domains import KnapsackDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import default_kp, map_kp, miw_kp


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


def generate_instances(dimension: int):
    kp_domain = KnapsackDomain(dimension, capacity_approach="percentage")
    portfolios = [
        [default_kp, map_kp, miw_kp],
        [map_kp, default_kp, miw_kp],
        [miw_kp, default_kp, map_kp],
    ]
    instances = {}
    for portfolio in portfolios:
        p_names = [s.__name__ for s in portfolio]
        status = f"\rRunning portfolio: {p_names}"
        print(status, end="")

        eig = EAGenerator(
            pop_size=10,
            generations=1000,
            domain=kp_domain,
            portfolio=portfolio,
            novelty_approach=NS(Archive(threshold=3), k=3),
            solution_set=NS(Archive(threshold=3), k=1),
            repetitions=1,
            descriptor_strategy="features",
            replacement=generational_replacement,
        )
        _, solution_set = eig()
        instances[portfolio[0].__name__] = copy.deepcopy(solution_set)

        status = f"\rRunning portfolio: {p_names} completed ✅"
        print(status, end="")
    # When completed clear the terminal
    blank = " " * 80
    print(f"\r{blank}\r", end="")
    return instances


def main(dimension: int, repetition: int = 0):
    exp_filename = f"instances_nsf_N_{dimension}_{repetition}.csv"
    print(
        f"Running experiment for dimension: {dimension} and repetition: {repetition} 🚀"
    )
    instances = generate_instances(dimension)
    save_instances(exp_filename, instances, dimension=dimension)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Error expected a dimension and a repetition number.\n\tpython3 ns_variable_size.py <size> <repetition_idx>"
        )
    dimension = int(sys.argv[1])
    rep = int(sys.argv[2])
    main(dimension, rep)
