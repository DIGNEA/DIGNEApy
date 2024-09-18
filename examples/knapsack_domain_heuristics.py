#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_domain_heuristics.py
@Time    :   2023/11/02 11:18:13
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   Example of how to generated diverse and biased KP instances using DIGNEApy
"""

import argparse
import configparser
from collections import deque

from digneapy import Archive
from digneapy.domains import KnapsackDomain
from digneapy.generators import EAGenerator
from digneapy.operators import first_improve_replacement
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp


def main(default_args):
    if default_args:
        dimension = 50
        capacity_approach = "percentage"
        population_size = 10
        generations = 1000
        t_a, t_ss, k = 3, 3, 3
        descriptor = "features"
    else:
        config = configparser.ConfigParser()
        config.read("knapsack_experiment.ini")
        # Reading the parameters
        dimension = int(config["domain"]["dimension"])
        capacity_approach = config["domain"]["capacity"]

        population_size = int(config["generator"]["population_size"])
        generations = int(config["generator"]["generations"])
        t_a = float(config["generator"]["t_a"])
        t_ss = float(config["generator"]["t_ss"])
        k = int(config["generator"]["k"])
        descriptor = config["generator"]["descriptor"]

    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=dimension, capacity_approach=capacity_approach)

    for i in range(len(portfolio)):
        portfolio.rotate(i)
        eig = EAGenerator(
            pop_size=population_size,
            generations=generations,
            domain=kp_domain,
            portfolio=portfolio,
            archive=Archive(threshold=t_a),
            s_set=Archive(threshold=t_ss),
            k=k,
            repetitions=1,
            descriptor=descriptor,
            replacement=first_improve_replacement,
        )
        print(eig)
        archive, solution_set = eig(verbose=True)

        print(f"The archive contains {len(archive)} instances.")
        print(f"The solution set contains {len(solution_set)} instances.")
        print("=" * 80 + " Solution Set Solutions " + "=" * 80)
        for i, instance in enumerate(eig.solution_set):
            filename = f"kp_instances_for_{portfolio[0].__name__}_{i}.kp"
            kp_domain.from_instance(instance).to_file(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Instance Generation for Knapsack Domain",
    )
    parser.add_argument(
        "-d",
        "--default",
        action="store_true",
        help="Using the default parameters for the experiment. Otherwise it uses the .ini file",
    )
    args = parser.parse_args()
    main(args.default)
