#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_heuristics.py
@Time    :   2024/06/19 08:19:47
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import argparse

from digneapy.domains import KnapsackDomain
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp


def generate_sample_problems(n: int, dimension: int = 50):
    domain = KnapsackDomain(dimension=dimension)
    instances = domain.generate_instances(n)
    problems = domain.generate_problems_from_instances(instances)
    for problem in problems:
        yield problem


def main(n: int, dimension: int = 50):
    for problem in generate_sample_problems(n, dimension):
        default_f = default_kp(problem)[0]
        map_f = map_kp(problem)[0]
        miw_f = miw_kp(problem)[0]
        mpw_f = mpw_kp(problem)[0]
        print(
            f"- Def: [{default_f.fitness}], MaP: [{map_f.fitness}], MiW: [{miw_f.fitness}], MPW: [{mpw_f.fitness}]  "
        )
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="knapsack_heuristics",
        description="Python script to exemplify how to solve Knapsack instances using digneapy",
    )
    parser.add_argument("n", help="Number of instances to solve")
    parser.add_argument("dimension", help="Number of items in the instances")
    args = parser.parse_args()
    main(int(args.n), int(args.dimension))
