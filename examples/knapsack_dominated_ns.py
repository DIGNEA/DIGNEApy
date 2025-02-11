#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_dominated_ns.py
@Time    :   2025/02/05 14:10:56
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from collections import deque
from digneapy.domains import KnapsackDomain
from digneapy.generators import DEAGenerator
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp


def main():
    dimension = 100
    k = 15
    capacity_approach = "percentage"
    population_size = 128
    generations = 1000
    descriptor = "features"

    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    kp_domain = KnapsackDomain(dimension=dimension, capacity_approach=capacity_approach)

    for i in range(len(portfolio)):
        portfolio.rotate(i)
        eig = DEAGenerator(
            pop_size=population_size,
            offspring_size=population_size,
            generations=generations,
            domain=kp_domain,
            portfolio=portfolio,
            k=k,
            repetitions=1,
            descriptor_strategy=descriptor,
        )
        print(eig)
        population = eig(verbose=True)

        print(f"The final population contains {len(population)} instances.")
        print("=" * 80 + " Solutions " + "=" * 80)
        for i, instance in enumerate(population):
            filename = f"kp_instances_for_{portfolio[0].__name__}_{i}.kp"
            kp_domain.from_instance(instance).to_file(filename)


if __name__ == "__main__":
    main()
