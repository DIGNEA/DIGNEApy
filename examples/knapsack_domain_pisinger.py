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


from digneapy.generator import EIG, pisinger_performance_metric
from digneapy.solvers.pisinger import combo, minknap, expknap
from digneapy.domains.knapsack import KPDomain
from digneapy.operators.replacement import first_improve_replacement
from collections import deque


def main():
    dimension = 1000
    capacity_approach = "percentage"
    population_size = 10
    generations = 1000
    t_a, t_ss, k = 1e-3, 1e-4, 3
    descriptor = "features"

    portfolio = deque([combo, minknap, expknap])
    kp_domain = KPDomain(dimension=dimension, capacity_approach=capacity_approach)

    for i in range(len(portfolio) + 1):
        portfolio.rotate(i)
        eig = EIG(
            pop_size=population_size,
            generations=generations,
            domain=kp_domain,
            portfolio=portfolio,
            t_a=t_a,
            t_ss=t_ss,
            k=k,
            repetitions=1,
            descriptor=descriptor,
            performance_function=pisinger_performance_metric,
            replacement=first_improve_replacement,
        )
        print(eig)
        archive, solution_set = eig()

        print(f"The archive contains {len(archive)} instances.")
        print(f"The solution set contains {len(solution_set)} instances.")
        print("=" * 80 + " Solution Set Solutions " + "=" * 80)
        for i, instance in enumerate(eig.solution_set):
            filename = f"kp_instances_for_{portfolio[0].__name__}_{i}.kp"
            kp_domain.from_instance(instance).to_file(filename)


if __name__ == "__main__":
    main()