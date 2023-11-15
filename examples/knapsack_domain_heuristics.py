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


from digneapy.generator import EIG
from digneapy.solvers.heuristics import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.domains.knapsack import KPDomain
from digneapy.operators.replacement import first_improve_replacement


def main():
    kp_domain = KPDomain(dimension=100, capacity_approach="percentage")
    eig = EIG(
        10,
        10000,
        domain=kp_domain,
        portfolio=[map_kp, default_kp, miw_kp, mpw_kp],
        t_a=3,
        t_ss=3,
        k=3,
        repetitions=1,
        descriptor="features",
        replacement=first_improve_replacement,
    )
    print(eig)
    archive, solution_set = eig()
    print(eig)

    print(f"The archive contains {len(archive)} instances.")
    print(f"The solution set contains {len(solution_set)} instances.")
    print("=" * 80 + " Solution Set Solutions " + "=" * 80)
    for i, instance in enumerate(eig.solution_set):
        filename = f"instance_{i}.kp"
        print(instance)
        kp_domain.from_instance(instance).to_file(filename)


if __name__ == "__main__":
    main()
