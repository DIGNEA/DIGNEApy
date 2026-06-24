#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   yamable.py
@Time    :   2026/06/24 14:01:47
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from pprint import pprint

from digneapy.core import DescriptorPipeline
from digneapy.domains import KnapsackDomain
from digneapy.generators import Dominated
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)


def main():

    population_size = 32
    generations = 100
    k = 3
    number_of_items = 100
    repetitions = 1
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    domain = KnapsackDomain(number_of_items=number_of_items)
    descriptor_pipeline = DescriptorPipeline("features")
    generator = Dominated(
        pop_size=population_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        repetitions=repetitions,
        k=k,
        descriptor_pipe=descriptor_pipeline,
    )
    print(generator.__repr__())
    input()
    pprint(vars(generator))


if __name__ == "__main__":
    main()
