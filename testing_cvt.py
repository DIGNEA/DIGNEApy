#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   testing_mpw.py
@Time    :   2024/09/12 11:39:13
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from collections import Counter

import digneapy as dpy

if __name__ == "__main__":
    dimension = 50
    cvt = dpy.CVTArchive(
        k=1000,
        ranges=[(1.0, 10_000), *[(1.0, 1_000) for _ in range(100)]],
        n_samples=10_000,
    )
    domain = dpy.domains.KnapsackDomain(dimension=dimension)
    map_elites = dpy.generators.MapElitesGenerator(
        domain,
        portfolio=[dpy.solvers.map_kp, dpy.solvers.default_kp, dpy.solvers.miw_kp],
        archive=cvt,
        initial_pop_size=32,
        mutation=dpy.operators.uniform_one_mutation,
        generations=1000,
        descriptor="instance",
        repetitions=1,
    )
    archive = map_elites()
    positive = list(filter(lambda x: x.p >= 0, archive))
    print(f"CVTArchive has {len(cvt)} where {len(positive)} are positive")
    for instance in filter(lambda x: x.p < 0, archive):
        print(f"Instance {instance} with performance score less than zero")
