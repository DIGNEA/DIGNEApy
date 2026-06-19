#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2023/10/30 12:35:54
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

from digneapy.domains.bpp import BPP, BPPDomain
from digneapy.domains.kp import Knapsack, KnapsackDomain
from digneapy.domains.sphere import Sphere, SphereDomain
from digneapy.domains.tsp import TSP, TSPDomain

__all__ = [
    "BPP",
    "BPPDomain",
    "Knapsack",
    "KnapsackDomain",
    "TSP",
    "TSPDomain",
    "Sphere",
    "SphereDomain",
]
