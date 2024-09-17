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

# from digneapy.domains.bin_packing import BPP, BPPDomain
# from digneapy.domains.knapsack import Knapsack, KPDomain


def __getattr__(attr_name):
    if attr_name == "kp":
        import digneapy.domains.kp as kp

        return kp
    elif attr_name == "bpp":
        import digneapy.domains.bpp as bpp

        return bpp
    else:
        raise ImportError(f"module 'digneapy.domains' has no attribute {attr_name}")
