#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/06/07 14:28:12
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

# from digneapy.qd._cma_me import CMA_ME
# from digneapy.qd._desc_strategies import (
#     DescStrategy,
#     descriptor_strategies,
#     instance_strategy,
#     performance_strategy,
#     rdstrat,
# )


def __getattr__(name):
    if name == "cma_me":
        import digneapy.qd._cma_me as cmae_me

        return cmae_me
