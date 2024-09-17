#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2023/11/03 10:33:37
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

__all__ = [
    "crossover",
    "mutation",
    "selection",
    "replacement",
]


def __getattr__(attr):
    if attr == "crossover":
        import digneapy.operators.crossover as crossover

        return crossover
    if attr == "mutation":
        import digneapy.operators.mutation as mutation

        return mutation

    if attr == "selection":
        import digneapy.operators.selection as selection

        return selection

    if attr == "replacement":
        import digneapy.operators.replacement as replacement

        return replacement
