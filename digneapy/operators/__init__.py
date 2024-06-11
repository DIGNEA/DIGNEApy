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

from digneapy.operators import crossover, mutation, replacement, selection

__all__ = [
    "crossover",
    "mutation",
    "selection",
    "replacement",
]
