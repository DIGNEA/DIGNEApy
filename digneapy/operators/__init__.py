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

from .crossover import OPCX, UCX, Crossover, OnePointCrossover, UniformCrossover
from .mutation import (
    BatchUMut,
    BatchUniformMutation,
    ILMut,
    ISOLineMutation,
    Mutation,
    UMut,
    UniformMutation,
)
from .replacement import (
    Elitist,
    Generational,
    GreedyReplacement,
    Replacement,
)
from .selection import BinarySelection, Selection

__all__ = [
    "Mutation",
    "UniformMutation",
    "UMut",
    "BatchUniformMutation",
    "BatchUMut",
    "ISOLineMutation",
    "ILMut",
    "Selection",
    "BinarySelection",
    "Crossover",
    "OnePointCrossover",
    "OPCX",
    "UniformCrossover",
    "UCX",
    "Replacement",
    "Generational",
    "GreedyReplacement",
    "Elitist",
]
