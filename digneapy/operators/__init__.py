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

from ._crossover import Crossover, one_point_crossover, uniform_crossover
from ._mutation import Mutation, uniform_one_mutation
from ._replacement import (
    Replacement,
    elitist_replacement,
    first_improve_replacement,
    generational_replacement,
)
from ._selection import Selection, binary_tournament_selection

__all__ = [
    "Crossover",
    "Mutation",
    "Selection",
    "Replacement",
    "one_point_crossover",
    "uniform_crossover",
    "uniform_one_mutation",
    "elitist_replacement",
    "first_improve_replacement",
    "generational_replacement",
    "binary_tournament_selection",
]
