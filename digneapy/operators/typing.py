#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   typing.py
@Time    :   2026/06/12 11:45:57
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from .crossover import Crossover, CrossoverFn
from .mutation import Mutation, MutationFn
from .replacement import Replacement, ReplacementFn
from .selection import Selection, SelectionFn

type CrossoverLike = Crossover | CrossoverFn
type MutationLike = Mutation | MutationFn
type ReplacementLike = Replacement | ReplacementFn
type SelectionLike = Selection | SelectionFn
