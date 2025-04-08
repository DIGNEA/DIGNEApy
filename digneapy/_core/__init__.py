#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/06/07 14:06:26
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from ._constants import Direction, IndType
from ._domain import Domain
from ._instance import Instance
from ._metrics import Logbook, qd_score, qd_score_auc, Statistics
from ._novelty_search import NS, DominatedNS
from ._problem import P, Problem
from ._solution import Solution
from ._solver import Solver, SupportsSolve
from .types import RNG
from ._knn import sparseness, sparseness_only_values

__all__ = [
    "Domain",
    "Instance",
    "Problem",
    "P",
    "Solution",
    "Solver",
    "SupportsSolve",
    "IndType",
    "Direction",
    "NS",
    "DominatedNS",
    "qd_score",
    "qd_score_auc",
    "Statistics",
    "Logbook",
    "RNG",
]
