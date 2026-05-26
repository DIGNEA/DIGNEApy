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

from ._domain import Domain
from ._instance import Instance
from ._metrics import Logbook, Statistics, qd_score, qd_score_auc
from ._problem import Problem
from ._solution import Solution
from ._solver import Solver
from ._types import Direction, IndType
from .descriptors import (
    DescriptorFn,
    DescriptorKey,
    DescriptorPipeline,
    descriptors_registry,
)

__all__ = [
    "Domain",
    "Instance",
    "Problem",
    "Solution",
    "Solver",
    "IndType",
    "Direction",
    "qd_score",
    "qd_score_auc",
    "Statistics",
    "Logbook",
    "DescriptorKey",
    "DescriptorFn",
    "DescriptorPipeline",
    "descriptors_registry",
]
