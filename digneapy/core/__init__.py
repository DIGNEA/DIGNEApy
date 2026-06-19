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

from digneapy.core._descriptors import DescriptorFn, DescriptorKey, DescriptorPipeline
from digneapy.core._domain import Domain
from digneapy.core._instance import Instance
from digneapy.core._metrics import Logbook, Statistics, qd_score, qd_score_auc
from digneapy.core._problem import Problem
from digneapy.core._scores import (
    maximise_perf_gap_easy,
    maximise_perf_gap_hard,
    maximise_runtime_gap,
)
from digneapy.core._solution import Solution
from digneapy.core._solver import Solver

__all__ = [
    "Domain",
    "Instance",
    "Problem",
    "Solution",
    "Solver",
    "Logbook",
    "Statistics",
    "DescriptorKey",
    "DescriptorFn",
    "DescriptorPipeline",
]
