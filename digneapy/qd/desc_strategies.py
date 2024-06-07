#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   desc_strategies.py
@Time    :   2024/06/07 14:29:09
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   Descriptors Strategies for Instance Generation
"""


import numpy as np
from digneapy.core import Instance
from collections.abc import Iterable


def features_strategy(iterable: Iterable[Instance]) -> list:
    """It generates the feature descriptor of an instance

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        list: List of the feature descriptors of each instance
    """
    return [i.features for i in iterable]


def performance_strategy(iterable: Iterable[Instance]) -> list[float]:
    """It generates the performance descriptor of an instance
    based on the scores of the solvers in the portfolio over such instance

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        List: List of performance descriptors of each instance
    """
    return [np.mean(i.portfolio_scores, axis=0) for i in iterable]


def instance_strategy(iterable: Iterable[Instance]) -> list:
    """It returns the instance information as its descriptor

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        List: List of descriptor instance (whole instace data)
    """
    return [*iterable]
