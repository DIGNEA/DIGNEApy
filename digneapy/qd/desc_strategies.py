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

from collections.abc import Callable, Iterable

import numpy as np

from digneapy.core import Instance

descriptor_strategies = {}


def rdstrat(key: str):
    def decorate(func: Callable[[Iterable[Instance]], np.ndarray]):
        print(f"Registering function: {func} with key: {key}")
        descriptor_strategies[key] = func
        return func

    return decorate


@rdstrat(key="features")
def features_strategy(iterable: Iterable[Instance]) -> np.ndarray:
    """It generates the feature descriptor of an instance

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        list: List of the feature descriptors of each instance
    """
    return np.asarray([i._descriptor for i in iterable])


@rdstrat(key="performance")
def performance_strategy(iterable: Iterable[Instance]) -> np.ndarray:
    """It generates the performance descriptor of an instance
    based on the scores of the solvers in the portfolio over such instance

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        List: List of performance descriptors of each instance
    """
    return np.asarray([np.mean(i.portfolio_scores, axis=1) for i in iterable])


@rdstrat(key="instance")
def instance_strategy(iterable: Iterable[Instance]) -> np.ndarray:
    """It returns the instance information as its descriptor

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        List: List of descriptor instance (whole instace data)
    """
    return np.asarray([*iterable])
