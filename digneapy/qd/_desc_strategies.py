#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _descriptor_strategies.py
@Time    :   2024/06/07 14:29:09
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   Descriptors Strategies for Instance Generation
"""

from collections.abc import Callable, Iterable, MutableMapping

import numpy as np

from digneapy.core import Instance

""" DescStrategy defines the type for a Descriptor Strategy.
    A descriptor strategy is any callable able to extract the
    valuable information to describe each instance in a iterable.
    Args:
        iterable of objects of the Instance class
    Returns:
        np.ndarray: Array with the descriptors of each instance in the iterable
"""
DescStrategy = Callable[[Iterable[Instance]], np.ndarray]


def rdstrat(key: str, verbose: bool = False):
    """Decorator to create new descriptor strategies

    Args:
        key (str): Key to refer the descriptor-function
        verbose (bool, optional): Prints a message when the function is registered. Defaults to False.
    """

    def decorate(func: DescStrategy):
        if verbose:
            print(f"Registering descriptor function: {func.__name__} with key: {key}")
        descriptor_strategies[key] = func
        return func

    return decorate


def __property_strategy(attr: str):
    """Returns a np.ndarray with the information required of the instances

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        np.ndarray: Array of the feature descriptors of each instance
    """
    try:
        if attr not in ("features", "transformed"):
            raise AttributeError()
    except AttributeError:
        raise ValueError(
            f"Object of class Instance does not have a property named {attr}"
        )

    def strategy(iterable: Iterable[Instance]) -> np.ndarray:
        return np.asarray([getattr(i, attr) for i in iterable])

    return strategy


def performance_strategy(iterable: Iterable[Instance]) -> np.ndarray:
    """It generates the performance descriptor of an instance
    based on the scores of the solvers in the portfolio over such instance

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        np.ndarray: Array of performance descriptors of each instance
    """
    return np.asarray([np.mean(i.portfolio_scores, axis=1) for i in iterable])


def instance_strategy(iterable: Iterable[Instance]) -> np.ndarray:
    """It returns the instance information as its descriptor

    Args:
        iterable (Iterable[Instance]): Instances to describe

    Returns:
        np.ndarray: Array of descriptor instance (whole instace data)
    """
    return np.asarray([*iterable])


""" Set of pre-defined descriptor strategies available in digneapy.
    - features --> Creates a np.ndarray with all the features of the instances.
    - performance --> Creates a np.ndarray with the mean performance score of each solver over the instances.
    - instance --> Creates a np.ndarray with the whole instance as its self descriptor.
    - transformed --> Creates a np.ndarray with all the transformed descriptors of the instances. Only when using a Transformer.
"""
descriptor_strategies: MutableMapping[str, DescStrategy] = {
    "features": __property_strategy(attr="features"),
    "performance": performance_strategy,
    "instance": instance_strategy,
    "transformed": __property_strategy(attr="transformed"),
}
