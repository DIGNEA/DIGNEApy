#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   replacement.py
@Time    :   2023/11/03 10:33:22
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""
from ..core import Instance, Solution
import numpy as np
import copy
import itertools
from operator import attrgetter
from typing import Callable, Union
from collections.abc import Sequence


Replacement = Callable[
    [Sequence[Union[Instance | Solution]], Sequence[Union[Instance | Solution]]],
    Sequence[Union[Instance | Solution]],
]


def generational(
    current_population: Sequence[Union[Instance | Solution]],
    offspring: Sequence[Union[Instance | Solution]],
) -> Sequence[Union[Instance | Solution]]:
    """Returns the offspring population as the new current population

    Args:
        current_population (Sequence[Union[Instance | Solution]]): Current population in the algorithm
        offspring (Sequence[Union[Instance | Solution]]): Offspring population

    Raises:
        AttributeError: Raises if the sizes of the population are different

    Returns:
        Sequence[Union[Instance | Solution]]: New population
    """
    if len(current_population) != len(offspring):
        msg = f"The size of the current population ({len(current_population)}) != size of the offspring ({len(offspring)}) in generational replacement"
        raise AttributeError(msg)

    return copy.deepcopy(offspring)


def first_improve_replacement(
    current_population: Sequence[Union[Instance | Solution]],
    offspring: Sequence[Union[Instance | Solution]],
) -> Sequence[Union[Instance | Solution]]:
    """Returns a new population produced by a greedy operator.
    Each individual in the current population is compared with its analogous in the offspring population
    and the best survives

    Args:
        current_population (Sequence[Union[Instance | Solution]]): Current population in the algorithm
        offspring (Sequence[Union[Instance | Solution]]): Offspring population

    Raises:
        AttributeError: Raises if the sizes of the population are different

    Returns:
        Sequence[Union[Instance | Solution]]: New population
    """
    if len(current_population) != len(offspring):
        msg = f"The size of the current population ({len(current_population)}) != size of the offspring ({len(offspring)}) in first_improve_replacement"
        raise AttributeError(msg)

    return [a if a > b else b for a, b in zip(current_population, offspring)]


def elitist_replacement(
    current_population: Sequence[Union[Instance | Solution]],
    offspring: Sequence[Union[Instance | Solution]],
    hof: int = 1,
) -> Sequence[Union[Instance | Solution]]:
    """Returns a new population constructed using the Elitist approach.
    HoF number of individuals from the current + offspring populations are
    kept in the new population. The remaining individuals are selected from
    the offspring population.

    Args:
        current_population Sequence[Union[Instance | Solution]]: Current population in the algorithm
        offspring Sequence[Union[Instance | Solution]]: Offspring population
        hof (int, optional): _description_. Defaults to 1.

    Raises:
        AttributeError: Raises if the sizes of the population are different

    Returns:
        Sequence[Union[Instance | Solution]]: New population
    """
    if len(current_population) != len(offspring):
        msg = f"The size of the current population ({len(current_population)}) != size of the offspring ({len(offspring)}) in elitist_replacement"
        raise AttributeError(msg)

    combined_population = sorted(
        itertools.chain(current_population, offspring),
        key=attrgetter("fitness"),
        reverse=True,
    )
    top = combined_population[:hof]
    return list(top + offspring[1:])
