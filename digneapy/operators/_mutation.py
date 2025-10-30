#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   mutation.py
@Time    :   2023/11/03 10:33:30
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["uniform_one_mutation", "batch_uniform_one_mutation", "Mutation"]

from collections.abc import Callable, Sequence
from typing import Tuple

import numpy as np

from .._core import IndType

Mutation = Callable[[IndType, Sequence[Tuple]], IndType]


def uniform_one_mutation(individual: IndType, bounds: Sequence[Tuple]) -> IndType:
    if len(individual) != len(bounds):
        msg = f"The size of individual ({len(individual)}) and bounds {len(bounds)} is different in uniform_one_mutation"
        raise ValueError(msg)

    rng = np.random.default_rng()
    mutation_point = rng.integers(low=0, high=len(individual))
    new_value = rng.uniform(
        low=bounds[mutation_point][0], high=bounds[mutation_point][1]
    )
    individual[mutation_point] = new_value
    return individual


def batch_uniform_one_mutation(
    population: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> np.ndarray:
    """Performs uniform one mutation in batches

    Args:
        population (np.ndarray): Batch of individuals to mutate
        lb (np.ndarray): Lower bound for each dimension
        up (np.ndarray): Upper bound for each dimension

    Raises:
        ValueError: If the dimension of the individuals do not match the bounds

    Returns:
        np.ndarray: mutated population
    """
    dimension = len(population[0])
    n_individuals = len(population)
    if len(lb) != len(ub) or dimension != len(lb):
        msg = f"The size of individuals ({dimension}) and bounds {len(lb)} is different in uniform_one_mutation"
        raise ValueError(msg)

    rng = np.random.default_rng(84793258734753)
    mutation_points = rng.integers(low=0, high=dimension, size=n_individuals)
    new_values = rng.uniform(
        low=lb[mutation_points], high=ub[mutation_points], size=n_individuals
    )

    population[np.arange(n_individuals), mutation_points] = new_values

    return population
