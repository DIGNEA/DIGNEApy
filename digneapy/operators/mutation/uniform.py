#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   uniform.py
@Time    :   2026/05/21 14:52:09
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Optional

import numpy as np

from digneapy.typing import IndType

from ._base_mutation import Mutation


class UniformMutation(Mutation):
    def __init__(self, seed: Optional[int | np.random.SeedSequence] = None):
        super().__init__(seed)

    def __call__(self, individual: IndType, lb: np.ndarray, ub: np.ndarray) -> IndType:
        """Performs Uniform One Mutation on Instances and Solution objects.

        Args:
            individual (IndType): Instance or Solution to mutate.
            lb (np.ndarray): Lower bound for each dimension
            ub (np.ndarray): Upper bound for each dimension
            seed (Optional[int  |  np.random.SeedSequence], optional): Seed for the random number generator. Defaults to None.

        Raises:
            ValueError: If bouns != dimension

        Returns:
            IndType: Newly mutated individual
        """
        if len(lb) != len(ub) or len(individual) != len(lb):
            msg = f"The size of individual ({len(individual)}) and bounds {len(lb)} is different in uniform_one_mutation."
            raise ValueError(msg)

        mutation_point = self._rng.integers(low=0, high=len(individual))
        new_value = self._rng.uniform(low=lb[mutation_point], high=ub[mutation_point])
        individual[mutation_point] = new_value
        return individual


class BatchUniformMutation(Mutation):
    def __init__(self, seed: Optional[int | np.random.SeedSequence] = None):
        super().__init__(seed)

    def __call__(
        self, population: np.ndarray, lb: np.ndarray, ub: np.ndarray
    ) -> np.ndarray:
        """Performs uniform one mutation in batches

        Args:
            population (np.ndarray): Batch of individuals to mutate
            lb (np.ndarray): Lower bound for each dimension
            ub (np.ndarray): Upper bound for each dimension
            seed (Optional[int  |  np.random.SeedSequence], optional): Seed for the random number generator. Defaults to None.


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
        mutation_points = self._rng.integers(low=0, high=dimension, size=n_individuals)
        new_values = self._rng.uniform(
            low=lb[mutation_points], high=ub[mutation_points], size=n_individuals
        )

        population[np.arange(n_individuals), mutation_points] = new_values

        return population


class UMut(UniformMutation):
    pass


class BatchUMut(BatchUniformMutation):
    pass
