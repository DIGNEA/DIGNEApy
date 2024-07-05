#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bin_packing.py
@Time    :   2024/06/18 09:15:05
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Iterable, Sequence
from typing import Mapping, Self

import numpy as np

from digneapy.core import Domain, Instance, Problem, Solution


class BPP(Problem):
    def __init__(
        self,
        items: Iterable[int],
        capacity: int,
        *args,
        **kwargs,
    ):
        self._items = tuple(items)
        self._capacity = capacity
        dim = len(self._items)
        assert len(self._items) > 0
        assert self._capacity > 0

        bounds = list((0, dim - 1) for _ in range(dim))
        super().__init__(dimension=dim, bounds=bounds, name="BPP")

    def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Bin Packing.
        The fitness of the solution is the amount of unused space, as well as the
        number of bins for a specific solution. Falkenauer (1998) performance metric
        defined as:
            (x) = \\frac{\\sum_{k=1}^{N} \\left(\\frac{fill_k}{C}\\right)^2}{N}

        Args:
            individual (Sequence | Solution): Individual to evaluate

        Returns:
            Tuple[float]: Falkenauer Fitness
        """
        if len(individual) != self._dimension:
            msg = f"Mismatch between individual variables ({len(individual)}) and instance variables ({self._dimension}) in {self.__class__.__name__}"
            raise ValueError(msg)

        used_bins = np.max(individual).astype(int) + 1
        fill_i = np.zeros(used_bins)

        for item_idx, bin in enumerate(individual):
            fill_i[bin] += self._items[item_idx]

        fitness = (
            sum(((f_i / self._capacity) * (f_i / self._capacity)) for f_i in fill_i)
            / used_bins
        )
        if isinstance(individual, Solution):
            individual.fitness = fitness
            individual.objectives = (fitness,)

        return (fitness,)

    def __call__(self, individual: Sequence | Solution) -> tuple[float]:
        return self.evaluate(individual)

    def __repr__(self):
        return f"BPP<n={self._dimension},C={self._capacity},I={self._items}>"

    def __len__(self):
        return self._dimension

    def create_solution(self) -> Solution:
        items = list(range(self._dimension))
        return Solution(chromosome=items)

    def to_file(self, filename: str = "instance.bpp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\t{self._capacity}\n\n")
            content = "\n".join(str(i) for i in self._items)
            file.write(content)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        (_, capacity) = lines[0].split()
        items = list(int(i) for i in lines[2:])

        return cls(items=items, capacity=int(capacity))

    def to_instance(self) -> Instance:
        _vars = [self._capacity, *self._items]
        return Instance(variables=_vars)


class BPPDomain(Domain):
    __capacity_approaches = ("evolved", "percentage", "fixed")

    def __init__(
        self,
        dimension: int = 50,
        min_i: int = 1,
        max_i: int = 1000,
        capacity_approach: str = "fixed",
        max_capacity: int = 100,
        capacity_ratio: float = 0.8,
    ):
        if dimension < 0:
            raise ValueError(f"Expected dimension > 0 got {dimension}")
        if min_i < 0:
            raise ValueError(f"Expected min_i > 0 got {min_i}")
        if max_i < 0:
            raise ValueError(f"Expected max_i > 0 got {max_i}")
        if min_i > max_i:
            raise ValueError(
                f"Expected min_i to be less than max_i got ({min_i}, {max_i})"
            )

        self._dimension = dimension
        self._min_i = min_i
        self._max_i = max_i
        self._max_capacity = max_capacity

        if capacity_ratio < 0.0 or capacity_ratio > 1.0 or not float(capacity_ratio):
            self.capacity_ratio = 0.8  # Default
            msg = "The capacity ratio must be a float number in the range [0.0-1.0]. Set as 0.8 as default."
            print(msg)
        else:
            self.capacity_ratio = capacity_ratio

        if capacity_approach not in self.__capacity_approaches:
            msg = f"The capacity approach {capacity_approach} is not available. Please choose between {self.__capacity_approaches}. Evolved approach set as default."
            print(msg)
            self._capacity_approach = "fixed"
        else:
            self._capacity_approach = capacity_approach

        bounds = [(1.0, self._max_capacity)] + [
            (self._min_i, self._max_i) for _ in range(self._dimension)
        ]
        super().__init__(dimension=dimension, bounds=bounds, name="BPP")

    @property
    def capacity_approach(self):
        return self._capacity_approach

    @capacity_approach.setter
    def capacity_approach(self, app):
        """Setter for the Maximum capacity generator approach.
        It forces to update the variable to one of the specify values

        Args:
            app (str): Approach for setting the capacity. It should be fixed, evolved or percentage.
        """
        if app not in self.__capacity_approaches:
            msg = f"The capacity approach {app} is not available. Please choose between {self.__capacity_approaches}. Evolved approach set as default."
            print(msg)
            self._capacity_approach = "fixed"
        else:
            self._capacity_approach = app

    def generate_instance(self) -> Instance:
        """Generates a new instances for the BPP domain

        Returns:
            Instance: New randomly generated instance
        """
        items = np.random.randint(
            low=self._min_i, high=self._max_i, size=self._dimension, dtype=int
        )
        np.random.shuffle(items)

        capacity = 0
        # Sets the capacity according to the method
        match self.capacity_approach:
            case "evolved":
                capacity = np.random.randint(1, self._max_capacity)
            case "percentage":
                capacity = np.sum(items) * self.capacity_ratio
            case "fixed":
                capacity = self._max_capacity

        variables = [capacity, *items]
        return Instance(variables)

    def extract_features(self, instance: Instance) -> tuple:
        """Extract the features of the instance based on the BPP domain.
           For the BPP the features are:
           N, Capacity, MeanWeights, MedianWeights, VarianceWeights, MaxWeight,
           MinWeight, Huge, Large, Medium, Small, Tiny

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple[float]: Values of each feature
        """
        capacity = instance.variables[0]
        vars = np.asarray(instance.variables[1:])
        vars_norm = vars / capacity
        huge = sum(k > 0.5 for k in vars_norm) / self._dimension
        large = sum(0.5 >= k > 1 / 3 for k in vars_norm) / self._dimension
        medium = sum(1 / 3 >= k > 0.25 for k in vars_norm) / self._dimension
        small = sum(0.25 >= k for k in vars_norm) / self._dimension
        tiny = sum(0.1 >= k for k in vars_norm) / self._dimension
        return (
            np.mean(vars_norm),
            np.std(vars_norm),
            np.median(vars_norm),
            np.max(vars_norm),
            np.min(vars_norm),
            tiny,
            small,
            medium,
            large,
            huge,
        )

    def extract_features_as_dict(self, instance: Instance) -> Mapping[str, float]:
        """Creates a dictionary with the features of the instance.
        The key are the names of each feature and the values are
        the values extracted from instance.

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Mapping[str, float]: Dictionary with the names/values of each feature
        """
        names = "mean,std,median,max,min,tiny,small,medium,large,huge"
        features = self.extract_features(instance)
        return {k: v for k, v in zip(names.split(","), features)}

    def from_instance(self, instance: Instance) -> BPP:
        items = instance.variables[1:]
        capacity = int(instance.variables[0])
        # Sets the capacity according to the method
        match self.capacity_approach:
            case "percentage":
                capacity = np.sum(items) * self.capacity_ratio
            case "fixed":
                capacity = self._max_capacity

        # The BPP capacity must be updated JIC
        instance.variables[0] = capacity
        return BPP(items=items, capacity=int(capacity))
