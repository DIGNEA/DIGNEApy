#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   tsp.py
@Time    :   2025/02/21 10:47:31
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Mapping, Self, Tuple

import numpy as np

from digneapy.core import Domain, Instance, Problem, Solution


class TSP(Problem):
    def __init__(
        self,
        nodes: int,
        coords: Tuple[Tuple[int, int], ...],
        *args,
        **kwargs,
    ):
        self._nodes = nodes
        self._coords = np.array(coords)

        x_min, y_min = np.min(self._coords, axis=0)
        x_max, y_max = np.max(self._coords, axis=0)
        bounds = list(((x_min, y_min), (x_max, y_max)) for _ in range(self._nodes))
        super().__init__(dimension=nodes, bounds=bounds, name="TSP")

    def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Travelling Salesmas Problem.

        The fitness of the solution is the fraction of the sum of the distances of the tour
        defined as:
            (x) = \\frac{\\sum_{k=1}^{N} \\left(\\frac{fill_k}{C}\\right)^2}{N}

        Args:
            individual (Sequence | Solution): Individual to evaluate

        Returns:
            Tuple[float]: Fitness
        """
        if len(individual) != self._nodes:
            msg = f"Mismatch between individual variables ({len(individual)}) and instance variables ({self._nodes}) in {self.__class__.__name__}"
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
        return f"TSP<n={self._nodes}>"

    def __len__(self):
        return self._nodes

    def create_solution(self) -> Solution:
        items = list(range(self._nodes))
        return Solution(chromosome=items)

    def to_file(self, filename: str = "instance.tsp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\n\n")
            content = "\n".join(f"{x}\t{y}" for (x, y) in self.coords)
            file.write(content)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        nodes = int(lines[0])
        coords = []
        for line in lines[2:]:
            x, y = line.split()
            coords.append((int(x), int(y)))

        return cls(nodes=nodes, coords=tuple(coords))

    def to_instance(self) -> Instance:
        return Instance(variables=self._coords.flatten())


class TSPDomain(Domain):
    def __init__(
        self,
        nodes: int = 100,
        x_range: Tuple[int, int] = (0, 1000),
        y_range: Tuple[int, int] = (0, 10000),
    ):
        if nodes < 0:
            raise ValueError(f"Expected dimension > 0 got {nodes}")
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError(
                f"Expected x_range and y_range to be a tuple with only to integers. Got: x_range = {x_range} and y_range = {y_range}"
            )
        x_min, x_max = x_range
        y_min, y_max = y_range
        if x_min < 0 or x_max <= x_min:
            raise ValueError(
                f"Expected x_range to be (x_min, x_max) where x_min >= 0 and x_max > x_min. Got: x_range {x_range}"
            )
        if y_min < 0 or y_max <= y_min:
            raise ValueError(
                f"Expected y_range to be (y_min, y_max) where y_min >= 0 and y_max > y_min. Got: y_range {y_range}"
            )

        self._nodes = nodes
        self._x_range = x_range
        self._y_range = y_range
        __bounds = [(self._x_range, self._y_range) for _ in range(self._nodes)]

        super().__init__(dimension=self._nodes, bounds=__bounds, name="TSP")

    def generate_instance(self) -> Instance:
        """Generates a new instances for the TSP domain

        Returns:
            Instance: New randomly generated instance
        """
        coords = np.random.randint(
            low=(self._x_range[0], self._y_range[0]),
            high=(self._x_range[1], self._y_range[1]),
            size=(self._nodes, 2),
            dtype=int,
        )
        coords = coords.flatten()
        return Instance(coords)

    def extract_features(self, instance: Instance) -> tuple:
        """Extract the features of the instance based on the TSP domain.
           For the TSP the features are:
           TODO: Update to the TSP case

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
        names = ""  # TODO: Update with the correct names for TSP features
        features = self.extract_features(instance)
        return {k: v for k, v in zip(names.split(","), features)}

    def from_instance(self, instance: Instance) -> TSP:
        n_nodes = len(instance) // 2
        coords = (0,)
        return TSP(nodes=n_nodes, coords=coords)
