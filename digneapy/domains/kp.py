#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack.py
@Time    :   2023/10/30 12:18:44
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["Knapsack", "KnapsackDomain"]

import itertools
from collections.abc import Sequence
from typing import Mapping

import numpy as np

from digneapy._core import Domain, Instance, Problem, Solution


class Knapsack(Problem):
    def __init__(
        self,
        profits: Sequence[int],
        weights: Sequence[int],
        capacity: int = 0,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        assert len(profits) == len(weights)
        assert capacity > 0

        bounds = list((0, 1) for _ in range(len(profits)))
        super().__init__(dimension=len(profits), bounds=bounds, name="KP", seed=seed)

        self.weights = weights
        self.profits = profits
        self.capacity = capacity
        self.penalty_factor = 100.0

    def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Knapsack

        Args:
            individual (Sequence | Solution): Individual to evaluate

        Raises:
            ValueError: Raises an error if the len(individual) != len(profits or weights)

        Returns:
            Tuple[float]: Profit
        """

        if len(individual) != len(self.profits):
            msg = f"Mismatch between individual variables and instance variables in {self.__class__.__name__}"
            raise ValueError(msg)

        profit = np.dot(individual, self.profits)
        packed = np.dot(individual, self.weights)
        difference = max(0, packed - self.capacity)
        penalty = self.penalty_factor * difference
        profit -= penalty

        return (profit,)

    def __call__(self, individual: Sequence | Solution) -> tuple[float]:
        return self.evaluate(individual)

    def __repr__(self):
        return f"KP<n={len(self.profits)},C={self.capacity}>"

    def __len__(self):
        return len(self.weights)

    def create_solution(self) -> Solution:
        chromosome = self._rng.integers(low=0, high=1, size=self._dimension)
        return Solution(chromosome=chromosome)

    def to_file(self, filename: str = "instance.kp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\t{self.capacity}\n\n")
            content = "\n".join(
                f"{w_i}\t{p_i}" for w_i, p_i in zip(self.weights, self.profits)
            )
            file.write(content)

    @classmethod
    def from_file(cls, filename: str):
        content = np.loadtxt(filename, dtype=int)
        capacity = content[0][1]
        weights, profits = content[1:, 0], content[1:, 1]
        return cls(profits=profits, weights=weights, capacity=capacity)

    def to_instance(self) -> Instance:
        _vars = [self.capacity] + list(
            itertools.chain.from_iterable([*zip(self.weights, self.profits)])
        )
        return Instance(variables=_vars)


class KnapsackDomain(Domain):
    __capacity_approaches = ("evolved", "percentage", "fixed")

    def __init__(
        self,
        dimension: int = 50,
        min_p: int = 1,
        min_w: int = 1,
        max_p: int = 1_000,
        max_w: int = 1_000,
        capacity_approach: str = "evolved",
        max_capacity: int = int(1e7),
        capacity_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.min_p = min_p
        self.min_w = min_w
        self.max_p = max_p
        self.max_w = max_w
        self.max_capacity = max_capacity

        if capacity_ratio < 0.0 or capacity_ratio > 1.0 or not float(capacity_ratio):
            self.capacity_ratio = 0.8  # Default
            msg = "The capacity ratio must be a float number in the range [0.0-1.0]. Set as 0.8 as default."
            print(msg)
        else:
            self.capacity_ratio = capacity_ratio

        if capacity_approach not in self.__capacity_approaches:
            msg = f"The capacity approach {capacity_approach} is not available. Please choose between {self.__capacity_approaches}. Evolved approach set as default."
            print(msg)
            self._capacity_approach = "evolved"
        else:
            self._capacity_approach = capacity_approach

        bounds = [(1.0, self.max_capacity)] + [
            (min_w, max_w) if i % 2 == 0 else (min_p, max_p)
            for i in range(2 * dimension)
        ]
        super().__init__(
            dimension=dimension,
            bounds=bounds,
            name="KP",
            feat_names="capacity,max_p,max_w,min_p,min_w,avg_eff,mean,std".split(","),
            seed=seed,
        )

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
            self._capacity_approach = "evolved"
        else:
            self._capacity_approach = app

    def generate_instance(self) -> Instance:
        """Generates a new instances for the domain

        Returns:
            Instance: New randomly generated instance
        """
        weights = self._rng.integers(
            low=self.min_w, high=self.max_w, size=self.dimension
        )
        profits = self._rng.integers(
            low=self.min_p, high=self.max_p, size=self.dimension
        )

        capacity = 0
        # Sets the capacity according to the method
        if self.capacity_approach == "evolved":
            capacity = self._rng.integers(1, self.max_capacity)
        elif self.capacity_approach == "percentage":
            capacity = np.sum(weights, dtype=int) * self.capacity_ratio
        elif self.capacity_approach == "fixed":
            capacity = self.max_capacity

        variables = [int(capacity)] + list(itertools.chain(*zip(weights, profits)))

        return Instance(variables)

    def extract_features(self, instance: Instance) -> tuple:
        """Extract the features of the instance based on the domain

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple[float]: Values of each feature
        """
        _vars = np.asarray(instance.variables[1:])
        weights = _vars[0::2]
        profits = _vars[1::2]
        avg_eff = np.sum([p / w for p, w in zip(profits, weights)]) / len(_vars)
        capacity = instance.variables[0]
        # Sets the capacity according to the method
        if self.capacity_approach == "percentage":
            capacity = np.sum(weights) * self.capacity_ratio
        elif self._capacity_approach == "fixed":
            capacity = self.max_capacity

        return (
            int(capacity),
            np.max(profits),
            np.max(weights),
            np.min(profits),
            np.min(weights),
            avg_eff,
            np.mean(_vars),
            np.std(_vars),
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
        if len(instance.features) == len(self.feat_names):
            return {k: v for k, v in zip(self.feat_names, instance.features)}
        else:
            features = self.extract_features(instance)
            return {k: v for k, v in zip(self.feat_names, features)}

    def from_instance(self, instance: Instance) -> Knapsack:
        variables = instance.variables
        capacity = variables[0]
        weights = variables[1::2]
        profits = variables[2::2]

        # Sets the capacity according to the method
        if self.capacity_approach == "percentage":
            capacity = np.sum(weights) * self.capacity_ratio
            instance.variables[0] = capacity
        elif self.capacity_approach == "fixed":
            capacity = self.max_capacity
            instance.variables[0] = capacity

        return Knapsack(profits=profits, weights=weights, capacity=int(capacity))
