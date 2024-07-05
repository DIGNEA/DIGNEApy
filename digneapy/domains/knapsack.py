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

import itertools
from collections.abc import Sequence
from typing import Mapping, Self

import numpy as np

from digneapy.core import Domain, Instance, Problem, Solution


class Knapsack(Problem):
    def __init__(
        self,
        profits: Sequence[int],
        weights: Sequence[int],
        capacity: int = 0,
        *args,
        **kwargs,
    ):
        assert len(profits) == len(weights)
        assert capacity > 0

        bounds = list((0, 1) for _ in range(len(profits)))
        super().__init__(dimension=len(profits), bounds=bounds, name="KP")

        self.weights = weights
        self.profits = profits
        self.capacity = capacity

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

        profit = 0.0
        packed = 0

        for i in range(len(individual)):
            profit += individual[i] * self.profits[i]
            packed += individual[i] * self.weights[i]

        difference = packed - self.capacity
        penalty = 100.0 * difference
        profit -= penalty if penalty > 0.0 else 0.0

        if isinstance(individual, Solution):
            individual.fitness = profit
            individual.objectives = (profit,)

        return (profit,)

    def __call__(self, individual: Sequence | Solution) -> tuple[float]:
        return self.evaluate(individual)

    def __repr__(self):
        return f"KP<n={len(self.profits)},C={self.capacity}>"

    def __len__(self):
        return len(self.weights)

    def create_solution(self) -> Solution:
        chromosome = list(np.random.randint(low=0, high=1, size=self._dimension))
        return Solution(chromosome=chromosome)

    def to_file(self, filename: str = "instance.kp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\t{self.capacity}\n\n")
            content = "\n".join(
                f"{w_i}\t{p_i}" for w_i, p_i in zip(self.weights, self.profits)
            )
            file.write(content)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        content = np.loadtxt(filename, dtype=int)
        capacity = content[0][1]
        weights, profits = content[1:, 0], content[1:, 1]
        return cls(profits=profits, weights=weights, capacity=capacity)

    def to_instance(self) -> Instance:
        _vars = [self.capacity] + list(
            itertools.chain.from_iterable([*zip(self.weights, self.profits)])
        )
        return Instance(variables=_vars)


class KPDomain(Domain):
    __capacity_approaches = ("evolved", "percentage", "fixed")

    def __init__(
        self,
        dimension: int = 50,
        min_p: int = 1,
        min_w: int = 1,
        max_p: int = 1000,
        max_w: int = 1000,
        capacity_approach: str = "evolved",
        max_capacity: int = int(1e4),
        capacity_ratio: float = 0.8,
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
        super().__init__(dimension=dimension, bounds=bounds, name="KP")

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
        weights = np.random.randint(
            low=self.min_w, high=self.max_w, size=self.dimension
        )
        profits = np.random.randint(
            low=self.min_p, high=self.max_p, size=self.dimension
        )

        capacity = 0
        # Sets the capacity according to the method
        match self.capacity_approach:
            case "evolved":
                capacity = np.random.randint(1, self.max_capacity)
            case "percentage":
                capacity = np.sum(weights) * self.capacity_ratio
            case "fixed":
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
        vars = instance.variables[1:]
        weights = vars[0::2]
        profits = vars[1::2]
        avg_eff = sum([p / w for p, w in zip(profits, weights)]) / len(vars)
        capacity = int(instance.variables[0])
        # Sets the capacity according to the method
        match self.capacity_approach:
            case "percentage":
                capacity = np.sum(weights) * self.capacity_ratio
            case "fixed":
                capacity = self.max_capacity

        return (
            int(capacity),
            max(profits),
            max(weights),
            min(profits),
            min(weights),
            avg_eff,
            np.mean(vars),
            np.std(vars),
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
        names = "capacity,max_p,max_w,min_p,min_w,avg_eff,mean,std"
        features = self.extract_features(instance)
        return {k: v for k, v in zip(names.split(","), features)}

    def from_instance(self, instance: Instance) -> Knapsack:
        variables = instance.variables
        weights = []
        profits = []
        capacity = int(variables[0])
        for i in range(1, len(variables[1:]), 2):
            weights.append(int(variables[i]))
            profits.append(int(variables[i + 1]))

        # Sets the capacity according to the method
        match self.capacity_approach:
            case "percentage":
                capacity = np.sum(weights) * self.capacity_ratio
            case "fixed":
                capacity = self.max_capacity
        # The KP capacity must be updated JIC
        instance.variables[0] = capacity

        return Knapsack(profits=profits, weights=weights, capacity=int(capacity))
