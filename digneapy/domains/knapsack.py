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


from digneapy.core import Instance, Domain, Problem
from typing import Tuple, Mapping
import numpy as np
import itertools
from collections.abc import Sequence


class Knapsack(Problem):
    def __init__(
        self,
        profits: Sequence[int],
        weights: Sequence[int],
        capacity: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__("KP", args, kwargs)
        self.weights = weights
        self.profits = profits
        self.capacity = capacity

    def evaluate(self, individual: Sequence) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Knapsack

        Args:
            individual (Sequence): Individual to evaluate

        Raises:
            AttributeError: Raises an error if the len(individual) != len(profits or weights)

        Returns:
            Tuple[float]: Profit
        """
        if len(individual) != len(self.profits):
            msg = f"Mismatch between individual variables and instance variables in {self.__class__.__name__}"
            raise AttributeError(msg)

        profit = 0.0
        packed = 0

        for i in range(len(individual)):
            profit += individual[i] * self.profits[i]
            packed += individual[i] * self.weights[i]

        difference = packed - self.capacity
        penalty = 100.0 * difference
        profit -= penalty if penalty > 0.0 else 0.0
        return (profit,)

    def __call__(self, individual: Sequence) -> tuple[float]:
        return self.evaluate(individual)

    def __repr__(self):
        return f"KP<n={len(self.profits)},C={self.capacity}>"

    def __len__(self):
        return len(self.weights)

    def to_file(self, filename: str = "instance.kp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\t{self.capacity}\n\n")
            for w, p in zip(self.weights, self.profits):
                file.write(f"{w}\t{p}\n")


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
            msg = f"The capacity ratio must be a float number in the range [0.0-1.0]. Set as 0.8 as default."
            print(msg)
        else:
            self.capacity_ratio = capacity_ratio

        if capacity_approach not in self.__capacity_approaches:
            msg = f"The capacity approach {capacity_approach} is not available. Please choose between {self.__capacity_approaches}. Evolved approach set as default."
            print(msg)
            self._capacity_approach = "evolved"
        else:
            self._capacity_approach = capacity_approach

        bounds = [(0.0, self.max_capacity)] + [
            (min_w, max_w) if i % 2 == 0 else (min_p, max_p)
            for i in range(2 * dimension)
        ]
        super().__init__(
            "KP",
            dimension=dimension,
            bounds=bounds,
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
        vars = instance._variables[1:]
        weights = vars[0::2]
        profits = vars[1::2]
        avg_eff = sum([p / w for p, w in zip(profits, weights)]) / len(vars)
        capacity = 0
        # Sets the capacity according to the method
        match self.capacity_approach:
            case "evolved":
                capacity = int(instance._variables[0])
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
        variables = instance._variables
        weights = []
        profits = []
        for i in range(1, len(variables[1:]), 2):
            weights.append(int(variables[i]))
            profits.append(int(variables[i + 1]))

        capacity = 0
        # Sets the capacity according to the method
        match self.capacity_approach:
            case "evolved":
                capacity = int(instance._variables[0])
            case "percentage":
                capacity = np.sum(weights) * self.capacity_ratio
            case "fixed":
                capacity = self.max_capacity

        return Knapsack(profits=profits, weights=weights, capacity=int(capacity))
