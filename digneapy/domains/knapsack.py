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


from ..core import OptProblem
from digneapy.core import Instance, Domain
from typing import Iterable, Tuple, List
import numpy as np
import itertools


class Knapsack(OptProblem):
    def __init__(
        self,
        profits: Iterable[int],
        weights: Iterable[int],
        capacity: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__("KP", args, kwargs)
        self.weights = weights
        self.profits = profits
        self.capacity = capacity

    def evaluate(self, individual: Iterable) -> Tuple[float]:
        """Evaluates the candidate individual with the information of the Knapsack"""
        if len(individual) != len(self.instance) // 2:
            msg = f"Mismatch between individual variables and instance variables in {self.__class__.__name__}"
            raise AttributeError(msg)

        profit = 0
        packed = 0

        for i in range(len(individual)):
            profit += individual[i] * self.profits[i]
            packed += individual[i] * self.weights[i]

        difference = packed - self.capacity
        penalty = 100.0 * difference
        profit -= penalty if penalty > 0.0 else 0.0
        return (profit,)

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
        max_capacity: int = 1e4,
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
            self.capacity_approach = "evolved"
        else:
            self.capacity_approach = capacity_approach

        bounds = [(0.0, self.max_capacity)] + [
            (min_w, max_w) if i % 2 == 0 else (min_p, max_p)
            for i in range(2 * dimension)
        ]
        super().__init__(
            "KP",
            dimension=dimension,
            bounds=bounds,
        )

    def generate_instance(self) -> Instance:
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

    def extract_features(self, instance: Instance) -> List[float]:
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

        return [
            int(capacity),
            max(profits),
            max(weights),
            min(profits),
            min(weights),
            avg_eff,
            np.mean(vars),
            np.std(vars),
        ]

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

        return Knapsack(weights, profits, int(capacity))
