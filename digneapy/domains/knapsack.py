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


class KPDomain(Domain):
    def __init__(
        self,
        dimension: int = 50,
        min_p: int = 1,
        min_w: int = 1,
        max_p: int = 1000,
        max_w: int = 1000,
    ):
        super().__init__(
            "KP",
            dimension=dimension,
            bounds=list((0, 1) for _ in range(2 * dimension + 1)),
        )
        self.min_p = min_p
        self.min_w = min_w
        self.max_p = max_p
        self.max_w = max_w

    def generate_instance(self) -> Instance:
        weights = np.random.randint(
            low=self.min_w, high=self.max_w, size=self.dimension
        )
        profits = np.random.randint(
            low=self.min_p, high=self.max_p, size=self.dimension
        )
        capacity = np.random.randint((sum(weights) * 0.85))

        variables = [capacity] + list(itertools.chain(*zip(weights, profits)))

        return Instance(variables)

    def extract_features(self, instance: Instance) -> List[float]:
        vars = instance._variables[1:]
        capacity = instance._variables[0]
        weights = vars[0::2]
        profits = vars[1::2]
        avg_eff = sum([p / w for p, w in zip(profits, weights)]) / len(vars)
        return [
            capacity,
            max(profits),
            max(weights),
            min(profits),
            min(weights),
            avg_eff,
            np.mean(vars),
            np.std(vars),
        ]

    @classmethod
    def from_instance(cls, instance: Instance) -> Knapsack:
        variables = instance._variables
        weights = []
        profits = []
        for i in range(0, len(variables[1:]), 2):
            weights.append(variables[i])
            profits.append(variables[i + 1])

        return Knapsack(weights, profits, variables[0])
