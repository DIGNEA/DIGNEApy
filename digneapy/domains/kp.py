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
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy import typing as npt
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
        if len(profits) != len(weights):
            raise ValueError(
                f"The number of weights and profits is different in Knapsack. Got {len(weights)} weights and {len(profits)} profits"
            )
        if capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer. Got {capacity}")

        super().__init__(dimension=len(profits), bounds=[], name="KP", seed=seed)

        self.weights = weights
        self.profits = profits
        self.capacity = capacity
        self.penalty_factor = 100.0

    def get_bounds_at(self, i: int) -> tuple:
        if i < 0 or i > self._dimension:
            raise ValueError(
                f"Index {i} out-of-range. The bounds are 0-{self._dimension} "
            )
        return (0, 1)

    @property
    def bounds(self):
        return list((0, 1) for _ in range(self._dimension))

    def evaluate(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        """Evaluates the candidate individual with the information of the Knapsack

        Args:
            individual (Sequence | Solution): Individual to evaluate

        Raises:
            ValueError: Raises an error if the len(individual) != len(profits or weights)

        Returns:
            Tuple[float]: Profit
        """

        if len(individual) != self._dimension:
            msg = f"Mismatch between individual variables and instance variables in {self.__class__.__name__}"
            raise ValueError(msg)

        profit = np.dot(individual, self.profits)
        packed = np.dot(individual, self.weights)
        difference = max(0, packed - self.capacity)
        penalty = self.penalty_factor * difference
        profit -= penalty

        return (profit,)

    def __call__(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        return self.evaluate(individual)

    def __array__(self, dtype=np.int32, copy: Optional[bool] = None) -> npt.ArrayLike:
        """Creates a numpy array from the Knapsack instance description.

        Returns:
            npt.ArrayLike: 1d numpy array of size 1 + (2 * dimension)
        """
        return np.asarray(
            [
                self.capacity,
                *list(
                    itertools.chain.from_iterable([*zip(self.weights, self.profits)])
                ),
            ],
            dtype=dtype,
            copy=copy,
        )

    def __repr__(self):
        return f"KP<n={len(self.profits)},C={self.capacity}>"

    def __len__(self):
        return len(self.weights)

    def create_solution(self) -> Solution:
        chromosome = self._rng.integers(low=0, high=1, size=self._dimension)
        return Solution(variables=chromosome)

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
        max_capacity: int = int(1e5),
        capacity_ratio: float = 0.8,
        seed: Optional[int] = None,
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

    def generate_instances(self, n: int = 1) -> List[Instance]:
        """Generates N instances for the domain.

        Args:
            n (int, optional): Number of instances to generate. Defaults to 1.

        Returns:
            List[Instance]: A list of Instance objects created from the raw numpy generation
        """
        weights_and_profits = np.empty(shape=(n, self.dimension * 2), dtype=np.int32)
        weights_and_profits[:, 0::2] = self._rng.integers(
            low=self.min_w, high=self.max_w, size=(n, self.dimension)
        )
        weights_and_profits[:, 1::2] = self._rng.integers(
            low=self.min_p, high=self.max_p, size=(n, self.dimension)
        )
        # Assume fixed
        capacities = np.full(n, fill_value=self.max_capacity, dtype=np.int32)
        match self.capacity_approach:
            case "evolved":
                capacities[:] = self._rng.integers(1, self.max_capacity, size=n)
            case "percentage":
                capacities[:] = (
                    np.sum(weights_and_profits[:, 1::2], axis=1) * self.capacity_ratio
                ).astype(np.int32)
        return list(
            Instance(i) for i in np.column_stack((capacities, weights_and_profits))
        )

    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        """Extract the features of the instance based on the domain

        Args:
            instances (Sequence[Instance]): Instances to extract the features from.

        Returns:
            ArrayLike: 2d array with the features of each instance
        """

        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances, copy=True)

        features = np.empty(shape=(len(instances), 8), dtype=np.float32)
        weights = instances[:, 1::2]
        profits = instances[:, 2::2]
        features[:, 0] = instances[:, 0]  # Qs
        features[:, 1] = np.max(profits, axis=1)
        features[:, 2] = np.max(weights, axis=1)
        features[:, 3] = np.min(profits, axis=1)
        features[:, 4] = np.min(weights, axis=1)
        features[:, 5] = np.mean(profits / weights)
        features[:, 6] = np.mean(instances[:, 1:], axis=1)
        features[:, 7] = np.std(instances[:, 1:], axis=1)

        return features

    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Dict[str, np.float32]]:
        """Creates a dictionary with the features of the instance.
        The key are the names of each feature and the values are
        the values extracted from instance.

        Args:
            instances (Sequence[Instance]): Instances to extract the features from. They should in the an array form.

        Returns:
            Dict[str, float]: Dictionary with the names/values of each feature
        """
        features = self.extract_features(instances)
        named_features: list[dict[str, np.float32]] = [{}] * len(features)
        for i, feats in enumerate(features):
            named_features[i] = {k: v for k, v in zip(self.feat_names, feats)}

        return named_features

    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List:
        """Generates a List of Knapsack objects from the instances

        Args:
            instances (Sequence[Instance]): Instances to create the problems from

        Returns:
            List: List containing len(instances) objects of type Knapsack
        """
        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances)

        capacities = instances[:, 0].astype(int)
        weights = instances[:, 1::2].astype(int)
        profits = instances[:, 2::2].astype(int)
        # Sets the capacity according to the method
        if self.capacity_approach == "percentage":
            capacities[:] = (np.sum(weights, axis=1) * self.capacity_ratio).astype(
                np.int32
            )
        elif self.capacity_approach == "fixed":
            capacities[:] = self.max_capacity
        return list(
            Knapsack(profits=profits[i], weights=weights[i], capacity=capacities[i])
            for i in range(len(instances))
        )
