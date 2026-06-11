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

import warnings
from pathlib import Path

__all__ = ["Knapsack", "KnapsackDomain"]

import itertools
from collections.abc import Sequence
from typing import Dict, List, Literal, Optional, Self, Tuple

import numpy as np

from digneapy._core import Domain, Instance, Problem, Solution


class Knapsack(Problem):
    """Representation of a 0/1 Knapsack Problem.

    Each item contributes a profit and consumes a weight.
    A solution is encoded as a binary vector where each entry indicates
    whether the corresponding item is selected.

    The objective rewards profit while penalizing solutions that exceed the assigned capacity.
    """

    def __init__(
        self,
        capacity: int,
        profits: Sequence[np.uint32] | np.ndarray,
        weights: Sequence[np.uint32] | np.ndarray,
        seed: Optional[int | np.random.SeedSequence] = None,
        penalty_factor: float = 100.0 * args,
        **kwargs,
    ):
        """Create a new knapsack problem from the given profit/weight data.

        Args:
            profits (Sequence[np.uint32] | np.ndarray): Profit associated with each item.
            weights (Sequence[np.uint32] | np.ndarray): Weight associated with each item.
            capacity (np.uint64, optional): Maximum total weight allowed in the knapsack.
            seed (Optional[int | np.random.SeedSequence], optional): Seed used to initialize the random generator.

        Raises:
            ValueError: If the profit and weight sequences do not have the same length
                        or if the capacity is not positive.
        """
        try:
            self.capacity = int(capacity)
            if self.capacity <= 0:
                raise ValueError("capacity cannot be negative")

        except (TypeError, ValueError) as exc:
            raise ValueError("invalid capacity for Knapsack Problem") from exc

        if len(profits) != len(weights):
            raise ValueError(
                f"Mismatch of weights and profits in Knapsack. "
                f" Got {len(weights)} weights and {len(profits)} profits."
            )

        self.weights = np.asarray(weights, dtype=np.int32)
        self.profits = np.asarray(profits, dtype=np.int32)
        self._penalization_factor = float(penalty_factor)

        super().__init__(dimension=len(profits), bounds=[], name="KP", seed=seed)

    def get_bounds_at(self, i: int) -> tuple:
        """Return the valid bounds for one decision variable.

        Each item is handled as a binary decision: selecting it corresponds to value 1,
        while leaving it out corresponds to value 0.

        Args:
            i (int): Index of the variable to inspect.

        Raises:
            IndexError: If the index is outside the valid range of the problem.

        Returns:
            tuple: A tuple containing the lower and upper bounds for the variable.
        """
        if i < 0 or i > self._dimension:
            raise IndexError(
                f"Index {i} out-of-range. The bounds are 0-{self._dimension} "
            )
        return (0, 1)

    @property
    def bounds(self):
        """Return the binary bounds for every decision variable in the problem."""

        return list((0, 1) for _ in range(self._dimension))

    def evaluate(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        """Evaluate a candidate solution and compute its objective value.

        The score is the total profit of the selected items minus a penalty for any
        excess weight beyond the knapsack capacity. This makes infeasible solutions
        receive a lower fitness than feasible ones.

        Args:
            individual (Sequence | Solution): Candidate solution to evaluate.

        Raises:
            ValueError: If the individual length does not match the number of items.

        Returns:
            Tuple[float]: A one-element tuple containing the objective value.
        """

        if len(individual) != self._dimension:
            raise ValueError(
                f"Mismatch between individual dimension ({len(individual)}) "
                f"and Knapsack problem ({self._dimension})"
            )

        profit = np.dot(individual, self.profits)
        packed = np.dot(individual, self.weights)
        difference = max(0, packed - self.capacity)
        penalty = self._penalization_factor * difference
        profit -= penalty

        return (profit,)

    def __call__(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        """Evaluate a candidate solution and compute its objective value.

        The score is the total profit of the selected items minus a penalty for any
        excess weight beyond the knapsack capacity. This makes infeasible solutions
        receive a lower fitness than feasible ones.

        Args:
            individual (Sequence | Solution): Candidate solution to evaluate.

        Raises:
            ValueError: If the individual length does not match the number of items.

        Returns:
            Tuple[float]: A one-element tuple containing the objective value.
        """

        return self.evaluate(individual)

    def __array__(self, dtype=np.uint64, copy: Optional[bool] = None) -> np.ndarray:
        """Return a NumPy array representation of the Knapsack Problme.

        The representation stores the capacity first and then alternates weight/profit
        pairs for each item, which is convenient for serialization and downstream processing.

        Returns:
            np.ndarray: A one-dimensional array describing the instance.
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

    def __str__(self):
        """Return a compact string representation of the problem instance."""

        return f"KP(n={self._dimension},C={self.capacity})"

    def __len__(self):
        """Return the number of items defined by the problem."""

        return len(self.weights)

    def create_solution(self) -> Solution:
        """Create a random initial solution for the Knapsack Problem.

        The returned solution is a binary vector that can be used as a starting point
        for an optimizer, although it may be infeasible if the selected items exceed the capacity.

        Returns:
            Solution: A solution object with binary decision variables and empty objectives (1d).
        """
        chromosome = self._rng.integers(low=0, high=1, size=self._dimension)
        return Solution(
            variables=chromosome, objectives=np.zeros(1), constraints=np.zeros(1)
        )

    def to_file(self, filename: str | Path = "instance.kp"):
        """Stores the Knapsack Problem in a plain text file.

        The file format contains the number of items and the capacity on the first line,
        followed by one row per item containing its weight and profit.

        Args:
            filename (str | Path, optional): Destination file for the serialized instance.
        """
        try:
            with open(filename, "w") as file:
                file.write(f"{len(self)}\t{self.capacity}\n\n")
                content = "\n".join(
                    f"{w_i}\t{p_i}" for w_i, p_i in zip(self.weights, self.profits)
                )
                file.write(content)
        except Exception as exc:
            raise RuntimeError(
                "Something went wrong when saving the Knapsack problem."
            ) from exc

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """Load a Knapsack Problem from a text file.

        Args:
            filename (str | Path): Path to the file containing the instance definition.

        Returns:
            Knapsack: A knapsack problem rebuilt from the stored contents.
        """
        try:
            content = np.loadtxt(filename, dtype=np.uint64)
            capacity = content[0][1]
            weights, profits = content[1:, 0], content[1:, 1]
            return cls(profits=profits, weights=weights, capacity=capacity)

        except Exception as exc:
            raise RuntimeError(
                f"Something went wrong when loading the Knapsack problem from {filename}."
            ) from exc

    def to_instance(self) -> Instance:
        """Convert the Knapsack Problem into an Instance object used by Digneapy.

        Returns:
            Instance: An instance object containing the capacity and all item weights/profits.
                The capacity is the first item in the instance variables,
                followed by interleaved weights and profits w_0, p_0, w_1, p_1, etc.
        """
        _variables = [self.capacity] + list(
            itertools.chain.from_iterable([*zip(self.weights, self.profits)])
        )
        return Instance(variables=_variables)


class KnapsackDomain(Domain):
    """Knapsack Domain for synthesizing Knapsack Problem instances.

    This class allows to create benchmark instances by sampling
    item weights and profits and then assigning a capacity using one of several strategies.
    Note that the number of dimensions defined produces instances of N = dimension items.
    Which means that the results Instance objects will have 2 * dimension + 1 variables:
        - Q, w_0, p_0, w_1, p_1, ..., w_N-1, p_N-1

    It also provides utilities to extract descriptive features
    and build concrete Knapsack problems from the generated data.
    """

    capacity_approaches = Literal["evolved", "percentage", "fixed"]

    def __init__(
        self,
        dimension: np.uint32 = np.uint32(50),
        minimum_weight: np.uint32 = np.uint32(1),
        maximum_weight: np.uint32 = np.uint32(1_000),
        minimum_profit: np.uint32 = np.uint32(1),
        maximum_profit: np.uint32 = np.uint32(1_000),
        maximum_capacity: np.uint = np.uint(1e5),
        capacity_approach: str = "evolved",
        capacity_ratio: float = 0.8,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Create a domain that can generate knapsack instances with configurable difficulty.

        Args:
            dimension (np.uint32, optional): Number of items in each generated instance.
            minimum_weight (np.uint32, optional): Lower bound for the weight of each item. Defaults to 1.
            maximum_weight (np.uint32, optional): Upper bound for the weight of each item. Defaults to 1,000.
            minimum_profit (np.uint32, optional): Lower bound for the profit of each item. Defaults to 1.
            maximum_profit (np.uint32, optional): Upper bound for the profit of each item. Defaults to 1,000.
            maximum_capacity (np.uint, optional): Maximum capacity that can be assigned to a
                Knapsack instance when using the evolved or fixed strategy. Defaults to 100,000.
            capacity_approach (str, optional): Strategy used to assign capacities to generated instances. Defaults to evolved.
            capacity_ratio (float, optional): Ratio used to derive the capacity when the percentage strategy is selected. Defaults to 0.8.
            seed (Optional[int | np.random.SeedSequence], optional): Seed used to initialize the random generator. Default to None.
        """
        self._minimum_profit = minimum_profit
        self._minimum_weight = minimum_weight
        self._maximum_profit = maximum_profit
        self._maximum_weight = maximum_weight
        self._maximum_capacity = maximum_capacity

        if (
            type(capacity_ratio) not in (int, float)
            or capacity_ratio < 0.0
            or capacity_ratio > 1.0
        ):
            self.capacity_ratio = 0.8  # Default
            msg = "The capacity ratio must be a float number in the range [0.0-1.0]. Set as 0.8 as default."
            print(msg)
        else:
            self.capacity_ratio = capacity_ratio

        if capacity_approach not in self.capacity_approaches:
            warnings.warn(
                f"The capacity approach {capacity_approach} is not available. "
                f"Please, consider choosing from {self.capacity_approaches}."
                "Evolved approach set as fallback.",
                RuntimeWarning,
            )
            self._capacity_approach = "evolved"
        else:
            self._capacity_approach = capacity_approach

        bounds = [(1.0, self._maximum_capacity)] + [
            (minimum_weight, maximum_weight)
            if i % 2 == 0
            else (minimum_profit, maximum_profit)
            for i in range(2 * dimension)
        ]
        features_names = "capacity,max_p,max_w,min_p,min_w,avg_eff,mean,std".split(",")
        super().__init__(
            dimension=dimension,
            bounds=bounds,
            domain_name="KP",
            features_names=features_names,
            seed=seed,
        )

    @property
    def capacity_approach(self):
        """Return the strategy currently used to assign capacities to generated instances."""
        return self._capacity_approach

    def generate_instances(self, n: np.uint32 = np.uint32(1)) -> List[Instance]:
        """Generate a batch of knapsack instances.

        The method samples item weights and profits for each instance and then assigns a
        capacity according to the selected strategy. This creates instances with varying
        levels of difficulty and tightness.

        Args:
            n (int, optional): Number of instances to generate. Defaults to 1.

        Returns:
            List[Instance]: A list of generated instance objects.
        """
        weights_and_profits = np.empty(shape=(n, self.dimension * 2), dtype=np.uint32)
        weights_and_profits[:, 0::2] = self._rng.integers(
            low=self._minimum_weight,
            high=self._maximum_weight,
            size=(n, self.dimension),
        )
        weights_and_profits[:, 1::2] = self._rng.integers(
            low=self._minimum_profit,
            high=self._maximum_profit,
            size=(n, self.dimension),
        )
        # Assume fixed
        capacities = np.full(n, fill_value=self._maximum_capacity, dtype=np.int32)
        match self.capacity_approach:
            case "evolved":
                capacities[:] = self._rng.integers(1, self._maximum_capacity, size=n)
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
        """Compute a compact set of numerical features for the supplied instances.

        These features summarize the Knapsack instance structure, they include:
            - Capacity
            - Maximum profit
            - Maximum weight
            - Minimum profit
            - Minimum weight
            - Average efficiency as the average ratio of profits / weights
            - Mean of the values (both profits and weights)
            - Standard deviation of the values (both profits and weights)

        Args:
            instances (Sequence[Instance]): Instances to characterize.

        Returns:
            np.ndarray: A two-dimensional array where each row contains the features of one instance.
        """

        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances, copy=True)

        features = np.empty(shape=(len(instances), 8), dtype=np.float64)
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
    ) -> List[Dict[str, np.float64]]:
        """Return the extracted features as dictionaries.

        These features summarize the Knapsack instance structure, they include:
            - Capacity
            - Maximum profit
            - Maximum weight
            - Minimum profit
            - Minimum weight
            - Average efficiency as the average ratio of profits / weights
            - Mean of the values (both profits and weights)
            - Standard deviation of the values (both profits and weights)

        Args:
            instances (Sequence[Instance]): Instances whose features should be extracted.

        Returns:
            List[Dict[str, np.float64]]: One dictionary per instance containing the named features.
        """
        features = self.extract_features(instances)
        named_features = []
        for instance_features in features:
            named_features.append({
                k: v for k, v in zip(self.features_names, instance_features)
            })

        return named_features

    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List:
        """Create Knapsack Problem objects from the given instances.

        This method converts the numerical representation of each instance into a fully
        functional Knapsack Problem that can be passed directly to a solver.

        Args:
            instances (Sequence[Instance]): Instances to transform into problems.

        Returns:
            List: A list containing one Knapsack problem per instance.
        """
        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances)

        capacities = instances[:, 0].astype(np.int32)
        weights = instances[:, 1::2].astype(np.uint32)
        profits = instances[:, 2::2].astype(np.uint32)
        # Sets the capacity according to the method
        if self.capacity_approach == "percentage":
            capacities[:] = (np.sum(weights, axis=1) * self.capacity_ratio).astype(
                np.int32
            )
            instances[:, 0] = capacities[:]
        elif self.capacity_approach == "fixed":
            capacities[:] = self._maximum_capacity
            instances[:, 0] = capacities[:]

        return list(
            Knapsack(profits=profits[i], weights=weights[i], capacity=capacities[i])
            for i in range(len(instances))
        )
