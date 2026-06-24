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

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Dict, List, Literal, Optional, Self, Tuple

import numpy as np

from digneapy.core import Domain, Instance, Problem, Solution


class BPP(Problem):
    """Bin Packing Problem"""

    def __init__(
        self,
        items: Iterable[int],
        maximum_capacity: np.uint32 | int,
        seed: Optional[int | np.random.SeedSequence] = None,
        *args,
        **kwargs,
    ):
        """Creates a new Bin Packing Problem (BPP) object

        Args:
            items (Iterable[int]): Items to store. It must be any iterable with
                integer values where each value is the weight of an item.
            capacity (np.uint32 | int): Maximum capacity of each bin in the problem.
            seed (Optional[int  |  np.random.SeedSequence], optional): Seed for random number engine. Defaults to None.

        Raises:
            ValueError: If the capacity is not an integer or it's negative.
            ValueError: If any item has a zero or negative weight.
        """

        try:
            self._maximum_capacity = int(maximum_capacity)
            if self._maximum_capacity <= 0:
                raise ValueError("maximum_capacity must be a positive integer in BPP.")

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid maximum_capacity value for a BPP object."
            ) from exc

        try:
            self._items = tuple(map(int, items))
            if len(self._items) == 0:
                raise ValueError(
                    "Invalid items for a BPP object. "
                    f"Expected an iterable with a least one item. Got: {items}"
                )
            if any(item < 0 for item in self._items):
                raise ValueError(
                    "Invalid items for a BPP object. "
                    "Expected all items to be positive integers. "
                    f"Got: {items}"
                )
        except Exception:
            raise

        dimension = len(self._items)
        bounds = [(0, dimension - 1)] * dimension
        super().__init__(dimension=dimension, bounds=bounds, name="BPP", seed=seed)

    @property
    def maximum_capacity(self) -> int:
        return self._maximum_capacity

    @property
    def items(self) -> Tuple[int, ...]:
        return self._items

    def evaluate(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
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
            raise ValueError(
                f"Mismatch between individual ({len(individual)}) "
                f"and problem dimension ({self._dimension}) in BPP."
            )

        used_bins = np.max(individual).astype(np.int32) + 1
        filled_bins = np.zeros(used_bins)

        # For each bin in the solution
        # we set is weight as the sum of the items they store
        # The individual is encoded as follows
        # Each index, refers to the ith item in the instance
        # The value of individual[i] refers to the bin where
        # such item is store
        for item_index, bin in enumerate(individual):
            filled_bins[bin] += self._items[item_index]

        ratio = filled_bins / self._maximum_capacity
        fitness = np.sum(ratio * ratio) / used_bins

        try:
            # We asume that individual is a Solution object
            # Therefore, it must have a fitness and objective attributes
            # Otherwise, we got sequence/ndarray and we just return the fitness
            individual.fitness = fitness
            individual.objectives = (fitness,)
        except Exception:
            return (fitness,)

        return (fitness,)

    def __call__(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
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
        return self.evaluate(individual)

    def __str__(self):
        return f"BPP(n={self._dimension},C={self._maximum_capacity},I={self._items})"

    def __len__(self):
        return self._dimension

    def __array__(self, dtype=np.int32, copy: Optional[bool] = None) -> np.ndarray:
        """Return a NumPy array representation of the Bin Packing Problem.

        The representation stores the capacity first and then all the items.

        Returns:
            np.ndarray: A one-dimensional array describing the instance.
        """
        return np.asarray(
            [self._maximum_capacity, *self._items], dtype=dtype, copy=copy
        )

    def create_solution(self) -> Solution:
        """Creates a random BPP solution

        The solution is created with variables equal to [0, dimension].
        Which means that each item is stored in a independent bin.
        Also, the number of objectives is set to 1.

        Returns:
            Solution: Initial valid solution.
        """
        items = np.arange(self._dimension)
        return Solution(
            variables=items,
            objectives=np.zeros(1),
        )

    def to_file(self, filename: str | Path = "instance.bpp"):
        """Saves the BPP problem to a file

        Args:
            filename (str | Path, optional): Name of the filename to store the problem.
                It can be either a string or a Path. Defaults to "instance.bpp".

        Raises:
            RuntimeError: If something goes wrong when saving the problem.
        """
        try:
            with open(filename, "w") as file:
                file.write(f"{len(self)}\t{self._maximum_capacity}\n\n")
                content = "\n".join(str(i) for i in self._items)
                file.write(content)

        except Exception as exc:
            raise RuntimeError("Failed to save BPP problem to file.") from exc

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """Loads a BPP problem from a file

        Args:
            filename (str | Path):  Name of the filename to load the problem from.
                It can be either a string or a Path.

        Raises:
            RuntimeError: If something goes wrong when loading the problem.

        Returns:
            BPP: Returns a new BPP object with the content of the file
        """
        try:
            with open(filename) as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]

            (_, capacity) = lines[0].split()
            items = list(int(i) for i in lines[2:])

            return cls(items=items, maximum_capacity=int(capacity))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BPP problem from file {filename}"
            ) from exc

    def to_instance(self) -> Instance:
        """Generates an Instance with the information of the Problem

        Returns:
            Instance: New Instance object that defines this BPP
        """
        _variables = [self._maximum_capacity, *self._items]
        return Instance(variables=_variables)


class BPPDomain(Domain):
    capacity_approaches = Literal["evolved", "percentage", "fixed"]

    def __init__(
        self,
        number_of_items: np.uint32 | int = 50,
        minimum_weight: np.uint32 = np.uint32(1),
        maximum_weight: np.uint32 = np.uint32(1_000),
        maximum_capacity: np.uint32 = np.uint32(100),
        capacity_approach: capacity_approaches = "fixed",
        capacity_ratio: float = 0.8,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Bin Packing Problem Domain

        Creates a new domain to generate instances for the Bin Packing Problem (BPP).

        Args:
            number_of_items (np.uint32 | int, optional): Number of items that the instance must contain. Defaults to 50.
            minimum_weight (np.uint32 | int, optional): Minimum value of each item. ç
                This is the lowest weight that an item can have. Defaults to np.uint32(1).
            maximum_weight (np.uint32, optional): Maximum value of each item. This is the highest
                weight that an item can have. Defaults to np.uint32(1_000).
            maximum_capacity (np.uint32, optional): Maximum capacity of each bin in the instance.
                Defaults to np.uint32(100).
            capacity_approach (capacity_approaches, optional): Literal to define how the capacities
                of the instances will be computed. If fixed, the capacity is defined as the
                maximum_capacity value. If evolved, the capacity can be updated during the evolution,
                and finally if `percentage` the capacity is defined as capacity_ratio * capacity
                during the evolution. Defaults to "fixed".
            capacity_ratio (float, optional): Capacity ratio used when the capacity_approach is set
                to percentage. It must be a float value in the range (0.0, 1.0]. Defaults to 0.8.
            seed (Optional[int | np.random.SeedSequence], optional): Seed for the random number engine. Defaults to None.

        Raises:
            ValueError: If the minimum_weight or maximum_weight are negative
            ValueError: If the minimum_weight > maximum_weight
            ValueError: If the maximum_capacity is not a valid integer, or it's <= 0
            ValueError: If the capacity_approach is not available
            ValueError: If the capacity_ratio is not a float or it's outside the range (0.0, 1.0]
        """
        try:
            self.number_of_items = int(number_of_items)
            if self.number_of_items <= 0:
                raise ValueError(
                    "number_of_items must be a "
                    "postive integer in BPPDomain. "
                    f"Got: {number_of_items}"
                )
        except (TypeError, ValueError) as exc:
            raise ValueError from exc

        try:
            self._minimum_weight = int(minimum_weight)
            self._maximum_weight = int(maximum_weight)
            # If we have negative bounds or min > max raise ValueError
            if (
                minimum_weight < 0
                or maximum_weight < 0
                or minimum_weight > maximum_weight
            ):
                raise ValueError()

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid min_i and/or max_i in BPPDomain."
                f"Expected min_i ({minimum_weight}) to be >= 0 and < max_i ({maximum_weight}) "
                f"and max_i to be >= 0."
            ) from exc

        try:
            self._max_capacity = int(maximum_capacity)
            self._capacity_ratio = float(capacity_ratio)
            if maximum_capacity <= 0:
                raise ValueError("invalid max_capacity value")
            if self._capacity_ratio <= 0 or self._capacity_ratio > 1:
                raise ValueError("invalid capacity_ratio value")

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid maximum capacity and/or capacity ratio for BPPDomain. "
                f"Capacity ({maximum_capacity}) was expected to be a positive integer, "
                f"and capacity_ratio ({capacity_ratio}) must be a float in the range (0.0, 1.0]."
            ) from exc

        if capacity_approach not in self.capacity_approaches.__args__:
            invalid_approach_msg = (
                f"The capacity approach {capacity_approach} is not available. "
                f" Please, consider choosing between {self.capacity_approaches.__args__}. "
                f" Set `evolved` approach set as fallback."
            )
            warnings.warn(invalid_approach_msg, RuntimeWarning)
            self._capacity_approach = "fixed"
        else:
            self._capacity_approach = capacity_approach

        bounds = [(1.0, self._max_capacity)] + [
            (self._minimum_weight, self._maximum_weight) for _ in range(number_of_items)
        ]
        features_names = "mean,std,median,max,min,tiny,small,medium,large,huge".split(
            ","
        )

        super().__init__(
            dimension=self.number_of_items + 1,
            bounds=bounds,
            domain_name="BPP",
            features_names=features_names,
            seed=seed,
        )

    def __str__(self):
        if self._capacity_approach != "evolved":
            capacity_str = f"\n    method: {self.capacity_approach}\n    ratio: {self.capacity_ratio}\n"
        else:
            capacity_str = "evolved\n"
        return (
            "Bin Packing Domain:\n"
            f"  number of items: {self._number_of_items}\n"
            f"  weight ranges: ({self._minimum_weight}, {self._maximum_weight})\n"
            f"  maximum capacity allowed: {self._maximum_capacity}\n"
            f"  capacity generation: {capacity_str}"
            f"  features: {','.join(self.features_names)}"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def capacity_approach(self):
        return self._capacity_approach

    @property
    def capacity_ratio(self):
        if self._capacity_approach == "percentage":
            return self._capacity_ratio
        else:
            return 1.0

    @property
    def maximum_capacity(self) -> int:
        return self._max_capacity

    @property
    def minimum_weight(self) -> int:
        return self._minimum_weight

    @property
    def maximum_weight(self) -> int:
        return self._maximum_weight

    def generate_instances(self, n: np.uint32 = np.uint32(1)) -> Sequence[Instance]:
        """Generates N new instances for the BPP domain.

        Args:
            n (int, optional): Number of instances to generate. Defaults to 1.

        Returns:
            List[Instance]: A list of Instance objects created from the raw numpy generation
        """
        # Dimension is set correctly to number_of_items + 1 to
        # allow the random generation of capacities
        instances = self._rng.integers(
            low=self._minimum_weight,
            high=self._maximum_weight,
            size=(n, self._dimension),
            dtype=int,
        )
        # Sets the capacity according to the method
        match self.capacity_approach:
            case "evolved":
                instances[:, 0] = self._rng.integers(1, self._max_capacity, size=n)
            case "percentage":
                instances[:, 0] = (
                    np.sum(instances[:, 1:], axis=1, dtype=np.int32)
                    * self._capacity_ratio
                )
            case "fixed":
                instances[:, 0] = self._max_capacity

        return list(Instance(variables=i) for i in instances)

    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        """Extract the features of the instance based on the BPP domain.

        For the BPP domain, the features consist of:
           - N as the number of items,
           - Capacity as the maximum capacity of each bin, MeanWeights of the items,
           - MedianWeights of the items, VarianceWeights of the weights of the items,
           - MaxWeight of the items in the instance, MinWeight of the items,
           - Huge as the ratio of items with normalised weights > 0.5,
           - Large as the ratio of items with normalised weights between 0.333 and 0.5,
           - Medium as the ratio of items with normalised weights between 0.25 and 0.333,
           - Small as the ratio of items with normalised weights >= 0.25,
           - Tiny as the ratio of items with normalised weights >= 0.1

        Args:
            instances (Instance): Instances to extract the features from

        Returns:
            np.ndarray: Values of each feature
        """
        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances)

        norm_variables = np.asarray(instances, copy=True, dtype=np.float64)
        norm_variables[:, 1:] = norm_variables[:, 1:] / norm_variables[:, 0:1]
        huge = 0.5
        medium = 0.33333
        large = 0.25
        tiny = 0.1
        return np.column_stack(
            [
                np.mean(norm_variables, axis=1),
                np.std(norm_variables, axis=1),
                np.median(norm_variables, axis=1),
                np.max(norm_variables, axis=1),
                np.min(norm_variables, axis=1),
                np.mean(norm_variables > huge, axis=1),  # Huge
                np.mean((huge >= norm_variables) & (norm_variables > medium), axis=1),
                np.mean((medium >= norm_variables) & (norm_variables > large), axis=1),
                np.mean(large >= norm_variables, axis=1),  # Small
                np.mean(tiny >= norm_variables, axis=1),  # Tiny
            ],
        ).astype(np.float64)

    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Dict[str, np.float64]]:
        """Creates a dictionary with the features of the instances.

        The key are the names of each feature and the values are
        the values extracted from instance.
        For the BPP domain, the features consist of:
           - N as the number of items,
           - Capacity as the maximum capacity of each bin, MeanWeights of the items,
           - MedianWeights of the items, VarianceWeights of the weights of the items,
           - MaxWeight of the items in the instance, MinWeight of the items,
           - Huge as the ratio of items with normalised weights > 0.5,
           - Large as the ratio of items with normalised weights between 0.333 and 0.5,
           - Medium as the ratio of items with normalised weights between 0.25 and 0.333,
           - Small as the ratio of items with normalised weights >= 0.25,
           - Tiny as the ratio of items with normalised weights >= 0.1

        Args:
            instances (Sequence[Instance]): Instances to extract the features from.

        Returns:
            Dict[str, np.float64]: Dictionary with the names/values of each feature
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
    ) -> List[BPP]:
        """Generates BPP problems from the given instances


        This method is used to generate a collection of (objects)
        of the BPP class ready to be solved from the definition of the instances.

        Args:
            instances (Sequence[Instance] | np.ndarray): Instances to generate
                the problems from.

        Returns:
            List[BPP]: List of BPP objects created from the instances
        """
        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances)

        # Assume evolved capacities
        capacities = instances[:, 0].astype(np.int32)
        match self.capacity_approach:
            case "percentage":
                capacities[:] = (
                    np.sum(instances[:, 1:], axis=1) * self._capacity_ratio
                ).astype(np.int32)
                instances[:, 0] = capacities[:]
            case "fixed":
                capacities[:] = self._max_capacity
                instances[:, 0] = self._max_capacity
        # The first item of each valid BPP instance is the capacity
        print(capacities)
        return list(
            BPP(items=instances[i, 1:], maximum_capacity=capacities[i])
            for i in range(len(instances))
        )
