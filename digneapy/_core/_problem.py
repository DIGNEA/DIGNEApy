#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   problem.py
@Time    :   2024/06/07 14:07:55
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Tuple, TypeVar, Optional, Any

import numpy as np
from numpy import typing as npt

from ._instance import Instance
from ._solution import Solution
from .types import RNG


class Problem(ABC, RNG):
    def __init__(
        self,
        dimension: int,
        bounds: Sequence[tuple],
        name: str = "DefaultProblem",
        dtype=np.float64,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """Creates a new problem instance.
        The problem is defined by its dimension and the bounds of each variable.

        Args:
            dimension (int): Number of variables in the problem
            bounds (Sequence[tuple]): Bounds of each variable in the problem
            name (str, optional): Name of the problem for printing and logging purposes. Defaults to "DefaultProblem".
            dtype (_type_, optional): Type of the variables. Defaults to np.float64.
            seed (int, optional): Seed for the RNG. Defaults to 42.
        """
        self._name = name
        self.__name__ = name
        self._dimension = dimension
        self._bounds = bounds
        self._dtype = dtype
        self.initialize_rng(seed=seed)
        if len(self._bounds) != 0:
            ranges = list(zip(*bounds))
            self._lbs = np.array(ranges[0], dtype=dtype)
            self._ubs = np.array(ranges[1], dtype=dtype)

    @property
    def dimension(self):
        return self._dimension

    @property
    def bounds(self):
        return self._bounds

    def get_bounds_at(self, i: int) -> tuple:
        if i < 0 or i > len(self._bounds):
            raise ValueError(
                f"Index {i} out-of-range. The bounds are 0-{len(self._bounds)} "
            )
        return (self._lbs[i], self._ubs[i])

    @abstractmethod
    def create_solution(self) -> Solution | np.ndarray:
        """Creates a random solution to the problem.
        This method can be used to initialise the solutions
        for any algorithm
        """
        msg = "create_solution method not implemented in Problem"
        raise NotImplementedError(msg)

    @abstractmethod
    def __array__(
        self, dtype: Any = None, copy: Optional[bool] = None
    ) -> npt.ArrayLike:
        msg = "__array__ method not implemented in Problem"
        raise NotImplementedError(msg)

    @abstractmethod
    def evaluate(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        """Evaluates the candidate individual with the information of the Knapsack

        Args:
            individual (Sequence | Solution | np.ndarray): Individual to evaluate

        Raises:
            ValueError: Raises an error if the len(individual) != len(instance) / 2

        Returns:
            Tuple[float]: fitness
        """
        msg = "evaluate method not implemented in Problem"
        raise NotImplementedError(msg)

    @abstractmethod
    def __call__(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        msg = "__call__ method not implemented in Problem"
        raise NotImplementedError(msg)

    @abstractmethod
    def to_instance(self) -> Instance:
        """Creates an instance from the information of the problem.
        This method is used in the generators to create instances to evolve
        """
        msg = "to_instance method not implemented in Problem"
        raise NotImplementedError(msg)

    @abstractmethod
    def to_file(self, filename: str):
        msg = "to_file method not implemented in Problem"
        raise NotImplementedError(msg)

    @classmethod
    def from_file(cls, filename: str):
        msg = "from_file method not implemented in Problem"
        raise NotImplementedError(msg)


P = TypeVar("P", bound=Problem, contravariant=True)
