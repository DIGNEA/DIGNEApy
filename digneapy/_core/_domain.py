#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   domain.py
@Time    :   2024/06/07 14:08:47
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Mapping, Optional

import numpy as np

from digneapy._core._instance import Instance
from digneapy._core._problem import Problem
from .types import RNG


class Domain(ABC, RNG):
    """Domain is a class that defines the domain of the problem.
    The domain is defined by its dimension and the bounds of each variable.

    Args:
        RNG: Subclass that implements the RNG protocol
    """
    def __init__(
        self,
        dimension: int,
        bounds: Sequence[tuple],
        dtype=np.float64,
        seed: int = 42,
        name: str = "Domain",
        feat_names: Optional[Sequence[str]] = None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.__name__ = name
        self._dimension = dimension
        self._bounds = bounds
        self._dtype = dtype
        self.feat_names = feat_names if feat_names else list()
        self.initialize_rng(seed=seed)
        if len(self._bounds) != 0:
            ranges = list(zip(*bounds))
            self._lbs = np.array(ranges[0], dtype=dtype)
            self._ubs = np.array(ranges[1], dtype=dtype)

    @abstractmethod
    def generate_instance(self) -> Instance:
        """Generates a new instances for the domain

        Returns:
            Instance: New randomly generated instance
        """
        msg = "generate_instances is not implemented in Domain class."
        raise NotImplementedError(msg)

    @abstractmethod
    def extract_features(self, instances: Instance) -> tuple:
        """Extract the features of the instance based on the domain

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple: Values of each feature
        """
        msg = "extract_features is not implemented in Domain class."
        raise NotImplementedError(msg)

    @abstractmethod
    def extract_features_as_dict(self, instance: Instance) -> Mapping[str, float]:
        """Creates a dictionary with the features of the instance.
        The key are the names of each feature and the values are
        the values extracted from instance.

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Mapping[str, float]: Dictionary with the names/values of each feature
        """
        msg = "extract_features_as_dict is not implemented in Domain class."
        raise NotImplementedError(msg)

    @abstractmethod
    def from_instance(self, instance: Instance) -> Problem:
        msg = "from_instance is not implemented in Domain class."
        raise NotImplementedError(msg)

    @property
    def bounds(self):
        return self._bounds

    def get_bounds_at(self, i: int) -> tuple:
        if i < 0 or i > len(self._bounds):
            raise ValueError(
                f"Index {i} out-of-range. The bounds are 0-{len(self._bounds)} "
            )
        return (self._lbs[i], self._ubs[i])

    @property
    def dimension(self):
        return self._dimension

    def __len__(self):
        return self._dimension
