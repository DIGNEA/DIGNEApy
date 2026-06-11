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
from typing import Dict, Optional, Tuple

import numpy as np

from ._instance import Instance
from ._problem import Problem


class Domain(ABC):
    """Domain is a class that defines the domain of the problem.

    The domain is defined by its dimension and the bounds of each variable.
    """

    def __init__(
        self,
        dimension: np.uint32,
        bounds: Sequence[Tuple],
        dtype=np.float64,
        domain_name: str = "Domain",
        features_names: Optional[Sequence[str]] = [],
        seed: Optional[int | np.random.SeedSequence] = None,
        *args,
        **kwargs,
    ):

        if (
            not isinstance(dimension, (int, np.integer, np.unsignedinteger))
            or dimension <= 0
        ):
            raise ValueError(
                f"Cannot create a Domain({domain_name}) with negative or equal to zero dimensions. Got {dimension}."
            )

        self.__name__ = domain_name
        self._dimension = dimension
        self._bounds = bounds
        self._dtype = dtype
        self.features_names = features_names
        print(self.features_names)
        if len(self._bounds) != 0:
            ranges = list(zip(*bounds))
            self._lbs = np.asarray(ranges[0], dtype=dtype)
            self._ubs = np.asarray(ranges[1], dtype=dtype)

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def generate_instances(self, n: np.uint32 = np.uint32(1)) -> Sequence[Instance]:
        """Generates N instances for the domain.

        Args:
            n (int, optional): Number of instances to generate. Defaults to 1.

        Returns:
            List[Instance]: A list of Instance objects created from the raw numpy generation
        """
        raise NotImplementedError(
            "generate_n_instances is not implemented in Domain class."
        )

    @abstractmethod
    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> Sequence[Problem]:
        msg = "generate_problems_from_instances is not implemented in Domain class."
        raise NotImplementedError(msg)

    @abstractmethod
    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        """Extract the features of the instances based on the domain

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple: Values of each feature
        """
        msg = "extract_features is not implemented in Domain class."
        raise NotImplementedError(msg)

    @abstractmethod
    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> Sequence[Dict]:
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

    @property
    def bounds(self):
        return self._bounds

    @property
    def lbs(self) -> np.ndarray:
        return self._lbs

    @property
    def ubs(self) -> np.ndarray:
        return self._ubs

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
