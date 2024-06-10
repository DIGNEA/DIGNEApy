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


from digneapy.core.instance import Instance
from digneapy.core.problem import Problem

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Mapping, Tuple, Optional


class Domain(ABC):
    def __init__(
        self,
        name: str = "Domain",
        dimension: int = 0,
        bounds: Optional[Sequence[Tuple]] = None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.dimension = dimension
        self.bounds = bounds if bounds else [(0.0, 0.0)]

    @abstractmethod
    def generate_instance(self) -> Instance:
        """Generates a new instances for the domain

        Returns:
            Instance: New randomly generated instance
        """
        msg = "generate_instances is not implemented in Domain class."
        raise NotImplementedError(msg)

    @abstractmethod
    def extract_features(self, instance: Instance) -> Tuple:
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

    def __len__(self):
        return self.dimension

    def lower_i(self, i):
        if i < 0 or i > len(self.bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self.bounds)}]"
            raise AttributeError(msg)
        return self.bounds[i][0]

    def upper_i(self, i):
        if i < 0 or i > len(self.bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self.bounds)}]"
            raise AttributeError(msg)
        return self.bounds[i][1]
