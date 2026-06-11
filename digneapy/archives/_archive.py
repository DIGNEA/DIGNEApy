#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _base_archive.py
@Time    :   2024/06/07 12:17:34
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Optional

import numpy as np

from .._core import Instance
from ._utils import check_valid_instance_batch


class Keys(Enum):
    instances = "instances"
    descriptors = "descriptors"
    grid = "grid"
    objectives = "objectives"


class Archive(ABC):
    """Base Archive

    Works as a foundation for all the types of archives allowed in Digneapy.
    """

    def __init__(
        self,
        initial_instances: Optional[Sequence[Instance]] = None,
    ):
        """Creates an instance of a Archive (unstructured) for QD algorithms

        Args:
            initial_instances (Sequence[Instance], optional): Instances to initialise the archive. Defaults to None.
        """
        self._storage: dict[Keys, list] = {
            Keys.instances: [],
            Keys.descriptors: [],
        }
        if initial_instances is not None and len(initial_instances) > 0:
            if check_valid_instance_batch(initial_instances):
                for instance in initial_instances:
                    self._storage[Keys.instances].append(instance)
                    self._storage[Keys.descriptors].append(instance.descriptor)
            else:
                raise TypeError(
                    "All objects of initial_instances must be of type Instance."
                )

    @property
    def instances(self) -> Sequence[Instance]:
        """Instances of the archive

        Returns:
            Sequence[Instance]: Sequence of instances stored in the archive.
        """
        return self._storage[Keys.instances]

    @property
    def descriptors(self) -> np.ndarray:
        """Descriptors of the instances

        Returns:
            np.ndarray: Returns a np.ndarray with the descriptors of
                the instances stored in the archive
        """
        return np.asarray(self._storage[Keys.descriptors])

    @abstractmethod
    def extend(
        self,
        instances: Sequence[Instance],
        *args,
        **kwargs,
    ):
        """Extends the archive with a collection of instances.

        This method must be implemented by each subclass of Archive.
        Args:
            instances (Sequence[Instance]): Collection of instances to insert in the archive.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("To be implemented in subclasses")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}")

    def __iter__(self) -> Iterator[Instance]:
        """Iterator of the Archive.

        Allows users to iterate the instances of the archive

        Returns:
            Iterator: Returns an Iterator of Instances stored
        """
        return iter(self._storage[Keys.instances])

    def __str__(self):
        return f"Archive(|{len(self)}|)"

    def __repr__(self):
        return self.__str__()

    def __array__(self, dtype=object, copy: Optional[bool] = None) -> np.ndarray:
        """Cast the Archive to an array

        Args:
            dtype (optional): Data type of the resulting array. Defaults to object.
            copy (bool, optional): Whether to copy the objects or reference them. Defaults to None.

        Returns:
            np.ndarray: Array with the instances stored in the archive
        """
        return np.asarray(self._storage[Keys.instances], dtype=object, copy=copy)

    def __eq__(self, other: Archive) -> bool:
        """Compares two archives

        Args:
            other (Archive): Other archive type to compare

        Returns:
            bool: Returns True if both archives have the same amount of instances
                and all of those instances are the same when compared Inst_a == Inst_b.
        """
        if not isinstance(other, Archive):
            raise TypeError(f"Archives cannot be compared with {other.__class__}.")

        self_cls = self.__class__
        other_cls = other.__class__
        if self_cls != other_cls:
            raise TypeError(
                f"Cannot compared an Archive of class {self_cls} with another of class {other_cls}. Use the same classes."
            )

        return len(self) == len(other) and all(
            a == b
            for a, b in zip(
                self._storage[Keys.instances], other._storage[Keys.instances]
            )
        )

    def __len__(self) -> int:
        """Length of the Archive

        Returns:
            int: Number of instances stored in the archive
        """
        return len(self._storage[Keys.instances])

    def to_dict(self) -> dict:
        """Converts the archive into a dictionary

        This method could be extended in the subclasses to include
        extra information. In this class, in only includes the instances.

        Returns:
            dict: Dictionary with the instances stored in the archive
        """
        return {
            "instances": {
                i: instance.to_dict()
                for i, instance in enumerate(self._storage[Keys.instances])
            }
        }

    def to_json(self) -> str:
        """Converts the archive into a JSON object

        Returns:
            str: JSON str of the archive content
        """
        # Todo: Need to check the NumPy datatypes and JSON encoder
        return json.dumps(self.to_dict(), indent=4)
