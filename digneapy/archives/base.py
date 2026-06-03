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
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Optional

import numpy as np

from .._core import Instance

type NestedDict = dict[int, Instance | np.ndarray]


class Keys(Enum):
    instances = "instances"
    descriptors = "descriptors"
    grid = "grid"
    objectives = "objectives"


class Archive(ABC):
    """Class Archive Stores a collection of Instances"""

    def __init__(
        self,
        initial_instances: Optional[Sequence[Instance]] = None,
        dtype=np.float64,
    ):
        """Creates an instance of a Archive (unstructured) for QD algorithms

        Args:
            threshold (float): Minimum value of sparseness to include an Instance into the archive.
            instances (Iterable[Instance], optional): Instances to initialise the archive. Defaults to None.
        """
        self._storage: dict[Keys, list] = {
            Keys.instances: [],
            Keys.descriptors: [],
        }
        if initial_instances is not None and len(initial_instances) > 0:
            for instance in enumerate(initial_instances):
                if isinstance(instance, Instance):
                    self._storage[Keys.instances].append(instance)
                    self._storage[Keys.descriptors].append(instance.descriptor)
                else:
                    raise TypeError(
                        f"Initial instances must be of type Instance. Got {instance}."
                    )

        self._dtype = dtype

    def purge_unfeasible(self, attr: str = "p") -> None:
        """Removes all the unfeasible instances from the grid"""
        for i, instance in self._storage[Keys.instances]:
            if getattr(instance, attr) < 0:
                del self._storage[Keys.instances][i]
                del self._storage[Keys.descriptors][i]

    @abstractmethod
    def extend(
        self,
        instances: Sequence[Instance],
        *args,
        **kwargs,
    ):
        raise NotImplementedError("To be implemented in subclasses")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("To be implemented in subclasses")

    @property
    def instances(self) -> Iterable[Instance]:
        return self._storage[Keys.instances]

    @property
    def descriptors(self) -> np.ndarray:
        return np.asarray(self._storage[Keys.descriptors])

    def __iter__(self):
        return iter(self._storage[Keys.instances])

    def __str__(self):
        return f"Archive(|{len(self)}|)"

    def __repr__(self):
        return f"Archive(|{len(self)}|)"

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Creates a ndarray with the instances"""
        return np.asarray(self._storage[Keys.instances], dtype=dtype, copy=copy)

    def __eq__(self, other):
        """Compares whether to Archives are equal"""
        return len(self) == len(other) and all(
            np.array_equal(a, b)
            for a, b in zip(
                self._storage[Keys.descriptors], other._storage[Keys.descriptors]
            )
        )

    def __hash__(self):
        from functools import reduce

        hashes = (hash(i) for i in self.instances)
        return reduce(lambda a, b: a ^ b, hashes, 0)

    def __bool__(self):
        """Returns True if len(self) >= 1"""
        return len(self) != 0

    def __len__(self):
        return len(self._storage[Keys.instances])

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError("Must be implemented in the subclasses")

    def __format__(self, fmt_spec=""):
        variables = self
        outer_fmt = "({})"
        components = (format(c, fmt_spec) for c in variables)
        return outer_fmt.format(", ".join(components))

    def asdict(self) -> dict:
        return {
            "instances": {
                i: instance.asdict()
                for i, instance in enumerate(self._storage[Keys.instances])
            }
        }

    def to_json(self) -> str:
        """Converts the archive into a JSON object

        Returns:
            str: JSON str of the archive content
        """

        return json.dumps(self.asdict(), indent=4)
