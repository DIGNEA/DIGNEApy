#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   novelty_search.py
@Time    :   2023/10/24 11:23:08
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
from collections.abc import Iterable
import reprlib
import operator
import itertools
from functools import reduce
from .core import Instance
from sklearn.neighbors import NearestNeighbors
from typing import List


class Archive:
    """Class Archive
    Stores a collection of diverse Instances
    """

    _typecode = "d"

    def __init__(self, instances: list[Instance] = None):
        """_summary_

        Args:
            instances (list[Instance], optional): Instances to initialise the archive. Defaults to None.
        """
        if instances:
            # self._instances = list(array(self.typecode, d) for d in instances)
            self._instances = list(i for i in instances)
        else:
            self._instances = []

    @property
    def instances(self):
        return self._instances

    def __iter__(self):
        return iter(self._instances)

    def __repr__(self):
        components = []
        for d in self._instances:
            comp = reprlib.repr(d)
            comp = comp[comp.find("[") : -1]
            components.append(comp)
        return f"Archive({components})"

    def __str__(self):
        return f"Archive with {len(self)} instances -> {str(tuple(self))}"

    def __array__(self) -> np.ndarray:
        """Creates a ndarray with the descriptors

        >>> import numpy as np
        >>> descriptors = [list(range(d, d + 5)) for d in range(10)]
        >>> archive = Archive(descriptors)
        >>> np_archive = np.array(archive)
        >>> assert len(np_archive) == len(archive)
        >>> assert type(np_archive) == type(np.zeros(1))
        """
        return np.array(self._instances)

    def __bytes__(self):
        return bytes([ord(self._typecode)]) + bytes(self._instances)

    def __eq__(self, other):
        """Compares whether to Archives are equal

        >>> import copy
        >>> variables = [list(range(d, d + 5)) for d in range(10)]
        >>> instances = [Instance(variables=v) for v in variables]
        >>> archive = Archive(instances)
        >>> empty_archive = Archive()

        >>> a1 = copy.copy(archive)
        >>> assert a1 == archive
        >>> assert empty_archive != archive
        """
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __hash__(self):
        hashes = (hash(i) for i in self.instances)
        return reduce(lambda a, b: a ^ b, hashes, 0)

    def __bool__(self):
        """Returns True if len(self) > 1

        >>> descriptors = [list(range(d, d + 5)) for d in range(10)]
        >>> archive = Archive(descriptors)
        >>> empty_archive = Archive()

        >>> assert archive
        >>> assert not empty_archive
        """
        return len(self) != 0

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, key):
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self.instances[key])
        index = operator.index(key)
        return self.instances[index]

    def append(self, i: Instance) -> None:
        if isinstance(i, Instance):
            self.instances.append(i)
        else:
            msg = f"Only objects of type {Instance.__class__.__name__} can be inserted into an archive"
            raise AttributeError(msg)

    def extend(self, iterable: Iterable[Instance]) -> None:
        for i in iterable:
            if isinstance(i, Instance):
                self.instances.append(i)

    def __format__(self, fmt_spec=""):
        variables = self
        outer_fmt = "({})"
        components = (format(c, fmt_spec) for c in variables)
        return outer_fmt.format(", ".join(components))

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)


def _features_descriptor_strategy(iterable) -> List[float]:
    return [i.features for i in iterable]


def _performance_descriptor_strategy(iterable) -> List[float]:
    return [np.mean(i.portfolio_scores, axis=0) for i in iterable]


class NoveltySearch:
    __descriptors = ("features", "performance")

    def __init__(
        self,
        t_a: float = 0.001,
        t_ss: float = 0.001,
        k: int = 15,
        descriptor="features",
    ):
        """_summary_

        Args:
            t_a (float, optional): Archive threshold. Defaults to 0.001.
            t_ss (float, optional): Solution set threshold. Defaults to 0.001.
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor (str, optional): Descriptor to calculate the diversity. The options are features and performance. Defaults to "features".
        """
        self._t_a = t_a
        self._t_ss = t_ss
        self._k = k
        self._archive = Archive()
        self._solution_set = Archive()

        if descriptor not in self.__descriptors:
            msg = f"describe_by {descriptor} not available in {self.__class__.__name__}.__init__. Set to features by default"
            print(msg)
            self._describe_by = "features"
            self._descriptor_strategy = _features_descriptor_strategy
        else:
            self._describe_by = descriptor
            self._descriptor_strategy = (
                _performance_descriptor_strategy
                if descriptor == "performance"
                else _features_descriptor_strategy
            )

    @property
    def archive(self):
        return self._archive

    @property
    def solution_set(self):
        return self._solution_set

    @property
    def k(self):
        return self._k

    @property
    def t_a(self):
        return self._t_a

    @property
    def t_ss(self):
        return self._t_ss

    def __str__(self):
        return f"NS(desciptor={self._describe_by},t_a={self._t_a},t_ss={self._t_ss},k={self._k},len(a)={len(self._archive)},len(ss)={len(self._solution_set)})"

    def __repr__(self) -> str:
        return f"NS<desciptor={self._describe_by},t_a={self._t_a},t_ss={self._t_ss},k={self._k},len(a)={len(self._archive)},len(ss)={len(self._solution_set)}>"

    def _combined_archive_and_population(
        self, current_pop: Archive, instances: List[Instance]
    ) -> np.ndarray[float]:
        components = self._descriptor_strategy(itertools.chain(instances, current_pop))
        return np.vstack([components])

    def sparseness(
        self,
        instances: List[Instance],
        verbose: bool = False,
    ) -> List[float]:
        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness on an empty Instance list"
            raise AttributeError(msg)

        if self._k >= len(instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness with k({self._k}) > len(instances)({len(instances)})"
            raise AttributeError(msg)
        # We need to concatentate the archive to the given descriptors
        # and to set k+1 because it predicts n[0] == self descriptor
        _descriptors_arr = self._combined_archive_and_population(
            self.archive, instances
        )
        neighbourhood = NearestNeighbors(n_neighbors=self._k + 1, algorithm="ball_tree")
        neighbourhood.fit(_descriptors_arr)
        sparseness = []
        # We're only interesed in the instances given not the archive
        for instance, descriptor in zip(
            instances, _descriptors_arr[0 : len(instances)]
        ):
            dist, ind = neighbourhood.kneighbors([descriptor])
            dist, ind = dist[0][1:], ind[0][1:]
            s = (1.0 / self._k) * sum(dist)
            instance.s = s
            sparseness.append(s)
            if verbose:
                print("=" * 120)
                print(f"{instance:.3f} |  s(i) > t_a? {(s>= self._t_a)!r:15} |")

        return sparseness

    def _update_archive(self, instances: List[Instance]):
        """Updates the Novelty Search Archive with all the instances that has a 's' greater than t_a"""
        if not instances:
            return
        self._archive.extend(filter(lambda x: x.s >= self.t_a, instances))

    def _update_solution_set(self, instances: List[Instance], verbose: bool = False):
        """Updates the Novelty Search Archive with all the instances that has a 's' greater than t_ss when K is set to 1"""

        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{self.__class__.__name__} trying to update the solution set with an empty instance list"
            raise AttributeError(msg)

        if self._k >= len(instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness_solution_set with k({self._k}) > len(instances)({len(instances)})"
            raise AttributeError(msg)

        _descriptors_arr = self._combined_archive_and_population(
            self.solution_set, instances
        )

        neighbourhood = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
        neighbourhood.fit(_descriptors_arr)
        for instance, descriptor in zip(
            instances, _descriptors_arr[0 : len(instances)]
        ):
            dist, ind = neighbourhood.kneighbors([descriptor])
            dist, ind = dist[0][1:], ind[0][1:]
            s = (1.0 / self._k) * sum(dist)
            if verbose:
                print("=" * 120)
                print(f"{instance} |  s = {s} > t_ss? {(s>= self._t_ss)!r:15} |")
            if s >= self._t_ss:
                self._solution_set.append(instance)
