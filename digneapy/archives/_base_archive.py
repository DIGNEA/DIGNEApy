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
import operator
from collections.abc import Sequence
from typing import Optional, List, Self

import numpy as np

from digneapy._core import Instance


class Archive:
    """Class Archive
    Stores a collection of diverse Instances
    """

    def __init__(
        self,
        threshold: float,
        instances: Optional[Sequence[Instance]] = None,
        dtype=np.float64,
    ):
        """Creates an instance of a Archive (unstructured) for QD algorithms

        Args:
            threshold (float): Minimum value of sparseness to include an Instance into the archive.
            instances (Iterable[Instance], optional): Instances to initialise the archive. Defaults to None.
        """
        self._storage = {"instances": [], "descriptors": []}

        if instances:
            self._storage["instances"].extend(instances)
            self._storage["descriptors"].extend(
                np.asarray([instance.descriptor for instance in instances])
            )

        self._threshold = threshold
        self._dtype = dtype

    @property
    def instances(self) -> Sequence[Instance]:
        return self._storage["instances"]

    @property
    def descriptors(self) -> np.ndarray:
        return np.asarray(self._storage["descriptors"])

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, t: float):
        try:
            t_f = float(t)
        except Exception:
            msg = f"The threshold value {t} is not a float in 'threshold' setter of class {self.__class__.__name__}"
            raise TypeError(msg)
        self._threshold = t_f

    def __iter__(self):
        return iter(self._storage["instances"])

    def __str__(self):
        return f"Archive(threshold={self._threshold},data=(|{len(self)}|))"

    def __repr__(self):
        return f"Archive(threshold={self._threshold},data=(|{len(self)}|))"

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Creates a ndarray with the descriptors

        >>> import numpy as np
        >>> descriptors = [list(range(d, d + 5)) for d in range(10)]
        >>> archive = Archive(descriptors)
        >>> np_archive = np.array(archive)
        >>> assert len(np_archive) == len(archive)
        >>> assert type(np_archive) == type(np.zeros(1))
        """
        return np.asarray(self._storage["instances"], dtype=dtype, copy=copy)

    def __eq__(self, other: Self):
        """Compares whether to Archives are equal

        >>> import copy
        >>> variables = [list(range(d, d + 5)) for d in range(10)]
        >>> instances = [Instance(variables=v, s=1.0) for v in variables]
        >>> archive = Archive(threshold=0.0, instances=instances)
        >>> empty_archive = Archive(threshold=0.0)

        >>> a1 = copy.copy(archive)
        >>> assert a1 == archive
        >>> assert empty_archive != archive
        """
        return len(self) == len(other) and all(
            np.array_equal(a, b)
            for a, b in zip(self._storage["descriptors"], other._storage["descriptors"])
        )

    def __hash__(self):
        from functools import reduce

        hashes = (hash(i) for i in self.instances)
        return reduce(lambda a, b: a ^ b, hashes, 0)

    def __bool__(self):
        """Returns True if len(self) > 1

        >>> descriptors = [list(range(d, d + 5)) for d in range(10)]
        >>> archive = Archive(threshold=0.0, instances=descriptors)
        >>> empty_archive = Archive(threshold=0.0)

        >>> assert archive
        >>> assert not empty_archive
        """
        return len(self) != 0

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, key):
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self._threshold, self.instances[key])
        index = operator.index(key)
        return self._storage["instances"][index]

    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
    ):
        """Extends the current archive with all the individuals inside iterable that have
        a sparseness value greater than the archive threshold.

        Args:
            instances (Sequence[Instance]): Sequence of instances to be include in the archive.
        """
        scores = (
            novelty_scores
            if novelty_scores is not None
            else np.asarray([instance.s for instance in instances])
        )

        descriptors = (
            descriptors
            if descriptors is not None
            else np.asarray([instance.descriptor for instance in instances])
        )
        to_insert = np.where(scores >= self.threshold)[0]
        self._storage["instances"].extend((instances[i] for i in to_insert))
        self._storage["descriptors"].extend(descriptors[to_insert])

    def __format__(self, fmt_spec=""):
        variables = self
        outer_fmt = "({})"
        components = (format(c, fmt_spec) for c in variables)
        return outer_fmt.format(", ".join(components))

    def asdict(self) -> dict:
        return {
            "threshold": self._threshold,
            "instances": {
                i: instance.asdict()
                for i, instance in enumerate(self._storage["instances"])
            },
        }

    def to_json(self) -> str:
        """Converts the archive into a JSON object

        Returns:
            str: JSON str of the archive content
        """

        return json.dumps(self.asdict(), indent=4)
