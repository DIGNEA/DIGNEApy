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
from collections.abc import Callable, Iterable
from functools import reduce
from typing import Optional

import numpy as np

from digneapy.core import Instance


class Archive:
    """Class Archive
    Stores a collection of diverse Instances
    """

    def __init__(
        self, threshold: float, instances: Optional[Iterable[Instance]] = None
    ):
        """Creates an instance of a Archive (unstructured) for QD algorithms

        Args:
            threshold (float): Minimum value of sparseness to include an Instance into the archive.
            instances (Iterable[Instance], optional): Instances to initialise the archive. Defaults to None.
        """
        if instances:
            self._instances = list(i for i in instances)
        else:
            self._instances = []

        self._threshold = threshold

    @property
    def instances(self):
        return self._instances

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
        return iter(self._instances)

    def __str__(self):
        return f"Archive(threshold={self._threshold},data=(|{len(self)}|))"

    def __repr__(self):
        return f"Archive(threshold={self._threshold},data=(|{len(self)}|))"

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

    def __eq__(self, other):
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
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __hash__(self):
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
        return self.instances[index]

    def append(self, i: Instance):
        if isinstance(i, Instance):
            self.instances.append(i)
        else:
            msg = f"Only objects of type {Instance.__class__.__name__} can be inserted into an archive"
            raise TypeError(msg)

    def __default_filter(self, instance: Instance):
        return instance.s >= self._threshold

    def extend(
        self, iterable: Iterable[Instance], filter_fn: Optional[Callable] = None
    ):
        """Extends the current archive with all the individuals inside iterable that have
        a sparseness value greater than the archive threshold.

        Args:
            iterable (Iterable[Instance]): Iterable of instances to be include in the archive.
            filter_fn (Callable, optional): A function that takes an instance and returns a boolean.
                                             Defaults to filtering by sparseness.
        """
        if not all(isinstance(i, Instance) for i in iterable):
            msg = "Only objects of type Instance can be inserted into an archive"
            raise TypeError(msg)

        default_filter = self.__default_filter
        actual_filter = filter_fn if filter_fn is not None else default_filter

        for i in filter(actual_filter, iterable):
            self.instances.append(i)

    def __format__(self, fmt_spec=""):
        variables = self
        outer_fmt = "({})"
        components = (format(c, fmt_spec) for c in variables)
        return outer_fmt.format(", ".join(components))

    def to_json(self) -> str:
        """Converts the archive into a JSON object

        Returns:
            str: JSON str of the archive content
        """
        data = {"threshold": self._threshold, "instances": self._instances}
        return json.dumps(data, indent=4)
