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
from typing import List, Callable


class Archive:
    """Class Archive
    Stores a collection of diverse Instances
    """

    _typecode = "d"

    def __init__(self, instances: Iterable[Instance] = None):
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


def _features_descriptor_strategy(iterable) -> List[float]:
    """It generates the feature descriptor of an instance

    Args:
        iterable (List[Instance]): Instances to describe

    Returns:
        List[float]: List of the feature descriptors of each instance
    """
    return [i.features for i in iterable]


def _performance_descriptor_strategy(iterable) -> List[float]:
    """It generates the performance descriptor of an instance
    based on the scores of the solvers in the portfolio over such instance

    Args:
        iterable (List[Instance]): Instances to describe

    Returns:
        List[float]: List of performance descriptors of each instance
    """
    return [np.mean(i.portfolio_scores, axis=0) for i in iterable]


def _instance_descriptor_strategy(iterable) -> List[float]:
    """It returns the instance information as its descriptor

    Args:
        iterable (List[Instance]): Instances to describe

    Returns:
        List[float]: List of descriptor instance
    """
    return [*iterable]


class NoveltySearch:
    """Descriptor strategies for the Novelty Search algorithm.
    The current version supports Features, Performance and Instance variations.
    """

    __descriptor_strategies = {
        "features": _features_descriptor_strategy,
        "performance": _performance_descriptor_strategy,
        "instance": _instance_descriptor_strategy,
    }

    def __init__(
        self,
        t_a: float = 0.001,
        t_ss: float = 0.001,
        k: int = 15,
        descriptor="features",
        transformer: Callable = None,
    ):
        """_summary_

        Args:
            t_a (float, optional): Archive threshold. Defaults to 0.001.
            t_ss (float, optional): Solution set threshold. Defaults to 0.001.
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor (str, optional): Descriptor to calculate the diversity. The options are features, performance or instance. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
        """
        self._t_a = t_a
        self._t_ss = t_ss
        self._k = k
        self._archive = Archive()
        self._solution_set = Archive()
        self._transformer = transformer
        if descriptor not in self.__descriptor_strategies:
            msg = f"describe_by {descriptor} not available in {self.__class__.__name__}.__init__. Set to features by default"
            print(msg)
            self._describe_by = "features"
            self._descriptor_strategy = _features_descriptor_strategy
        else:
            self._describe_by = descriptor
            self._descriptor_strategy = self.__descriptor_strategies[descriptor]

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
        """Combines the archive and the given population before computing the sparseness

        Args:
            current_pop (Archive): Current archive of solution.
            instances (List[Instance]): List of instances to evaluate.

        Returns:
            np.ndarray[float]: Returns an ndarray of descriptors.
        """
        components = self._descriptor_strategy(itertools.chain(instances, current_pop))
        return np.vstack([components])

    def __compute_sparseness(
        self, instances: List[Instance], current_archive: Archive, neighbours: int
    ) -> List[float]:
        """This method does the calculation of sparseness either for the archive or the solution set.
        It gets called by 'sparssenes' and 'sparseness_solution_set'. Note that this method update the `s`
        attribute of each instances with the result of the computation. It also returns a list with all the
        values for further used if necessary.

        Args:
            instances (List[Instance]): Instances to evaluate.
            current_archive (Archive): Current archive/solutio set of instances.
            neighbours (int): Number of neighbours to calculate the KNN (K + 1 or 2). Always have to add 1.

        Returns:
            List[float]: Sparseness values of each instance.
        """
        # We need to concatentate the archive to the given descriptors
        # and to set k+1 because it predicts n[0] == self descriptor
        # The _desc_arr is a ndarray which contains the descriptor of the instances
        # from the archive and the new list of instances. The order is [instances, current_archive]
        # so we can easily calculate and update `s`
        _desc_arr = self._combined_archive_and_population(current_archive, instances)
        if self._transformer is not None:
            # Transform the descriptors if necessary
            _desc_arr = self._transformer(_desc_arr)

        neighbourhood = NearestNeighbors(n_neighbors=neighbours, algorithm="ball_tree")
        neighbourhood.fit(_desc_arr)
        sparseness = []
        # We're only interesed in the instances given not the archive
        frac = 1.0 / neighbours
        for instance, descriptor in zip(instances, _desc_arr[: len(instances)]):
            dist, ind = neighbourhood.kneighbors([descriptor])
            dist, ind = dist[0][1:], ind[0][1:]
            s = frac * sum(dist)
            instance.s = s
            sparseness.append(s)

        return sparseness

    def sparseness(
        self,
        instances: List[Instance],
    ) -> List[float]:
        """Calculates the sparseness of the given instances against the individuals
        in the Archive.

        Args:
            instances (List[Instance]): Instances to calculate their sparseness
            verbose (bool, optional): Flag to show the progress. Defaults to False.

        Raises:
            AttributeError: If len(d) where d is the descriptor of each instance i differs from another
            AttributeError: If NoveltySearch.k >= len(instances)

        Returns:
            List[float]: List of sparseness values, one for each instance
        """
        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness on an empty Instance list"
            raise AttributeError(msg)

        if self._k >= len(instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness with k({self._k}) > len(instances)({len(instances)})"
            raise AttributeError(msg)

        return self.__compute_sparseness(
            instances, self.archive, neighbours=self._k + 1
        )

    def sparseness_solution_set(self, instances: List[Instance]) -> List[float]:
        """Calculates the sparseness of the given instances against the individuals
        in the Solution Set.

        Args:
            instances (List[Instance]): Instances to calculate their sparseness

        Raises:
            AttributeError: If len(d) where d is the descriptor of each instance i differs from another
            AttributeError: If 2 >= len(instances)

        Returns:
            List[float]: List of sparseness values, one for each instance
        """

        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{self.__class__.__name__} trying to update the solution set with an empty instance list"
            raise AttributeError(msg)

        if len(instances) <= 2:
            msg = f"{self.__class__.__name__} trying to calculate sparseness_solution_set with k(2) >= len(instances)({len(instances)})"
            raise AttributeError(msg)

        return self.__compute_sparseness(instances, self.solution_set, neighbours=2)

    def _update_archive(self, instances: List[Instance]):
        """Updates the Novelty Search Archive with all the instances that has a 's' greater than t_a"""
        if not instances:
            return
        self._archive.extend(filter(lambda x: x.s >= self.t_a, instances))

    def _update_solution_set(self, instances: List[Instance]):
        """Updates the Novelty Search Solution Set with all the instances that has a 's' greater than t_ss"""
        if not instances:
            return
        self.solution_set.extend(filter(lambda x: x.s >= self.t_ss, instances))
