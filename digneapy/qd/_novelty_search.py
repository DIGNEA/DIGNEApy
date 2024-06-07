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
import itertools

from digneapy.archives import Archive
from digneapy.core import Instance
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Optional
from collections.abc import Sequence, Iterable

from digneapy.qd.desc_strategies import (
    features_strategy,
    instance_strategy,
    performance_strategy,
)


class NS:
    """Descriptor strategies for the Novelty Search algorithm.
    The current version supports Features, Performance and Instance variations.
    """

    __descriptor_strategies = {
        "features": features_strategy,
        "performance": performance_strategy,
        "instance": instance_strategy,
    }

    def __init__(
        self,
        archive: Optional[Archive] = None,
        s_set: Optional[Archive] = None,
        k: int = 15,
        descriptor="features",
        transformer: Optional[Callable[[Sequence | Iterable], np.ndarray]] = None,
    ):
        """Creates an instance of the NoveltySearch Algorithm
        Args:
            archive (Archive, optional): Archive to store the instances to guide the evolution. Defaults to Archive(threshold=0.001)..
            s_set (Archive, optional): Solution set to store the instances. Defaults to Archive(threshold=0.001).
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor (str, optional): Descriptor to calculate the diversity. The options are features, performance or instance. Defaults to "features".
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
        """
        self._archive = archive if archive is not None else Archive(threshold=0.001)
        self._solution_set = s_set if s_set is not None else Archive(threshold=0.001)
        self._k = k
        self._transformer = transformer

        if descriptor not in self.__descriptor_strategies:
            msg = f"describe_by {descriptor} not available in {self.__class__.__name__}.__init__. Set to features by default"
            print(msg)
            self._describe_by = "features"
            self._descriptor_strategy = features_strategy
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

    def __str__(self):
        return f"NS(descriptor={self._describe_by},k={self._k},A={self._archive},S_S={self._solution_set})"

    def __repr__(self) -> str:
        return f"NS<descriptor={self._describe_by},k={self._k},A={self._archive},S_S={self._solution_set}>"

    def _combined_archive_and_population(
        self, current_pop: Archive, instances: Sequence[Instance]
    ) -> np.ndarray[float]:
        """Combines the archive and the given population before computing the sparseness

        Args:
            current_pop (Archive): Current archive of solution.
            instances (Sequence[Instance]): Sequence of instances to evaluate.

        Returns:
            np.ndarray[float]: Returns an ndarray of descriptors.
        """
        components = self._descriptor_strategy(itertools.chain(instances, current_pop))
        return np.vstack([components])

    def __compute_sparseness(
        self, instances: Sequence[Instance], current_archive: Archive, neighbours: int
    ) -> list:
        """This method does the calculation of sparseness either for the archive or the solution set.
        It gets called by 'sparssenes' and 'sparseness_solution_set'. Note that this method update the `s`
        attribute of each instances with the result of the computation. It also returns a list with all the
        values for further used if necessary.

        Args:
            instances (Sequence[Instance]): Instances to evaluate.
            current_archive (Archive): Current archive/solutio set of instances.
            neighbours (int): Number of neighbours to calculate the KNN (K + 1 or 2). Always have to add 1.

        Returns:
            list[float]: Sparseness values of each instance.
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
        instances: Sequence[Instance],
    ) -> list[float]:
        """Calculates the sparseness of the given instances against the individuals
        in the Archive.

        Args:
            instances (Sequence[Instance]): Instances to calculate their sparseness
            verbose (bool, optional): Flag to show the progress. Defaults to False.

        Raises:
            AttributeError: If len(d) where d is the descriptor of each instance i differs from another
            AttributeError: If NoveltySearch.k >= len(instances)

        Returns:
            list[float]: List of sparseness values, one for each instance
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

    def sparseness_solution_set(self, instances: Sequence[Instance]) -> list[float]:
        """Calculates the sparseness of the given instances against the individuals
        in the Solution Set.

        Args:
            instances (Sequence[Instance]): Instances to calculate their sparseness

        Raises:
            AttributeError: If len(d) where d is the descriptor of each instance i differs from another
            AttributeError: If 2 >= len(instances)

        Returns:
            list[float]: List of sparseness values, one for each instance
        """

        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{self.__class__.__name__} trying to update the solution set with an empty instance list"
            raise AttributeError(msg)

        if len(instances) <= 2:
            msg = f"{self.__class__.__name__} trying to calculate sparseness_solution_set with k = 2 >= len(instances)({len(instances)})"
            raise AttributeError(msg)

        return self.__compute_sparseness(instances, self.solution_set, neighbours=2)
