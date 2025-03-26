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

import heapq
from collections.abc import Sequence
from operator import attrgetter
from typing import Optional, Tuple

import numpy as np

from digneapy._core._instance import Instance
from digneapy.archives import Archive


class NS:
    """Descriptor strategies for the Novelty Search algorithm.
    The current version supports Features, Performance and Instance variations.
    """

    _EXPECTED_METRICS = "euclidean"

    def __init__(
        self,
        archive: Optional[Archive] = None,
        k: Optional[int] = 15,
        dist_metric: Optional[str] = "euclidean",
    ):
        """Creates an instance of the NoveltySearch Algorithm
        Args:
            archive (Archive, optional): Archive to store the instances to guide the evolution. Defaults to Archive(threshold=0.001).
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            dist_metric (str, optional): Defines the distance metric used by NearestNeighbor in the archives. Defaults to euclidean.
        """
        if k < 0:
            raise ValueError(
                f"{__name__} k must be a positive integer and less than the number of instances."
            )
        self._k = k
        self._archive = archive if archive is not None else Archive(threshold=0.001)
        self._dist_metric = (
            dist_metric if dist_metric in NS._EXPECTED_METRICS else "euclidean"
        )

    @property
    def archive(self):
        return self._archive

    @property
    def k(self):
        return self._k

    def __str__(self):
        return f"NS(k={self._k },A={self._archive})"

    def __repr__(self) -> str:
        return f"NS<k={self._k },A={self._archive}>"

    def __call__(
        self, instances: Sequence[Instance]
    ) -> Tuple[list[Instance], list[float]]:
        """Calculates the sparseness of the given instances against the individuals
        in the Archive.

        Args:
            instances (Sequence[Instance]): Instances to calculate their sparseness
            verbose (bool, optional): Flag to show the progress. Defaults to False.

        Raises:
            ValueError: If len(d) where d is the descriptor of each instance i differs from another
            ValueError: If NoveltySearch.k >= len(instances)

        Returns:
            Tuple(list[Instance], list[float]): Tuple with the instances and the list of sparseness values, one for each instance
        """
        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness on an empty Instance list"
            raise ValueError(msg)

        if self._k >= len(instances):
            msg = f"{self.__class__.__name__} trying to calculate sparseness with k({self._k}) > len(instances)({len(instances)})"
            raise ValueError(msg)

        num_instances = len(instances)
        num_archive = len(self._archive)
        descriptor_length = len(instances[0].descriptor)
        _desc_arr = np.empty((num_instances + num_archive, descriptor_length))
        for i, ind in enumerate(instances):
            _desc_arr[i] = ind.descriptor
        for i, ind in enumerate(self._archive):
            _desc_arr[num_instances + i] = ind.descriptor

        sparseness = np.zeros(num_instances)
        for i in range(num_instances):
            mask = np.ones(num_instances, bool)
            mask[i] = False
            differences = _desc_arr[i] - _desc_arr[np.where(mask)[0]]
            distances = np.linalg.norm(differences, axis=1)
            _neighbors = heapq.nsmallest(self._k, distances)
            s_ = np.sum(_neighbors) / self._k

            instances[i].s = s_
            sparseness[i] = s_
        return instances, sparseness

        # if descriptor_length > 50:
        #     # Apply brute force
        #     return self.__distance_brute_force(_desc_arr, instances, num_instances)
        # elif (num_instances + num_archive) < 500:
        #     # Apply brute force
        #     return self.__distance_brute_force(_desc_arr, instances, num_instances)
        # else:
        # Fit NNs
        # set k+1 because it predicts n[0] == self descriptor
        # neighbourhood = NearestNeighbors(
        #     n_neighbors=self._k + 1,
        #     algorithm="ball_tree",
        #     metric=self._dist_metric,
        # )
        # neighbourhood.fit(_desc_arr)
        # sparseness = (
        #     np.sum(neighbourhood.kneighbors(_desc_arr[: (len(instances))])[0], axis=1)
        #     / self._k
        # )
        # for i in range(num_instances):
        #     instances[i].s = sparseness[i]
        # return (instances, sparseness)


class DominatedNS(NS):
    """
    Dominated Novelty Search (DNS)
    Bahlous-Boldi, R., Faldor, M., Grillotti, L., Janmohamed, H., Coiffard, L., Spector, L., & Cully, A. (2025).
    Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity. 1.
    https://arxiv.org/abs/2502.00593v1
    Quality-Diversity algorithm that implements local competition through dynamic fitness transformations,
    eliminating the need for predefined bounds or parameters. The competition fitness, also known as the dominated novelty score,
    is calculated as the average distance to the k nearest neighbors with higher fitness.
    The value is set in the ``p'' attribute of the Instance class.
    """

    def __init__(self, k: Optional[int] = 15):
        super().__init__(k=k)
        self._archive = None

    def __call__(
        self, instances: Sequence[Instance]
    ) -> Tuple[list[Instance], list[float]]:
        """

        The method returns a descending sorted list of instances by their competition fitness value (p).
        For each instance ``i'' in the sequence, we calculate all the other instances that dominate it.
        Then, we compute the distances between their descriptors using the norm of the difference for each dimension of the descriptors.
        Novel instances will get a competition fitness of np.inf (assuring they will survive).
        Less novel instances will be selected by their competition fitness value. This competition mechanism creates two complementary evolutionary
        pressures: individuals must either improve their fitness or discover distinct behaviors that differ from better-performing
        solutions. Solutions that have no fitter neighbors (D𝑖 = ∅) receive an infinite competition fitness, ensuring their preservation in the
        population.

        Args:
            instances (Sequence[Instance]): Instances to calculate their competition

        Raises:
            ValueError: If len(d) where d is the descriptor of each instance i differs from another
            ValueError: If DNS.k >= len(instances)

        Returns:
            List[Instance]: Numpy array with the instances sorted by their competition fitness value (p). Descending order.
        """
        if self._k >= len(instances):
            raise ValueError(
                f"{__name__} k must be a positive integer and less than the number of instances. Trying to calculate competition with k({self._k}) > len(instances)({len(instances)})"
            )
        if len(instances) == 0 or any(len(d) == 0 for d in instances):
            msg = f"{__name__} trying to calculate competition on an empty list"
            raise ValueError(msg)

        perf_values = np.array([instance.p for instance in instances])
        descriptors = np.array([instance.descriptor for instance in instances])
        N = len(instances)
        for i in range(N):
            # Note: Here it is where we enforce the performance bias
            current_perf = perf_values[i]
            mask = (perf_values > current_perf) & np.ones(N, bool)
            mask[i] = False
            dominated_indices = np.where(mask)[0]
            if len(dominated_indices) == 0:
                instances[i].fitness = np.finfo(np.float32).max
            else:
                differences = descriptors[i] - descriptors[dominated_indices]
                distances = np.linalg.norm(differences, axis=1)
                _neighbors = (
                    heapq.nsmallest(self._k, distances)
                    if len(distances) >= self._k
                    else distances
                )
                instances[i].fitness = np.sum(_neighbors) / self._k

        instances.sort(key=attrgetter("fitness"), reverse=True)
        return (instances, [])
