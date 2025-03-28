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
import itertools
import numpy as np

from digneapy._core._instance import Instance
from digneapy.archives import Archive
from digneapy._core._knn import sparseness


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
        num_instances = len(instances)
        if num_instances <= self._k:
            msg = f"{self.__class__.__name__} trying to calculate sparseness with k({self._k}) > len(instances)({num_instances})"
            raise ValueError(msg)

        results = sparseness(instances, self._archive, k=self._k)
        return instances, results


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
        num_instances = len(instances)
        if num_instances <= self._k:
            msg = f"{self.__class__.__name__} trying to calculate sparseness with k({self._k}) > len(instances)({num_instances})"
            raise ValueError(msg)

        perf_values = np.array([instance.p for instance in instances])
        descriptors = np.array([instance.descriptor for instance in instances])
        for i in range(num_instances):
            # Note: Here it is where we enforce the performance bias
            current_perf = perf_values[i]
            mask = (perf_values > current_perf) & np.ones(num_instances, bool)
            mask[i] = False
            dominated_indices = np.where(mask)[0]
            if len(dominated_indices) == 0:
                instances[i].fitness = np.finfo(np.float32).max
            else:
                dist = np.linalg.norm(
                    descriptors[i] - descriptors[dominated_indices], axis=1
                )
                if len(dist) > self._k:
                    dist = np.partition(dist, self._k)[self._k]

                instances[i].fitness = np.sum(dist) / self._k

        instances.sort(key=attrgetter("fitness"), reverse=True)
        return (instances, [])
