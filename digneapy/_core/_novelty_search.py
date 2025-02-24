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

import itertools
from collections.abc import Sequence
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from digneapy.archives import Archive

from ._instance import Instance


class NS:
    """Descriptor strategies for the Novelty Search algorithm.
    The current version supports Features, Performance and Instance variations.
    """

    _EXPECTED_METRICS = ("euclidean", "cosine", "manhattan")

    def __init__(
        self,
        archive: Optional[Archive] = None,
        k: int = 15,
        dist_metric: Optional[str] = "minkowski",
    ):
        """Creates an instance of the NoveltySearch Algorithm
        Args:
            archive (Archive, optional): Archive to store the instances to guide the evolution. Defaults to Archive(threshold=0.001).
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            transformer (callable, optional): Define a strategy to transform the high-dimensional descriptors to low-dimensional.Defaults to None.
            dist_metric (str, optional): Defines the distance metric used by NearestNeighbor in the archives. Defaults to Euclidean.
        """
        if k < 0:
            raise ValueError(
                f"{__name__} k must be a positive integer and less than the number of instances."
            )
        self._k = k + 1
        self._archive = archive if archive is not None else Archive(threshold=0.001)
        # Set k+1 to nbr because the first neighbour is the instance itself
        self._dist_metric = (
            dist_metric if dist_metric in NS._EXPECTED_METRICS else "minkowski"
        )
        self._nbr_algorithm = "auto" if self._dist_metric == "cosine" else "ball_tree"

    @property
    def archive(self):
        return self._archive

    @property
    def k(self):
        return self._k - 1

    def __str__(self):
        return f"NS(k={self._k - 1},A={self._archive})"

    def __repr__(self) -> str:
        return f"NS<k={self._k - 1},A={self._archive}>"

    def _combined_archive_and_population(
        self, current_pop: Archive, instances: Sequence[Instance]
    ) -> np.ndarray:
        """Combines the archive and the given population before computing the sparseness

        Args:
            current_pop (Archive): Current archive of solution.
            instances (Sequence[Instance]): Sequence of instances to evaluate.

        Returns:
            np.ndarray: Returns an ndarray of descriptors.
        """
        return np.vstack(
            [ind.descriptor for ind in itertools.chain(instances, current_pop)]
        )

    def __compute_sparseness(
        self,
        instances: Sequence[Instance],
        current_archive: Archive,
        neighbours: int,
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
        neighbourhood = NearestNeighbors(
            n_neighbors=neighbours,
            algorithm=self._nbr_algorithm,
            metric=self._dist_metric,
        )
        neighbourhood.fit(_desc_arr)
        sparseness = []
        # We're only interesed in the new instances
        frac = 1.0 / neighbours
        for instance, descriptor in zip(instances, _desc_arr[: len(instances)]):
            dist, ind = neighbourhood.kneighbors([descriptor])
            dist, ind = dist[0][1:], ind[0][1:]
            s = frac * sum(dist)
            instance.s = s
            instance.descriptor = descriptor
            sparseness.append(s)

        return sparseness

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

        sparseness = self.__compute_sparseness(
            instances, self.archive, neighbours=self._k
        )
        return (instances, sparseness)


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

    def __init__(self, k: int = 15):
        super().__init__(k=k)
        self._archive = None

    def __call__(
        self, instances: Sequence[Instance]
    ) -> Tuple[list[Instance], list[float]]:
        """

        The method returns a descending sorted list of instances by their competition fitness value (p).
        For each instance ``i'' in the sequence, we calculate all the other instances that dominate it.
        Then, we compute the distances between their descriptors using the norm of the difference for each dimension of the descriptors.
        Novel instances will get a competition fitness of np.inf (assuring they will survive). Less novel instances will be selected by their competition fitness value.

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
            msg = (
                f"{__name__} trying to calculate competition on an empty Instance list"
            )
            raise ValueError(msg)
        for i, individual in enumerate(instances):
            dominate_i = filter(
                lambda j: instances[j].p > individual.p and i != j,
                range(len(instances)),
            )

            ind_descriptor = np.array(individual.descriptor)
            distances = sorted(
                [
                    np.linalg.norm(
                        np.array(ind_descriptor) - np.array(instances[j].descriptor)
                    )
                    for j in dominate_i
                ]
            )
            ld = len(distances)

            if ld > 0:
                _neighbors = distances[: self._k] if ld >= self._k else distances
                individual.fitness = 1.0 / self._k * sum(_neighbors)
            else:
                individual.fitness = np.inf

        instances = list(sorted(instances, key=lambda x: x.fitness, reverse=True))
        return (instances, [])
