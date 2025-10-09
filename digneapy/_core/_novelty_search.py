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

from typing import Tuple, Optional

import numpy as np
import sys
from digneapy.archives import Archive


class NS:
    def __init__(
        self,
        archive: Optional[Archive] = None,
        k: int = 15,
    ):
        """Creates an instance of the Novelty Search Algorithm
        Args:
            archive (Archive): Archive to store the instances to guide the evolution. Defaults to Archive(threshold=0.001).
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
        """
        if k < 0:
            raise ValueError(
                f"{__name__} k must be a positive integer and less than the number of instances."
            )

        if archive is not None and not isinstance(archive, Archive):
            raise ValueError("You must provide a valid Archive object")
        self._k = k
        self._archive = archive if archive is not None else Archive(threshold=0.001)

    @property
    def archive(self):
        return self._archive

    @property
    def k(self):
        return self._k

    def __str__(self):
        return f"NS(k={self._k},A={self._archive})"

    def __repr__(self) -> str:
        return f"NS<k={self._k},A={self._archive}>"

    def __call__(self, instances_descriptors: np.ndarray) -> np.ndarray:
        """Computes the Novelty Search of the instance descriptors with respect to the archive.
           It uses the Euclidean distance to compute the sparseness.

        Args:
            instance_descriptors (np.ndarray): Numpy array with the descriptors of the instances
            archive (Archive): Archive which stores the novelty instances found so far
            k (int, optional): Number of neighbors to consider in the computation of the sparseness. Defaults to 15.

        Raises:
            ValueError: If len(instance_descriptors) <= k

        Returns:
            np.ndarray: novelty scores (s) of the instances descriptors
        """
        if len(instances_descriptors) == 0:
            raise ValueError(
                f"NS was given an empty population to compute the sparseness. Shape is: {instances_descriptors.shape}"
            )
        num_instances = len(instances_descriptors)
        num_archive = len(self.archive)
        result = np.zeros(num_instances, dtype=np.float64)
        if num_archive == 0 and num_instances <= self._k:
            # Initially, the archive is empty and we may not have enough instances to evaluate
            print(
                f"NS has an empty archive at this moment and the given population is not large enough to compute the sparseness. {num_instances} < k ({self._k}). Returning zeros.",
                file=sys.stderr,
            )
            return result

        if num_instances + num_archive <= self._k:
            msg = f"Trying to calculate novelty search with k({self._k}) >= {num_instances} (instances) + {num_archive} (archive)."
            raise ValueError(msg)

        combined = (
            instances_descriptors
            if num_archive == 0
            else np.vstack([instances_descriptors, self._archive.descriptors])
        )
        for i in range(num_instances):
            mask = np.ones(num_instances, bool)
            mask[i] = False
            differences = combined[i] - combined[np.nonzero(mask)]
            distances = np.linalg.norm(differences, axis=1)
            _neighbors = np.partition(distances, self._k + 1)[1 : self._k + 1]
            result[i] = np.sum(_neighbors) / self._k

        return result


def dominated_novelty_search(
    descriptors: np.ndarray, performances: np.ndarray, k: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dominated Novelty Search (DNS)
        Bahlous-Boldi, R., Faldor, M., Grillotti, L., Janmohamed, H., Coiffard, L., Spector, L., & Cully, A. (2025).
        Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity. 1.
        https://arxiv.org/abs/2502.00593v1

        Quality-Diversity algorithm that implements local competition through dynamic fitness transformations,
        eliminating the need for predefined bounds or parameters. The competition fitness, also known as the dominated novelty score,
        is calculated as the average distance to the k nearest neighbors with higher fitness.

    The method returns a descending sorted list of instances by their competition fitness value.
    For each instance ``i'' in the sequence, we calculate all the other instances that dominate it.
    Then, we compute the distances between their descriptors using the norm of the difference for each dimension of the descriptors.
    Novel instances will get a competition fitness of np.inf (assuring they will survive).
    Less novel instances will be selected by their competition fitness value. This competition mechanism creates two complementary evolutionary
    pressures: individuals must either improve their fitness or discover distinct behaviors that differ from better-performing
    solutions. Solutions that have no fitter neighbors (D𝑖 = ∅) receive an infinite competition fitness, ensuring their preservation in the
    population.

    Args:
        descriptors (np.ndarray): Numpy array with the descriptors of the instances
        performances (np.ndarray): Numpy array with the performance values of the instances

    Raises:
        ValueError: If len(d) where d is the descriptor of each instance i differs from another
        ValueError: If k >= len(instances)

    Returns:
        Tuple[np.ndarray]: Tuple with the descriptors and competition fitness values sorted, plus the sorted indexing (descending order).
    """
    num_instances = len(descriptors)
    if num_instances <= k:
        msg = f"Trying to calculate the dominated novelty search with k({k}) > len(instances) = {num_instances}"
        raise ValueError(msg)

    if len(performances) != len(descriptors):
        raise ValueError(
            f"Array mismatch between peformances and descriptors. len(performance) = {len(performances)} != {len(descriptors)} len(descriptors)"
        )

    mask = performances[:, None] > performances
    dominated_indices = [np.nonzero(row) for row in mask]
    competition_fitness = np.full(num_instances, np.finfo(np.float32).max)
    for i in range(num_instances):
        if dominated_indices[i][0].size > 0:
            dist = np.linalg.norm(
                descriptors[i] - descriptors[dominated_indices[i]], axis=1
            )
            if len(dist) > k:
                dist = np.partition(dist, k)[:k]
            competition_fitness[i] = np.sum(dist) / k

    indexing = np.argsort(-competition_fitness)
    return (descriptors[indexing], competition_fitness[indexing], indexing)
