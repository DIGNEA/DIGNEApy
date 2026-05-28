#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   proximity.py
@Time    :   2026/05/22 10:03:25
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import operator
import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from .._core import Instance
from .base import Archive, Keys


class UnstructuredArchive(Archive):
    """Archive that stores solutions based on their distances"""

    def __init__(
        self,
        k: int,
        threshold: float,
        instances: Optional[Sequence[Instance]] = None,
        dtype=np.float64,
    ):
        """Creates an instance of a Archive (unstructured) for QD algorithms

        Args:
            threshold (float): Minimum value of sparseness to include an Instance into the archive.
            instances (Iterable[Instance], optional): Instances to initialise the archive. Defaults to None.
        """
        if type(k) is not int or k < 0:
            raise ValueError(
                f"UnstructuredArchive expects k to be a positive integer. Got {k}"
            )
        if type(threshold) not in (float, int) or threshold < 0.0:
            raise ValueError(
                f"UnstructuredArchive expects a floating point threshold >= 0. Got {threshold}"
            )
        super().__init__(instances, dtype)
        self._k = k
        self._threshold = float(threshold)

    @property
    def k(self):
        return self._k

    @property
    def threshold(self):
        return self._threshold

    def __getitem__(self, key):
        if isinstance(key, slice):
            return UnstructuredArchive(
                k=self._k,
                threshold=self._threshold,
                instances=self._storage[Keys.instances][key],
            )
        index = operator.index(key)
        return self._storage[Keys.instances][index]

    def __str__(self):
        return f"UnstructuredArchive(threshold={self._threshold},data=(|{len(self)}|))"

    def __repr__(self):
        return f"UnstructuredArchive(threshold={self._threshold},data=(|{len(self)}|))"

    def __call__(self, descriptors: np.ndarray) -> np.ndarray:
        return self.compute_novelty(descriptors)

    def compute_novelty(self, descriptors: np.ndarray) -> np.ndarray:
        """Computes the Novelty Search of the instance descriptors with respect to the archive.
           It uses the Euclidean distance to compute the sparseness.

        Args:
            descriptors (np.ndarray): Numpy array with the descriptors of the instances

        Raises:
            ValueError: If len(instance_descriptors) == 0 or in combination with archive is <= k

        Returns:
            np.ndarray: novelty scores (s) of the instances descriptors
        """
        if len(descriptors) == 0:
            raise ValueError(
                f"UnstructuredArchive was given an empty population to compute the sparseness. Shape is: {descriptors.shape}"
            )
        num_instances = len(descriptors)
        num_archive = len(self._storage[Keys.descriptors])
        novelty_scores = np.zeros(num_instances, dtype=np.float64)
        if (num_archive + num_instances) <= self._k:
            # The archive may not have enough instances to evaluate
            warnings.warn(
                f"Not enough neighbors to compute sparseness for k={self._k}. "
                f"(archive={num_archive}, instances={num_instances}). "
                f"Returning zeros.",
                RuntimeWarning,
                stacklevel=3,
            )
            return novelty_scores

        if num_archive == 0:
            combined = descriptors
        else:
            combined = np.vstack([descriptors, self._storage[Keys.descriptors]])

        distances = cdist(descriptors, combined)
        # We set the diagonal to INF to avoid select d(i,i) = 0 in partition
        np.fill_diagonal(distances, np.inf)
        knn = np.partition(distances, self._k - 1, axis=1)[:, : self._k]
        novelty_scores = np.mean(knn, axis=1)
        return novelty_scores

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
            novelty_scores (Optional[np.ndarray], optional): Novelty scores of the instances. Defaults to None.
            descriptors (Optional[np.ndarray], optional): Descriptors of the instances. Defaults to None.
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
        self._storage[Keys.instances].extend((instances[i] for i in to_insert))
        self._storage[Keys.descriptors].extend(descriptors[to_insert])
