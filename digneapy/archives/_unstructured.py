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
from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from .._core import Instance
from .base import Archive, Keys


class UnstructuredArchive(Archive):
    """Archive that stores solutions based on their distances"""

    def __init__(
        self,
        k: np.uint32,
        novelty_threshold: float,
        instances: Optional[Sequence[Instance]] = None,
        dtype=np.float64,
    ):
        """Creates an instance of a Archive (unstructured) for QD algorithms

        Args:
            k (np.uint32): Number of neighbours for KNN.
            threshold (float): Minimum value of sparseness to include an Instance into the archive.
            instances (Optional[Sequence[Instance]], optional): Instances to initialise the archive. Defaults to None.
            dtype (_type_, optional): Defaults to np.float64.

        Raises:
            ValueError: K is not valid. K must be a positive integer.
            ValueError: Threshold is not valid. Threshold must be a positive float.
        """
        if not isinstance(k, (int, np.integer, np.unsignedinteger)) or k <= 0:
            raise ValueError(
                f"UnstructuredArchive expects k to be a positive integer. Got {k}"
            )
        if type(novelty_threshold) not in (float, int) or novelty_threshold < 0.0:
            raise ValueError(
                f"UnstructuredArchive expects a floating point threshold >= 0. Got {novelty_threshold}"
            )

        super().__init__(None, dtype)
        self._k = k
        self._threshold = float(novelty_threshold)
        try:
            self._obj_min_value = float(objective_min_value)
        except ValueError:
            raise ValueError("objective_min_value must be an integer or float value")

        if instances is not None and len(instances) > 0:
            descriptors = [instance.descriptor for instance in instances]
            self._tree = KDTree(np.asarray(descriptors), metric="euclidean")
            self._storage[Keys.instances].extend(instances)
            self._storage[Keys.descriptors].extend(descriptors)

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
                novelty_threshold=self._threshold,
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
        num_instances = len(descriptors)
        num_archive = len(self)
        if num_archive == 0:
            return np.full(num_instances, fill_value=self._threshold)

        else:
            effective_k = min((num_archive + num_instances) - 1, self._k)
            distances = cdist(
                descriptors, np.vstack([descriptors, self._storage[Keys.descriptors]])
            )
            np.fill_diagonal(distances, np.inf)
            knn = np.partition(distances, effective_k, axis=1)[:, : effective_k + 1]
            novelty = np.mean(knn, axis=1)
            return novelty

    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: np.ndarray,
        descriptors: Optional[np.ndarray] = None,
    ):
        """Extends the current archive with all the individuals inside iterable that have
        a sparseness value greater than the archive threshold.

        Args:
            instances (Sequence[Instance]): Sequence of instances to be include in the archive.
            novelty_scores (Optional[np.ndarray], optional): Novelty scores of the instances. Defaults to None.
            descriptors (Optional[np.ndarray], optional): Descriptors of the instances. Defaults to None.

        """
        if descriptors is None:
            descriptors = np.asarray([instance.descriptor for instance in instances])
        
        novel_enough_mask = novelty_scores >= self._threshold
        valid_instances = np.where(novel_enough_mask)[0].astype(np.int32)
        for idx in valid_instances:
            self._storage[Keys.instances].append(instances[idx])
            self._storage[Keys.descriptors].append(descriptors[idx])
