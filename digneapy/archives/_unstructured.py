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

from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from digneapy.core import Instance

from ._archive import Archive, Keys
from ._utils import check_valid_instance_batch, check_valid_shapes


class UnstructuredArchive(Archive):
    """Unstructured Archive for Novelty Search algorithms.

    The archive stores new instances if their novelty or distance
    is larger than the :attr:novelty_threshold.
    """

    def __init__(
        self,
        k: np.uint32 | int,
        novelty_threshold: float,
        instances: Optional[Sequence[Instance]] = None,
    ):
        """Creates an instance of a UnstructuredArchive for Quality Diversity algorithms

        Args:
            k (np.uint32 | int): Number of neighbours to calculate the diversity of the instances.
            threshold (float): Minimum value of novelty to include an Instance into the archive.
            instances (Optional[Sequence[Instance]], optional): Instances to initialise the archive. Defaults to None.

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

        super().__init__(instances)
        self._k = int(k)
        self._novelty_threshold = float(novelty_threshold)

    @property
    def k(self) -> int:
        """Number of neighbours to calculate the novelty

        Returns:
            int
        """
        return self._k

    @property
    def novelty_threshold(self) -> float:
        """Novelty Threshold

        Returns:
            float: Novelty threshold of the archive
        """
        return self._novelty_threshold

    def __str__(self):
        return f"UnstructuredArchive(k={self._k},threshold={self._novelty_threshold},data=(|{len(self)}|))"

    def __call__(self, descriptors: np.ndarray) -> np.ndarray:
        """Computes the Novelty Search of the instances


        It uses the descriptors of the instances to calculate their novelty
        with respect to the archive and the current batch of instances.
        It uses the Euclidean distance to compute the sparseness.
        If the archive is empty, it returns a NumPy array with the minimum threshold.

        Args:
            descriptors (np.ndarray): Numpy array with the descriptors of the instances

        Returns:
            np.ndarray: novelty scores (s) of the instances descriptors
        """
        return self.compute_novelty(descriptors)

    def compute_novelty(self, descriptors: np.ndarray) -> np.ndarray:
        """Computes the Novelty Search of the instances


        It uses the descriptors of the instances to calculate their novelty
        with respect to the archive and the current batch of instances.
        It uses the Euclidean distance to compute the sparseness.
        If the archive is empty, it returns a NumPy array with the minimum threshold.

        Args:
            descriptors (np.ndarray): Numpy array with the descriptors of the instances

        Returns:
            np.ndarray: novelty scores (s) of the instances descriptors
        """
        num_instances = len(descriptors)
        num_archive = len(self)
        if num_archive == 0:
            return np.full(num_instances, fill_value=self._novelty_threshold)

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
        novelty_scores: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        """Extends the archive with the new batch of instances.

        Args:
            instances (Sequence[Instance]): Instances to insert in the archive.
            novelty_scores (Optional[np.ndarray], optional): Novelty scores of the instances.
                If not given, the novelyt_scores are calculated using compute_novelty. Defaults to None.
            descriptors (Optional[np.ndarray], optional): Descriptors of the instances to calculate their novelty.
                If not given, they are extracted from the Instance objects inside the instances collection. Defaults to None.

        Raises:
            TypeError: If any object inside the instances parameter is not a object of the Instance class.
            ValueError: If there is a mismatch in the shapes (lens) of the instances, novelty_scores and descriptors.
        """
        if check_valid_instance_batch(instances=instances):
            if descriptors is None:
                descriptors = np.asarray([
                    instance.descriptor for instance in instances
                ])
            if novelty_scores is None:
                novelty_scores = self.compute_novelty(descriptors=descriptors)

            if check_valid_shapes(instances, novelty_scores, descriptors):
                novel_enough_mask = novelty_scores >= self._novelty_threshold
                valid_instances = np.where(novel_enough_mask)[0].astype(np.int32)
                for idx in valid_instances:
                    self._storage[Keys.instances].append(instances[idx].clone())
                    self._storage[Keys.descriptors].append(descriptors[idx])
            else:
                raise ValueError(
                    "Shape mismatch between the instances, novelty_scores and descriptors."
                    f"instances have {len(instances)} instances, "
                    f"novelty_scores contains {len(novelty_scores)} and "
                    f"descriptors contains {len(descriptors)}."
                )
        else:
            raise TypeError(
                "All objects inside the instances sequence must be object of the Instance class."
            )
