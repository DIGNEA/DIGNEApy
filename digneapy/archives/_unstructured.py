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
        threshold: float,
        instances: Optional[Sequence[Instance]] = None,
        local_competition: bool = False,
        iterations_without_improve: Optional[int] = None,
        decay: Optional[float] = None,
        dtype=np.float64,
    ):
        """Creates an instance of a Archive (unstructured) for QD algorithms

        Args:
            k (np.uint32): Number of neighbours for KNN.
            threshold (float): Minimum value of sparseness to include an Instance into the archive.
            instances (Optional[Sequence[Instance]], optional): Instances to initialise the archive. Defaults to None.
            local_competition (bool, optional): Whether to enable Local Competition between instances in the archive. Defaults to False.
            iterations_without_improve (Optional[int], optional): Number of iterations without updating the archive before decreasing the threshold. Defaults to None.
            decay (Optional[float], optional): Threshold decay after X iterations without inserting instances. Defaults to None.
            dtype (_type_, optional): Defaults to np.float64.

        Raises:
            ValueError: K is not valid. K must be a positive integer.
            ValueError: Threshold is not valid. Threshold must be a positive float.
        """
        if not isinstance(k, (int, np.integer, np.unsignedinteger)) or k <= 0:
            raise ValueError(
                f"UnstructuredArchive expects k to be a positive integer. Got {k}"
            )
        if type(threshold) not in (float, int) or threshold < 0.0:
            raise ValueError(
                f"UnstructuredArchive expects a floating point threshold >= 0. Got {threshold}"
            )

        super().__init__(None, dtype)
        self._k = k
        self._threshold = float(threshold)
        self._local_comp = local_competition
        if iterations_without_improve is not None:
            self._max_its_without_imp = int(iterations_without_improve)
            self._it_without_improve = 0
            try:
                self._decay = float(decay)
            except ValueError:
                raise ValueError(
                    "Decay must be a valid float if iterations_without_improve is not None"
                )

        else:
            self._max_its_without_imp = None
        if instances is not None and len(instances) > 0:
            descriptors = [instance.descriptor for instance in instances]
            self._tree = KDTree(np.asarray(descriptors), metric="euclidean")
            self._storage[Keys.instances].extend(instances)
            self._storage[Keys.descriptors].extend(descriptors)

        else:
            self._tree = KDTree(np.zeros((2, 2)), metric="euclidean")

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

        num_instances = len(descriptors)
        num_archive = len(self)
        # We compute the distances between the descriptors in the current batch and those stored inside the archive
        # If the archive is empty (num_archive == 0), then we compute the novelty within the batch only.
        effective_k = min((num_archive + num_instances) - 1, self._k)
        combined = (
            np.vstack([descriptors, self._storage[Keys.descriptors]])
            if num_archive > 0
            else descriptors
        )
        distances = cdist(descriptors, combined)
        # We set the diagonal to INF to avoid select d(i,i) = 0 in partition
        np.fill_diagonal(distances, np.inf)
        knn = np.partition(distances, effective_k - 1, axis=1)[:, :effective_k]
        novelty_scores = np.mean(knn, axis=1)
        return novelty_scores

    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: np.ndarray,
        descriptors: Optional[np.ndarray] = None,
        objectives: Optional[np.ndarray] = None,
    ):
        """Extends the current archive with all the individuals inside iterable that have
        a sparseness value greater than the archive threshold.

        Args:
            instances (Sequence[Instance]): Sequence of instances to be include in the archive.
            novelty_scores (Optional[np.ndarray], optional): Novelty scores of the instances. Defaults to None.
            descriptors (Optional[np.ndarray], optional): Descriptors of the instances. Defaults to None.
            objectives (Optional[np.ndarray], optional): Objectives of the instances, used when local_competition is True. Defaults to None.

        """
        if self._local_comp and objectives is None:
            objectives = np.asarray([instance.p for instance in instances])
        if descriptors is None:
            descriptors = np.asarray([instance.descriptor for instance in instances])

        novel_enough_mask = novelty_scores >= self._threshold
        not_novel_mask = ~novel_enough_mask

        novel_indices = np.where(novel_enough_mask)[0].astype(np.int32)
        # The only invariant I need to maintain is:
        # The KDTree is always built from np.array(self._descriptors), and _descriptors[i] always corresponds to _instances[i].
        # This means two rules:
        # Append to both lists together, atomically — never one without the other.
        # Rebuild the tree immediately after any mutation — which you already do at the end of extend.
        for idx in novel_indices:
            self._storage[Keys.instances].append(instances[idx])
            self._storage[Keys.descriptors].append(descriptors[idx])

        if self._local_comp and np.any(not_novel_mask) and len(self) > 0:
            not_novel_indices = np.where(not_novel_mask)[0].astype(np.int32)
            not_novel_descriptors = descriptors[not_novel_mask]
            not_novel_objectives = objectives[not_novel_mask]

            # Find each candidate's nearest neighbour in the current archive
            archive_objectives = np.array([
                instance.p for instance in self._storage[Keys.instances]
            ])

            _, nn_positions = self._tree.query(not_novel_descriptors, k=1)
            nn_positions = nn_positions[:, 0]  # position in tree == archive list index
            current_objectives = archive_objectives[nn_positions]

            # Keep only candidates that strictly improve their neighbours
            improves = not_novel_objectives > current_objectives  # shape (n_not_novel,)
            if np.any(improves):
                improving_candidates = not_novel_indices[improves]
                improving_descriptors = not_novel_descriptors[improves]
                improving_objectives = not_novel_objectives[improves]
                target_indices = nn_positions[improves]

                # When multiple candidates target the same slot, keep the best one
                # Build a dict: archive_key == best (candidate_idx, objective, descriptor)
                best_per_slot: dict[int, tuple] = {}
                for cand_idx, desc, obj, slot_key in zip(
                    improving_candidates,
                    improving_descriptors,
                    improving_objectives,
                    target_indices,
                    strict=True,
                ):
                    if (
                        slot_key not in best_per_slot
                        or obj > best_per_slot[slot_key][2]
                    ):
                        best_per_slot[slot_key] = (cand_idx, desc, obj)
                for slot_key, (cand_idx, desc, _) in best_per_slot.items():
                    self._storage[Keys.instances][slot_key] = instances[cand_idx]
                    self._storage[Keys.descriptors][slot_key] = desc
        if len(self) > 0:
            # The tree rebuild is what re-establishes the mapping after any mutation
            self._tree = KDTree(
                np.asarray(self._storage[Keys.descriptors]),
                metric="euclidean",
            )
        if self._max_its_without_imp is not None:
            self._it_without_improve = (
                self._it_without_improve + 1 if len(novel_indices) == 0 else 0
            )
            if self._it_without_improve == self._max_its_without_imp:
                self._it_without_improve = 0
                self._threshold = max(0.01, self._threshold * self._decay)
