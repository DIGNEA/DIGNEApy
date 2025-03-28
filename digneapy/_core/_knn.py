#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _knn.py
@Time    :   2025/03/28 11:43:45
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from collections.abc import Sequence
from digneapy._core import Instance


def sparseness_only_values(
    instances: Sequence[Instance], archive: Sequence[Instance], k: int = 15
) -> np.ndarray:
    _instance_desc = np.array([instance.descriptor for instance in instances])
    _archive_desc = np.array([instance.descriptor for instance in archive])
    combined = (
        _instance_desc
        if len(archive) == 0
        else np.vstack(
            [
                _instance_desc,
                _archive_desc,
            ]
        )
    )

    dist = (
        (_instance_desc**2).sum(-1)[:, None]
        + (combined**2).sum(-1)[None, :]
        - 2 * _instance_desc @ combined.T
    )
    dist = np.nan_to_num(dist, nan=np.inf)
    # clipping necessary - numerical approx make some distancies negative
    dist = np.sqrt(np.clip(dist, min=0.0))
    _neighbors = np.partition(dist, k + 1, axis=1)[:, 1 : k + 1]
    s_ = np.sum(_neighbors, axis=1) / k
    return s_


def sparseness(
    instances: Sequence[Instance], archive: Sequence[Instance], k: int = 15
) -> Sequence[Instance]:
    num_instances = len(instances)
    _instance_desc = np.array([instance.descriptor for instance in instances])
    _archive_desc = np.array([instance.descriptor for instance in archive])
    combined = (
        _instance_desc
        if len(archive) == 0
        else np.vstack(
            [
                _instance_desc,
                _archive_desc,
            ]
        )
    )

    result = np.zeros(num_instances)
    for i in range(num_instances):
        mask = np.ones(num_instances, bool)
        mask[i] = False
        differences = combined[i] - combined[np.where(mask)[0]]
        distances = np.linalg.norm(differences, axis=1)
        _neighbors = np.partition(distances, k + 1)[1 : k + 1]
        result[i] = np.sum(_neighbors) / k
        instances[i].s = result[i]
    return result
