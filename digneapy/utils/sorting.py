#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sorting.py
@Time    :   2025/10/17 16:34:46
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import numpy as np


def sort_knapsack_instances(instances: np.ndarray) -> np.ndarray:
    """Sorts a collection of Knapsack Instances Genotypes based on lexicograph order by (w_i, p_i)

    Args:
        instances (np.ndarray): Instances to sort

    Raises:
        ValueError: If the dimension of the genotypes (minus Q) is not even. Note that KP instances should contain N pairs of values plus the capacity.

    Returns:
        np.ndarray: Sorted instances
    """
    if (instances.shape[1] - 1) % 2 != 0:
        raise ValueError(
            f"Something is wrong with these KP instances. Shape 1 should be even and got {instances.shape[1]}"
        )

    M, N = instances.shape
    pairs = instances[:, 1:].reshape(M, -1, 2)
    order = np.lexsort((pairs[:, :, 1], pairs[:, :, 0]), axis=1)
    sorted_pairs = np.take_along_axis(pairs, order[:, :, None], axis=1)
    instances[:, 1:] = sorted_pairs.reshape(M, -1)
    return instances
