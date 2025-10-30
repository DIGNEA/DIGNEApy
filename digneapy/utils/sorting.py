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

from typing import overload, Sequence, List
from digneapy import Instance
import numpy as np


@overload
def sort_knapsack_instances(instances: np.ndarray) -> np.ndarray: ...
@overload
def sort_knapsack_instances(instances: Sequence[Instance]) -> List[Instance]: ...


def sort_knapsack_instances(
    instances: np.ndarray | Sequence[Instance],
) -> np.ndarray | List[Instance]:
    """Sorts a collection of Knapsack Instances Genotypes based on lexicograph order by (w_i, p_i)

    Args:
        instances (np.ndarray | Sequence[Instance]): Instances to sort

    Raises:
        ValueError: If the dimension of the genotypes (minus Q) is not even. Note that KP instances should contain N pairs of values plus the capacity.

    Returns:
        np.ndarray | Sequence[Instance]: Sorted instances
    """
    genotypes = np.empty(0)
    if isinstance(instances, np.ndarray) and (instances.shape[1] - 1) % 2 != 0:
        raise ValueError(
            f"Something is wrong with these KP instances. Shape 1 should be even and got {instances.shape[1]}"
        )
    elif dimension := (len(instances[0]) - 1) % 2 != 0:
        raise ValueError(
            f"Something is wrong with these KP instances. Shape 1 should be even and got {dimension}"
        )

    genotypes = np.asarray(instances, copy=True)
    M, N = genotypes.shape

    pairs = genotypes[:, 1:].reshape(M, -1, 2)
    order = np.lexsort((pairs[:, :, 1], pairs[:, :, 0]), axis=1)
    sorted_pairs = np.take_along_axis(pairs, order[:, :, None], axis=1)
    genotypes[:, 1:] = sorted_pairs.reshape(M, -1)

    if isinstance(instances, np.ndarray):
        return genotypes
    else:
        return [
            instances[i].clone_with(variables=genotypes[i])
            for i in range(len(instances))
        ]
