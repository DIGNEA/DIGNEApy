#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _grid_archive.py
@Time    :   2024/06/07 12:18:10
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from ._base_archive import Archive
from digneapy.core import Instance
import numpy as np
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import Optional, Tuple, Callable


class GridArchive(Archive):
    def __init__(
        self,
        dimensions: Sequence[int],
        ranges: Sequence[Tuple[float]],
        instances: Optional[Iterable[Instance]] = None,
    ):
        Archive.__init__(self, threshold=np.inf)
        if len(ranges) != len(dimensions):
            raise AttributeError(
                "len(dimensions) != len(ranges) in GridArchive.__init__()"
            )

        self._dimensions = tuple(dimensions)
        self._bounds = ranges
        self._bins = np.prod(self._dimensions, dtype=np.int32)
        self._grid = np.empty(self._bins, dtype=np.float32)
        self._data = {}

    def __str__(self):
        return f"GridArchive(dim={self._dimensions},bins={self._bins},bounds=[{self._bounds},{self._bounds}])"

    def __repr__(self):
        return f"GridArchive(dim={self._dimensions},bins={self._bins},bounds=[{self._bounds},{self._bounds}])"

    def __len__(self):
        return len(self._data)

    def lower_i(self, i):
        if i < 0 or i > len(self._bounds):
            msg = (
                f"index {i} is out of bounds. Valid values are [0-{len(self._bounds)}]"
            )
            raise AttributeError(msg)
        return self._bounds[i][0]

    def upper_i(self, i):
        if i < 0 or i > len(self._bounds):
            msg = (
                f"index {i} is out of bounds. Valid values are [0-{len(self._bounds)}]"
            )
            raise AttributeError(msg)
        return self._bounds[i][1]

    def append(self, i: Instance):
        if isinstance(i, Instance):
            # Todo: Include descriptor field in the instances
            index = self.grid_to_int_index(np.array(i.features))
            if self._grid[index] < i.fitness:
                self._grid[index] = i.fitness
                self._data[index] = i
        else:
            msg = f"Only objects of type {Instance.__class__.__name__} can be inserted into a GridArchive"
            raise AttributeError(msg)

    def extend(self, iterable: Iterable[Instance], *args, **kwargs):
        indeces = self.grid_to_int_index(np.array(list(i.features for i in iterable)))
        for idx, instance in zip(indeces, iterable):
            if self._grid[idx] < instance.fitness:
                self._grid[idx] = instance.fitness
                self._data[idx] = instance

    def grid_to_int_index(self, grid_indices) -> np.ndarray:
        grid_indices = np.asarray(grid_indices)
        return np.ravel_multi_index(grid_indices.T, self._dimensions).astype(np.int32)

    def int_to_grid_index(self, int_indices) -> np.ndarray:
        int_indices = np.asarray(int_indices)
        return np.asarray(
            np.unravel_index(
                int_indices,
                self._dimensions,
            )
        ).T.astype(np.int32)
