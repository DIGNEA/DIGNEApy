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

import copy
import json
from collections.abc import Iterable, Sequence
from typing import Dict, Optional, Tuple

import numpy as np

from digneapy.core import Instance

from ._base_archive import Archive


class GridArchive(Archive):
    """An archive that divides each dimension into a uniformly-sized cells.
    The source code of this class is inspired by the GridArchive class of pyribs <https://github.com/icaros-usc/pyribs/blob/master/ribs/archives/_grid_archive.py>
    This archive is the container described in `Mouret 2015
    <https://arxiv.org/pdf/1504.04909.pdf>`_. It can be visualized as an
    n-dimensional grid in the measure space that is divided into a certain
    number of cells in each dimension. Each cell contains an elite, i.e. a
    solution that `maximizes` the objective function for the measures in that
    cell.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
        ranges: Sequence[Tuple[float, float]],
        instances: Optional[Iterable[Instance]] = None,
        eps: float = 1e-6,
        dtype=np.float64,
    ):
        """Creates a GridArchive instance

        Args:
            dimensions (Sequence[int]): (array-like of int): Number of cells in each dimension of the
            measure space, e.g. ``[20, 30, 40]`` indicates there should be 3
            dimensions with 20, 30, and 40 cells. (The number of dimensions is
            implicitly defined in the length of this argument).
            ranges (Sequence[Tuple[float]]): (array-like of (float, float)): Upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
            instances (Optional[Iterable[Instance]], optional): Instances to pre-initialise the archive. Defaults to None.
            eps (float, optional): Due to floating point precision errors, we add a small
            epsilon when computing the archive indices in the :meth:`index_of`
            method -- refer to the implementation `here. Defaults to 1e-6.
            dtype(str or data-type): Data type of the solutions, objectives,
            and measures.

        Raises:
            ValueError: ``dimensions`` and ``ranges`` are not the same length
        """
        Archive.__init__(self, threshold=np.inf)
        if len(ranges) == 0 or len(dimensions) == 0:
            raise ValueError("dimensions or ranges must have length >= 1")
        if len(ranges) != len(dimensions):
            raise ValueError(
                f"len(dimensions) = {len(dimensions)} != len(ranges) = {len(ranges)} in GridArchive.__init__()"
            )

        self._dimensions = np.asarray(dimensions)
        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=dtype)
        self._upper_bounds = np.array(ranges[1], dtype=dtype)
        self._interval = self._upper_bounds - self._lower_bounds
        self._eps = eps
        self._cells = np.prod(self._dimensions, dtype=int)
        self._grid: Dict[int, Instance] = {}

        _bounds = []
        for dimension, l_b, u_b in zip(
            self._dimensions, self._lower_bounds, self._upper_bounds
        ):
            _bounds.append(np.linspace(l_b, u_b, dimension))

        self._boundaries = np.asarray(_bounds)

        if instances is not None:
            self.extend(instances)

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def bounds(self):
        """list of numpy.ndarray: The boundaries of the cells in each dimension.

        Entry ``i`` in this list is an array that contains the boundaries of the
        cells in dimension ``i``. The array contains ``self.dims[i] + 1``
        entries laid out like this::

            Archive cells:  | 0 | 1 |   ...   |    self.dims[i]    |
            boundaries[i]:    0   1   2            self.dims[i] - 1  self.dims[i]

        Thus, ``boundaries[i][j]`` and ``boundaries[i][j + 1]`` are the lower
        and upper bounds of cell ``j`` in dimension ``i``. To access the lower
        bounds of all the cells in dimension ``i``, use ``boundaries[i][:-1]``,
        and to access all the upper bounds, use ``boundaries[i][1:]``.
        """
        return self._boundaries

    @property
    def n_cells(self):
        return self._cells

    @property
    def coverage(self):
        """Get the coverage of the hypercube space.
        The coverage is calculated has the number of cells filled over the total space available.

        Returns:
            float: Filled cells over the total available.
        """
        if len(self._grid) == 0:
            return 0.0

        return len(self._grid) / self._cells

    @property
    def filled_cells(self):
        return self._grid.keys()

    @property
    def instances(self):
        return list(self._grid.values())

    def __str__(self):
        return f"GridArchive(dim={self._dimensions},cells={self._cells},bounds={self._boundaries})"

    def __repr__(self):
        return f"GridArchive(dim={self._dimensions},cells={self._cells},bounds={self._boundaries})"

    def __len__(self):
        return len(self._grid)

    def __getitem__(self, key):
        """Returns a dictionary with the descriptors as the keys. The values are the instances found.
        Note that some of the given keys may not be in the archive.

        Args:
            key (array-like or descriptor): Descriptors of the instances that want to retrieve.
            Valid examples are:
            -   archive[[0,11], [0,5]] --> Get the instances with the descriptors (0,11) and (0, 5)
            -   archive[0,11] --> Get the instance with the descriptor (0,11)

        Raises:
            TypeError: If the key is an slice. Not allowed.
            ValueError: If the shape of the keys are not valid.

        Returns:
            dict: Returns a dict with the found instances.
        """
        if isinstance(key, slice):
            raise TypeError(
                "Slicing is not available in GridArchive. Use 1D index or descriptor-type indeces"
            )
        descriptors = np.asarray(key)
        if descriptors.ndim == 1 and descriptors.shape[0] != len(self._dimensions):
            raise ValueError(
                f"Expected descriptors to be an array with shape "
                f"(batch_size, 1) or (batch_size, dimensions) (i.e. shape "
                f"(batch_size, {len(self._dimensions)})) but it had shape "
                f"{descriptors.shape}"
            )

        indeces = self.index_of(descriptors).tolist()
        if isinstance(indeces, int):
            indeces = [indeces]
            descriptors = [descriptors]

        instances = {}
        for idx, desc in zip(indeces, descriptors):
            if idx not in self._grid:
                print(f"There is not any instance in the cell {desc}.")
            else:
                instances[tuple(desc)] = copy.copy(self._grid[idx])
        return instances

    def __iter__(self):
        """Iterates over the dictionary of instances

        Returns:
            Iterator: Yields position in the hypercube and instance located in such position
        """
        return iter(self._grid.values())

    def lower_i(self, i):
        if i < 0 or i > len(self._lower_bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self._boundaries)}]"
            raise ValueError(msg)
        return self._lower_bounds[i]

    def upper_i(self, i):
        if i < 0 or i > len(self._upper_bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self._boundaries)}]"
            raise ValueError(msg)
        return self._upper_bounds[i]

    def append(self, instance: Instance):
        """Inserts an Instance into the Grid

        Args:
            instance (Instance): Instace to be inserted

        Raises:
            TypeError: ``instance`` is not a instance of the class Instance.
        """
        if isinstance(instance, Instance):
            index = self.index_of(np.asarray(instance.descriptor))
            if index not in self._grid or instance > self._grid[index]:
                self._grid[index] = copy.deepcopy(instance)

        else:
            msg = "Only objects of type Instance can be inserted into a GridArchive"
            raise TypeError(msg)

    def extend(self, iterable: Iterable[Instance], *args, **kwargs):
        """Includes all the instances in iterable into the Grid

        Args:
            iterable (Iterable[Instance]): Iterable of instances
        """
        if not all(isinstance(i, Instance) for i in iterable):
            msg = "Only objects of type Instance can be inserted into a GridArchive"
            raise TypeError(msg)

        indeces = self.index_of([i.descriptor for i in iterable])
        for idx, instance in zip(indeces, iterable, strict=True):
            if idx not in self._grid or instance.fitness > self._grid[idx].fitness:
                self._grid[idx] = copy.deepcopy(instance)

    def index_of(self, descriptors):
        """Computes the indeces of a batch of descriptors.

        Args:
            descriptors (array-like): (batch_size, dimensions) array of descriptors for each instance

        Raises:
            ValueError: ``descriptors`` is not shape (batch_size, dimensions)

        Returns:
            np.ndarray:  (batch_size, ) array of integer indices representing the flattened grid coordinates.
        """
        descriptors = np.asarray(descriptors)
        if (
            descriptors.ndim == 1
            and descriptors.shape[0] != len(self._dimensions)
            or descriptors.ndim == 2
            and descriptors.shape[1] != len(self._dimensions)
        ):
            raise ValueError(
                f"Expected descriptors to be an array with shape "
                f"(batch_size, dimensions) (i.e. shape "
                f"(batch_size, {len(self._dimensions)})) but it had shape "
                f"{descriptors.shape}"
            )

        grid_indices = (
            (self._dimensions * (descriptors - self._lower_bounds) + self._eps)
            / self._interval
        ).astype(int)

        # Clip the indexes to make sure they are in the expected range for each dimension
        clipped = np.clip(grid_indices, 0, self._dimensions - 1)
        return self._grid_to_int_index(clipped)

    def _grid_to_int_index(self, grid_indices) -> np.ndarray:
        grid_indices = np.asarray(grid_indices)
        return np.ravel_multi_index(grid_indices.T, self._dimensions).astype(int)

    def int_to_grid_index(self, int_indices) -> np.ndarray:
        int_indices = np.asarray(int_indices)
        return np.asarray(
            np.unravel_index(
                int_indices,
                self._dimensions,
            )
        ).T.astype(int)

    def to_json(self):
        data = {
            "dimensions": self._dimensions.tolist(),
            "lbs": self._lower_bounds.tolist(),
            "ubs": self._upper_bounds.tolist(),
            "n_cells": self._cells.astype(int),
        }
        return json.dumps(data, indent=4)
