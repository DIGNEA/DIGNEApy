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

from collections.abc import Iterable, Sequence, Set
from typing import Iterator, Optional, Tuple

import numpy as np

from digneapy.core import Instance

from ._archive import Archive, Keys
from ._utils import check_valid_instance_batch, check_valid_shapes


class GridArchive(Archive):
    """Archive that divides each dimension into uniformly-sized cells.


    The source code of this class is inspired by the GridArchive class of pyribs
    <https://github.com/icaros-usc/pyribs/blob/master/ribs/archives/_grid_archive.py>

    This archive is the container described in `Mouret 2015 <https://arxiv.org/pdf/1504.04909.pdf>`_.
    It can be visualized as a n-dimensional grid in the measure space that is divided into a certain
    number of cells in each dimension. Each cell contains an elite, i.e. a solution that `maximizes`
    the objective function for the measures in that cell.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
        ranges: Sequence[Tuple[float, float]],
        instances: Optional[Sequence[Instance]] = None,
        eps: float = 1e-6,
    ):
        """Creates a GridArchive object

        Args:
            dimensions (Sequence[int]): (Sequence[int]): Number of cells in each dimension of the descriptor
                space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions with 20, 30, and 40 cells.
                (The number of dimensions is implicitly defined in the length of this argument).
            ranges (Sequence[Tuple[float, float]]): Lower and upper bound of each dimension of the descriptorç
                space, e.g. ``[(-1, 1), (-2, 2)]`` indicates the first dimension should have bounds :math:`[-1,1]`
                (inclusive), and the second dimension should have bounds :math:`[-2,2]` (inclusive).
                ``ranges`` should be the same length as ``dims``.
            instances (Optional[Sequence[Instance]], optional): Instances to pre-initialise the archive. Defaults to None.
            eps (float, optional): Due to floating point precision errors, we add a small epsilon when computing
                the archive indices in the :meth:`index_of` method -- refer to the implementation `here. Defaults to 1e-6.

        Raises:
            ValueError: ``dimensions`` and ``ranges`` are not the same length
        """
        super().__init__(None)

        if len(ranges) == 0 or len(dimensions) == 0:
            raise ValueError("dimensions and ranges must have length >= 1")

        if len(ranges) != len(dimensions):
            raise ValueError(
                f"len(dimensions) = {len(dimensions)} != len(ranges) = {len(ranges)} in GridArchive"
            )

        self._dimensions = np.asarray(dimensions)

        ranges = list(zip(*ranges))
        self._lower_bounds = np.asarray(ranges[0], dtype=np.float64, copy=True)
        self._upper_bounds = np.asarray(ranges[1], dtype=np.float64, copy=True)
        del ranges

        self._interval = self._upper_bounds - self._lower_bounds
        self._eps = eps
        self._cells = np.prod(self._dimensions, dtype=object)

        # We override the storage from Archive to build a dictionary of dictionaries and set
        del self._storage
        self._storage = {
            Keys.instances: {},
            Keys.descriptors: {},
            Keys.grid: set(),
        }

        _bounds = []
        for dimension, l_b, u_b in zip(
            self._dimensions, self._lower_bounds, self._upper_bounds
        ):
            _bounds.append(np.linspace(l_b, u_b, dimension))

        self._boundaries = np.asarray(_bounds, copy=True)
        del _bounds

        # Extend the archive with the initial instances
        if instances is not None:
            self.extend(instances)

    @property
    def descriptors(self) -> np.ndarray:
        """Descriptors of the instances

        Returns:
            np.ndarray: Returns a np.ndarray with the descriptors of
                the instances stored in the archive
        """
        return np.asarray(list(self._storage[Keys.descriptors].values()))

    @property
    def dimensions(self) -> np.ndarray:
        """Dimensions of the GridArchive

        Returns:
            np.ndarray: NumPy array with the dimensions of the Grid and
                the number of cell of each dimension.
        """
        return self._dimensions

    @property
    def bounds(self) -> np.ndarray:
        """Boundaries of the cells in each dimension.

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

    def lower_i(self, i: int) -> np.float64:
        """Returns the lower bound of the ith dimension

        Args:
            i (int): Dimension to retrieve

        Raises:
            TypeError: If i is not an int
            ValueError: If i is outside of the bounds

        Returns:
            np.float64: Lower bound of the ith dimension
        """
        if type(i) is not int:
            raise TypeError(f"lower_i expects i to be an integer. Got: {type(i)}")

        if i < 0 or i > len(self._lower_bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self._boundaries)}]"
            raise IndexError(msg)

        return self._lower_bounds[i]

    def upper_i(self, i: int):
        """Returns the upper bound of the ith dimension

        Args:
            i (int): Dimension to retrieve

        Raises:
            TypeError: If i is not an int
            ValueError: If i is outside of the bounds

        Returns:
            np.float64: Upper bound of the ith dimension
        """
        if type(i) is not int:
            raise TypeError(f"upper_i expects i to be an integer. Got: {type(i)}")
        if i < 0 or i > len(self._upper_bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self._boundaries)}]"
            raise IndexError(msg)

        return self._upper_bounds[i]

    @property
    def n_cells(self) -> int:
        """Number of cells of the Grid

        Returns:
            int
        """
        return self._cells

    @property
    def coverage(self) -> np.float64:
        """Get the coverage of the hypercube space.

        The coverage is calculated has the number of cells filled over the total space available.

        Returns:
            np.float64: Filled cells over the total available.
        """
        if len(self._storage[Keys.grid]) == 0:
            return np.float64(0)

        return np.float64(len(self._storage[Keys.grid]) / self._cells)

    @property
    def filled_cells(self) -> Set[int]:
        """Filled cells of the grid

        Returns:
            Set[int]: Set with the indices of the filled cells
        """
        return self._storage[Keys.grid]

    @property
    def instances(self) -> Iterable[Instance]:
        """Instances of the GridArchive

        Returns:
            Iterable[Instance]: Returns a ValueView of the instances
        """
        return self._storage[Keys.instances].values()

    def __iter__(self) -> Iterator[Instance]:
        """Iterator of the GridArchive

        Allows users to iterate the instances of the GridArchive

        Returns:
            Iterator[Instance]
        """
        return iter(self._storage[Keys.instances].values())

    def __str__(self) -> str:
        return f"GridArchive(dim={self._dimensions},cells={self._cells:,},bounds={self._boundaries})"

    def __len__(self) -> int:
        """Length of the GridArchive

        Number of instances stored in the archive

        Returns:
            int: Number of instances stored
        """
        return len(self._storage[Keys.grid])

    def extend(
        self,
        instances: Sequence[Instance],
        descriptors: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> None:
        """Extends the archive with instances

        Args:
            instances (Sequence[Instance]): Collection of instances to insert in the archive
            descriptors (Optional[np.ndarray], optional): Descriptors of the instances.
                If not given, they are extracted from the Instance objects inside the instances collection. Defaults to None.

        Raises:
            TypeError: If instances contains objects which class is not Instance
            ValueError: If there is a mismatch in the shapes (lens) of the instances and descriptors.
        """
        if check_valid_instance_batch(instances):
            if descriptors is None:
                descriptors = np.asarray([
                    instance.descriptor for instance in instances
                ])
            if check_valid_shapes(instances, descriptors):
                indices = self.index_of(descriptors)
                for index, instance, descriptor in zip(
                    indices, instances, descriptors, strict=True
                ):
                    if (
                        index not in self._storage[Keys.grid]
                        or instance.fitness
                        > self._storage[Keys.instances][index].fitness
                    ):
                        self._storage[Keys.grid].add(index)
                        self._storage[Keys.instances][index] = instance.clone()
                        self._storage[Keys.descriptors][index] = descriptor
            else:
                raise ValueError(
                    "Shape mismatch between the instances, novelty_scores and descriptors."
                    f"instances have {len(instances)} instances and "
                    f"descriptors contains {len(descriptors)}."
                )
        else:
            raise TypeError(
                "All objects inside the instances sequence must be object of the Instance class."
            )

    def retrieve(self, descriptors: np.ndarray) -> Sequence[Instance]:
        """Returns a sequence of instances that match the given descriptors.

        Args:
            descriptors (array-like ): Descriptors of the instances that want to retrieve.
            Valid examples are:
            -   archive.retrieve([[0,11], [0,5]) --> Get the instances with the descriptors (0,11) and (0, 5)

        Raises:
            TypeError: If the key is an slice. Not allowed.
            ValueError: If the shape of the keys are not valid.

        Returns:
            Sequence[Instance]: Returns a dict with the found instances.
        """

        descriptors = np.asarray(descriptors)
        if descriptors.ndim != 2 or descriptors.shape[1] != len(self._dimensions):
            raise ValueError(
                f"Expected descriptors to be an array with shape "
                f"(batch_size, dimensions) (i.e. shape "
                f"(batch_size, {len(self._dimensions)})) but it had shape "
                f"{descriptors.shape}"
            )

        else:
            indices = self.index_of(descriptors).tolist()
            instances = [self._storage[Keys.instances][idx] for idx in indices]
            return instances

    def retrieve_filled_cells(self, cells: np.ndarray) -> Sequence[Instance]:
        """Returns instances stored in the requested cells.

        Args:
            cells (array-like ): Cells of the instances that want to retrieve.
            Valid examples are:
            -   archive.retrieve([0,11,5]) --> Get the instances in the cells 0, 11 and 5.

        Raises:
            ValueError: If the shape of the cells is not valid.

        Returns:
            Sequence[Instance]: Returns a collection of instances.
        """

        cells = np.asarray(cells)
        if cells.ndim != 1:
            raise ValueError(
                f"Expected cells to be an 1d-array but it had shape {cells.shape}"
            )
        try:
            instances = [self._storage[Keys.instances][idx] for idx in cells]
        except Exception as exc:
            raise RuntimeError(
                f"requested an invalid cell in method retrieve_filled_cells. {exc}"
            )

        return instances

    def index_of(self, descriptors: np.ndarray) -> np.ndarray:
        """Computes the indices of a batch of descriptors.

        Args:
            descriptors (np.ndarray): (batch_size, dimensions) array of descriptors for each instance

        Raises:
            ValueError: ``descriptors`` is not shape (batch_size, dimensions)

        Returns:
            np.ndarray:  (batch_size, ) array of integer indices representing the flattened grid coordinates.
        """
        if len(descriptors) == 0:
            return np.empty(0)

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

        grid_indices = np.asarray(
            (self._dimensions * (descriptors - self._lower_bounds) + self._eps)
            / self._interval,
            dtype=int,
        )

        # Clip the indexes to make sure they are in the expected range for each dimension
        clipped = np.clip(grid_indices, 0, self._dimensions - 1)
        return self.grid_to_int_index(clipped)

    def grid_to_int_index(self, grid_indices: np.ndarray) -> np.ndarray:
        """Converts indices from the GridArchive to int

        Args:
            grid_indices (np.ndarray): Batch of indices from the GridArchive

        Returns:
            np.ndarray: NumPy array with the indices recasted
        """
        grid_indices = np.asarray(grid_indices)
        if len(self._dimensions) > 64:
            strides = np.cumprod((1,) + tuple(self._dimensions[::-1][:-1]))[::-1]
            # Reshape strides to (1, num_dimensions) to make it broadcastable with grid_indices
            strides = strides.reshape(1, -1)
            flattened_indices = np.sum(grid_indices * strides, axis=1, dtype=object)
        else:
            flattened_indices = np.ravel_multi_index(
                grid_indices.T, self._dimensions
            ).astype(int)
        return flattened_indices

    def int_to_grid_index(self, int_indices: np.ndarray) -> np.ndarray:
        """Calculates the corresponding grid index for a given integer.

        Args:
            int_indices (np.ndarray): Batch of integer indices to convert to grid indices

        Returns:
            np.ndarray: NumPy array with the corresponding grid indices
        """
        int_indices = np.asarray(int_indices)
        if len(self._dimensions) > 64:
            # Manually unravel the index for dimensions > 64
            unravel_indices = []
            remaining_indices = int_indices.astype(object)

            for dim_size in self._dimensions[::-1]:
                unravel_indices.append(remaining_indices % dim_size)
                remaining_indices //= dim_size

            unravel_indices = np.asarray(unravel_indices[::-1]).T
        else:
            unravel_indices = np.asarray(
                np.unravel_index(
                    int_indices,
                    self._dimensions,
                )
            ).T.astype(int)
        return unravel_indices

    def to_dict(self) -> dict:
        """Converts the GridArchive into a dictionary

        Includes dimensions, lbs, ubs, n_cells and other information from Archive

        Returns:
            dict: Dictionary with the instances stored in the archive
        """
        return {
            "dimensions": self._dimensions.tolist(),
            "lbs": self._lower_bounds.tolist(),
            "ubs": self._upper_bounds.tolist(),
            "n_cells": self._cells,
            **super().to_dict(),
        }
