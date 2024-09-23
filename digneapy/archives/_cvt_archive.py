#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _cvt_archive.py
@Time    :   2024/09/18 14:44:44
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
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

from digneapy._core import Instance

from ._base_archive import Archive


class CVTArchive(Archive):
    """An Archive that divides a high-dimensional measure space into k homogeneous geometric regions.
    Based on the paper from Vassiliades et al (2018) <https://ieeexplore.ieee.org/document/8000667>
    > The computational complexity of the method we provide for constructing the CVT (in Algorithm 1) is O(ndki),
    > where n is the number of d-dimensional samples to be clustered, k is the number of clusters,
    > and i is the number of iterations needed until convergence
    """

    def __init__(
        self,
        k: int,
        ranges: Sequence[Tuple[float, float]],
        n_samples: int,
        centroids: Optional[npt.NDArray | str] = None,
        samples: Optional[npt.NDArray | str] = None,
        dtype=np.float64,
    ):
        """Creates a CVTArchive object

        Args:
            k (int): Number of centroids (regions) to create
            ranges (Sequence[Tuple[float, float]]): Ranges of the measure space. Upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). The legnth of ``ranges`` indicates the number of dimensions of the measure space.
            n_samples (int): Number of samples to generate before calculating the centroids.
            centroids (Optional[npt.NDArray  |  str], optional): Precalculated centroids for the archive.
            The options are a np.ndarray with the values of ``k`` centroids or a .txt with the centroids to be loaded by Numpy. Defaults to None.
            samples (Optional[npt.NDArray  |  str], optional): Precalculated samples for the archive.
            The options are a np.ndarray with the values of ``n_samples`` samples or a .txt with the samples to be loaded by Numpy. Defaults to None.

        Raises:
            ValueError: If len(ranges) <= 0.
            ValueError: If the number of samples is less than zero or less than the number of regions (k).
            ValueError: If the number of regions is less than zero.
            ValueError: If the samples file cannot be loaded.
            ValueError: If given a samples np.ndarray the number of samples in the file is different from the number of expected samples (n_samples).
            ValueError: If the centroids file cannot be loaded.
            ValueError: If given a centroids np.ndarray the number of centroids in the file is different from the number of regions (k).
        """
        Archive.__init__(self, threshold=np.inf, dtype=dtype)
        if k <= 0:
            raise ValueError(f"The number of regions (k = {k}) must be >= 1")

        if len(ranges) <= 0:
            raise ValueError(
                f"ranges must have length >= 1 and it has length {len(ranges)}"
            )

        if n_samples <= 0 or n_samples < k:
            raise ValueError(
                f"The number of samples (n_samples = {n_samples}) must be >= 1 and >= regions (k = {k})"
            )

        self._dimensions = len(ranges)
        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self._dtype)
        self._upper_bounds = np.array(ranges[1], dtype=self._dtype)
        self._interval = self._upper_bounds - self._lower_bounds
        self._k = k
        self._n_samples = n_samples
        self._samples = None
        self._centroids = None
        self._kmeans = KMeans(n_clusters=self._k, n_init=1)

        # Data Structure to store the instances in the CVT
        self._grid: Dict[int, Instance] = {}
        # Loading samples if given
        if samples is not None:
            if isinstance(samples, str):
                try:
                    self._samples = np.load(samples)
                    self._n_samples = len(self._samples)
                except Exception as _:
                    raise ValueError(
                        f"Error in CVTArchive.__init__() loading the samples file {samples}."
                    )
            elif isinstance(samples, np.ndarray) and len(samples) != n_samples:
                raise ValueError(
                    f"The number of samples {len(samples)} must be equal to the number of expected samples (n_samples = {n_samples})"
                )
            else:
                self._samples = np.asarray(samples)

        if centroids is not None:
            if isinstance(centroids, str):
                try:
                    self._centroids = np.load(centroids)
                    self._k = len(self._centroids)
                except Exception as _:
                    raise ValueError(
                        f"Error in CVTArchive.__init__() loading the centroids file {centroids}."
                    )
            elif isinstance(centroids, np.ndarray) and len(centroids) != k:
                raise ValueError(
                    f"The number of centroids {len(centroids)} must be equal to the number of regions (k = {self._k})"
                )
            else:
                self._centroids = np.asarray(centroids)
        else:
            # Generate centroids
            if self._samples is None:
                # Generate uniform samples if not given
                self._samples = np.random.uniform(
                    low=self._lower_bounds,
                    high=self._upper_bounds,
                    size=(self._n_samples, self._dimensions),
                )
            self._kmeans.fit(self._samples)
            self._centroids = self._kmeans.cluster_centers_

        self._kdtree = KDTree(self._centroids, metric="euclidean")

    @property
    def dimensions(self) -> int:
        """Dimensions of the measure space used

        Returns:
            int: Dimensions of the measure space used
        """
        return self._dimensions

    @property
    def samples(self) -> np.ndarray:
        """Returns the samples used to generate the centroids

        Returns:
            np.ndarray: Samples
        """
        return self._samples

    @property
    def centroids(self) -> np.ndarray:
        """Returns k centroids calculated from the samples

        Returns:
            np.ndarray: K d-dimensional centroids
        """
        return self._centroids

    @property
    def regions(self) -> int:
        """Number of regions (k) of centroids in the CVTArchive

        Returns:
            int: k
        """
        return self._k

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tuple with the lower and upper bounds of the measure space
        The first value is the lower bounds and the second value is the upper bounds.
        Each value is a list with the corresponding lower/upper bound of the ith dimension
        in the measure space
        """
        return (self._lower_bounds, self._upper_bounds)

    @property
    def instances(self) -> list[Instance]:
        return list(self._grid.values())

    def __str__(self):
        return f"CVArchive(dim={self._dimensions},regions={self._k},centroids={self._centroids})"

    def __repr__(self):
        return f"CVArchive(dim={self._dimensions},regions={self._k},centroids={self._centroids})"

    def __iter__(self):
        """Iterates over the dictionary of instances

        Returns:
            Iterator: Yields position in the hypercube and instance located in such position
        """
        return iter(self._grid.values())

    def lower_i(self, i) -> np.float64:
        if i < 0 or i > len(self._lower_bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self._lower_bounds)}]"
            raise ValueError(msg)
        return self._lower_bounds[i]

    def upper_i(self, i) -> np.float64:
        if i < 0 or i > len(self._upper_bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self._upper_bounds)}]"
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
            msg = "Only objects of type Instance can be inserted into a CVTArchive"
            raise TypeError(msg)

    def extend(self, iterable: Iterable[Instance]):
        """Includes all the instances in iterable into the Grid

        Args:
            iterable (Iterable[Instance]): Iterable of instances
        """
        if not all(isinstance(i, Instance) for i in iterable):
            msg = "Only objects of type Instance can be inserted into a CVTArchive"
            raise TypeError(msg)

        indeces = self.index_of([i.descriptor for i in iterable])
        for idx, instance in zip(indeces, iterable, strict=True):
            if idx not in self._grid or instance.fitness > self._grid[idx].fitness:
                self._grid[idx] = copy.deepcopy(instance)

    def remove(self, iterable: Iterable[Instance]):
        """Removes all the instances in iterable from the grid"""
        if not all(isinstance(i, Instance) for i in iterable):
            msg = "Only objects of type Instance can be removed from a CVTArchive"
            raise TypeError(msg)

        indeces_to_remove = self.index_of([i.descriptor for i in iterable])
        for index in indeces_to_remove:
            if index in self._grid:
                del self._grid[index]

    def index_of(self, descriptors) -> np.ndarray:
        """Computes the indeces of a batch of descriptors.

        Args:
            descriptors (array-like): (batch_size, dimensions) array of descriptors for each instance

        Raises:
            ValueError: ``descriptors`` is not shape (batch_size, dimensions)

        Returns:
            np.ndarray:  (batch_size, ) array of integer indices representing the flattened grid coordinates.
        """
        descriptors = np.asarray(descriptors)
        if len(descriptors) == 0:
            return np.empty(0)
        if (
            descriptors.ndim == 1
            and descriptors.shape[0] != self._dimensions
            or descriptors.ndim == 2
            and descriptors.shape[1] != self._dimensions
        ):
            raise ValueError(
                f"Expected descriptors to be an array with shape "
                f"(batch_size, dimensions) (i.e. shape "
                f"(batch_size, {self._dimensions})) but it had shape "
                f"{descriptors.shape}"
            )
        indeces = self._kdtree.query(descriptors, return_distance=False)
        indeces = indeces[:, 0]
        return indeces.astype(np.int32)

    def to_file(self, file_pattern: str = "CVTArchive"):
        """Saves the centroids and the samples of the CVTArchive to .npy files
            Each attribute is saved in its own filename.
            Therefore, file_pattern is expected not to contain any extension

        Args:
            file_pattern (str, optional): Pattern of the expected filenames. Defaults to "CVTArchive".
        """
        np.save(f"{file_pattern}_centroids.npy", self._centroids)
        np.save(f"{file_pattern}_samples.npy", self._samples)

    @classmethod
    def load_from_json(cls, filename: str):
        """Creates a CVTArchive object from the content of a previously created JSON file

        Args:
            filename (str): Filename of the JSON file with the CVTArchive information

        Raises:
            ValueError: If there's any error while loading the file. (IOError)
            ValueError: If the JSON file does not contain all the expected keys

        Returns:
            Self: Returns a CVTArchive object
        """
        expected_keys = {
            "dimensions",
            "n_samples",
            "regions",
            "lbs",
            "ubs",
            "centroids",
            "samples",
        }
        try:
            with open(filename, "r") as file:
                json_data = json.load(file)
                if not expected_keys == json_data.keys():
                    raise ValueError(
                        f"The JSON file does not contain all the expected keys. Expected keys are {expected_keys} and got {json_data.keys()}"
                    )
                _ranges = [
                    (l_i, u_i) for l_i, u_i in zip(json_data["lbs"], json_data["ubs"])
                ]
                new_archive = cls(
                    k=json_data["regions"],
                    ranges=_ranges,
                    n_samples=json_data["n_samples"],
                    centroids=json_data["centroids"],
                    samples=json_data["samples"],
                )
                return new_archive

        except IOError as io:
            raise ValueError(f"Error opening file {filename}. Reason -> {io.strerror}")

    def to_json(self, filename: Optional[str] = None) -> str:
        """Returns the content of the CVTArchive in JSON format.
        It also allows to save the information in a .json file in the current work directory whhen passing a filename

        Args:
            filename (Optional[str], optional): Filename to the CVTArchive. Must include the .json extension. Defaults to None.

        Returns:
            str: String in JSON format with the content of the CVTArchive
        """
        data = {
            "dimensions": self._dimensions,
            "n_samples": self._n_samples,
            "regions": self._k,
            "lbs": self._lower_bounds.tolist(),
            "ubs": self._upper_bounds.tolist(),
            "centroids": self._centroids.tolist(),
            "samples": self._samples.tolist(),
        }
        json_data = json.dumps(data, indent=4)
        if filename is not None:
            filename = (
                f"{filename}.json" if not filename.endswith(".json") else filename
            )
            with open(filename, "w") as f:
                f.write(json_data)

        return json_data
