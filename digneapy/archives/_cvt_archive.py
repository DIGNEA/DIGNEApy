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

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import k_means
from sklearn.neighbors import KDTree

from digneapy.archives._utils import check_valid_instance_batch, check_valid_shapes

from .._core import Instance
from ._archive import Archive, Keys


def compute_centroids(
    n_centroids: int,
    descriptor_dimension: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    samples: int | np.ndarray,
    seed: int | np.random.SeedSequence | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the centroid for a CVTArchive

    Args:
        n_centroids (int): Number of centroids to compute.
        descriptor_dimension (int): Number of dimensions of the descriptor space.
        lower_bounds (np.ndarray): Lower bounds for each dimension of the descriptor space.
        upper_bounds (np.ndarray): Upper bounds for each dimension of the descriptor space.
        samples (int | np.ndarray): Number of samples to calculate the centroids or a NumPy array
            with the precalculated samples.
        seed (int | np.random.SeedSequence | None): Seed to random number generator engine.

    Raises:
        ValueError: If samples is a np.ndarray and the shape doesn't match the expected dimensions.
        RuntimeError: If the generated centroids are less than the required amount.

    Returns:
        Tuple[np.ndarray, np.ndarray]: centroids and samples calculated
    """
    if isinstance(samples, int):
        rng = np.random.default_rng(seed)
        samples = rng.uniform(
            low=lower_bounds, high=upper_bounds, size=(samples, descriptor_dimension)
        )
    else:
        samples = np.asarray(samples)
        if samples.shape[1] != descriptor_dimension:
            raise ValueError(
                f"Loaded samples doesn't have the appropiated shape, expected a (batch_size, {descriptor_dimension}) array. "
                f"Samples is {samples.shape}. "
            )
    try:
        centroids_points = k_means(
            samples,
            n_clusters=n_centroids,
            n_init=1,
            init="random",
            algorithm="lloyd",
            random_state=seed,
        )[0]
        if centroids_points.shape[0] < n_centroids:
            raise RuntimeError(
                "While generating the CVT, k-means clustering found "
                f"{centroids_points.shape[0]} centroids, but this "
                f"archive needs {n_centroids} cells. This most "
                "likely happened because there are too few samples "
                "and/or too many cells."
            )
    except Exception as exc:
        raise RuntimeError("Something went wrong when computing centroids") from exc

    return centroids_points, samples


class CVTArchive(Archive):
    """An Archive that divides a high-dimensional measure space into k homogeneous geometric regions.


    Based on the paper from Vassiliades et al (2018) <https://ieeexplore.ieee.org/document/8000667>

    The computational complexity of the method we provide for constructing the CVT (in Algorithm 1) is O(ndki),
    where n is the number of d-dimensional samples to be clustered, k is the number of clusters,
    and i is the number of iterations needed until convergence
    """

    def __init__(
        self,
        dimensions: int,
        centroids: int | str | Path | np.ndarray,
        ranges: Sequence[Tuple[float, float]],
        samples: int | np.ndarray = 100_000,
        instances: Optional[Sequence[Instance]] = None,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates a CVTArchive

        Args:
            dimensions (int): Number of dimensions of the descriptor space.
            centroids (int | str | Path | np.ndarray): This parameter may be an integer,
                which indicates the number of centroids in the CVT.
                In this case, the centroids will be automatically generated
                with :func:`digneapy.archive.compute_centroids`. Alternatively, this
                parameter can be a (num_centroids, descriptor_dimension) array with the descriptor space
                coordinates of the centroids. Finally, this parameter can specify a file
                holding the centroids; this file will be read with :func:`numpy.load`.
            ranges (Sequence[Tuple[float, float]]): Upper and lower bound of each dimension
                of the descriptor space, e.g. ``[(-1, 1), (-2, 2)]`` indicates the first dimension
                should have bounds :math:`[-1,1]` (inclusive), and the second dimension
                should have bounds :math:`[-2,2]` (inclusive). ``ranges``
                should be the same length as ``descriptor_dimension``.
            samples (int | np.ndarray, optional): This parameter is directly passed
                to :func:`compute_centroids`. Defaults to 100_000.
            instances (Optional[Sequence[Instance]], optional): Initial collection of instances
                to populate the archive. Defaults to None.
            seed (Optional[int | np.random.SeedSequence], optional): Seed for the random engine. Defaults to None.

        Raises:
            ValueError: If the dimension is not an integer or its value is less or equal to zero
            RuntimeError: If the given centroids doesn't have the expected shape
        """
        try:
            self._dimensions = int(dimensions)
            if self._dimensions <= 0:
                raise ValueError("dimensions cannot be negative in CVTArchive")
        except Exception as exc:
            raise ValueError from exc

        if len(ranges) != self._dimensions:
            raise ValueError(
                f"ranges must have {self._dimensions} in CVTArchive. Got: {len(ranges)}"
            )

        super().__init__(initial_instances=None)

        self._seed = seed
        self._rng = np.random.default_rng(seed)

        ranges = list(zip(*ranges))
        self._lower_bounds = np.asarray(ranges[0], dtype=np.float64, copy=True)
        self._upper_bounds = np.asarray(ranges[1], dtype=np.float64, copy=True)
        del ranges

        if isinstance(centroids, int):
            self._centroids, _ = compute_centroids(
                n_centroids=centroids,
                descriptor_dimension=self._dimensions,
                lower_bounds=self._lower_bounds,
                upper_bounds=self._upper_bounds,
                samples=samples,
                seed=seed,
            )
        else:
            try:
                if isinstance(centroids, (str, Path)):
                    self._centroids = np.load(centroids).astype(np.float64)
                else:
                    # Asume is a np.ndarray
                    self._centroids = np.asarray(centroids, copy=True, dtype=np.float64)

                if self._centroids.shape[1] != self._dimensions:
                    raise RuntimeError(
                        "Custom centroids doesn't have the appropiate shape "
                        f"{self._centroids.shape[1]} centroids, but this "
                        f"archive needs centroids of {self._dimensions} cells."
                    )
            except Exception as exc:
                raise ValueError("Custom centroids are not valid.") from exc

        self._kdtree = KDTree(self._centroids)
        del self._storage
        self._storage = {
            Keys.instances: {},
            Keys.descriptors: {},
            Keys.grid: set(),
        }
        if instances is not None:
            self.extend(instances)

    @property
    def dimensions(self) -> int:
        """Dimensions of the measure space used

        Returns:
            int: Dimensions of the measure space used
        """
        return self._dimensions

    @property
    def centroids(self) -> np.ndarray:
        """Returns k centroids calculated from the samples

        Returns:
            np.ndarray: K d-dimensional centroids
        """
        return self._centroids

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bounds of the descriptor space.

        Returns:
            np.ndarray
        """
        return self._lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bounds of the descriptor space.

        Returns:
            np.ndarray
        """
        return self._upper_bounds

    @property
    def instances(self) -> Iterable[Instance]:
        """Instances of the GridArchive

        Returns:
            Iterable[Instance]: Returns a ValueView of the instances
        """
        return self._storage[Keys.instances].values()

    def __str__(self):
        return (
            f"CVTArchive(dim={self._dimensions},centroids=|{self._centroids.shape[0]}|)"
        )

    def __len__(self) -> int:
        """Length of the CVTArchive

        Number of instances stored in the archive

        Returns:
            int: Number of instances stored
        """
        return len(self._storage[Keys.grid])

    def index_of(self, descriptors: np.ndarray) -> np.ndarray:
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
        elif (
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

        indices = self._kdtree.query(descriptors, return_distance=False)
        indices = indices[:, 0]
        return indices.astype(np.int32)

    def extend(
        self,
        instances: Sequence[Instance],
        descriptors: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> None:

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

    def to_file(self, filename: str | Path = "CVTArchive_centroids.npy"):
        """Saves the centroids of the CVTArchive to .npy files

        Args:
            filename (str, optional): Filename. Defaults to "CVTArchive_centroids.npy".
        """
        if self._centroids is None:
            raise AttributeError(
                "Centroids are uninitialised.",
            )
        else:
            try:
                np.save(filename, self._centroids)
            except Exception as exc:
                raise RuntimeError("Couldn't save the centroids in CVTArchive") from exc

    def to_dict(self) -> dict:
        """Converts the CVTArchive into a dictionary

        Includes dimensions, lbs, ubs, centroids and other information from Archive

        Returns:
            dict: Dictionary with the instances stored in the archive
        """
        return {
            "dimensions": self._dimensions,
            "lbs": self._lower_bounds.tolist(),
            "ubs": self._upper_bounds.tolist(),
            "centroids": self._centroids.tolist(),
            **super().to_dict(),
        }

    def to_json(self, filename: Optional[str] = None) -> str:
        """Returns the content of the CVTArchive in JSON format.

        Returns:
            str: String in JSON format with the content of the CVTArchive
        """
        json_data = json.dumps(self.to_dict(), indent=4)
        if filename is not None:
            filename = (
                f"{filename}.json" if not filename.endswith(".json") else filename
            )
            with open(filename, "w") as f:
                f.write(json_data)

        return json_data

    @staticmethod
    def load_from_json(filename: str):
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
            "lbs",
            "ubs",
            "centroids",
        }
        try:
            with open(filename, "r") as file:
                json_data = json.load(file)
                if expected_keys != json_data.keys():
                    raise ValueError(
                        f"The JSON file does not contain all the minimum expected keys. "
                        f"Expected keys are {expected_keys} and got {json_data.keys()}"
                    )
                _ranges = [
                    (l_i, u_i) for l_i, u_i in zip(json_data["lbs"], json_data["ubs"])
                ]
                new_archive = CVTArchive(
                    dimensions=json_data["dimensions"],
                    centroids=json_data["centroids"],
                    ranges=_ranges,
                )
                return new_archive

        except Exception as exc:
            raise ValueError(f"Error opening file {filename}.") from exc
