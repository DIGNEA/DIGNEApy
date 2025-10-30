#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   tsp.py
@Time    :   2025/02/21 10:47:31
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from collections import Counter
from collections.abc import Sequence
from typing import Dict, Self, Tuple, List, Optional

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN

from digneapy._core import Domain, Instance, Problem, Solution


class TSP(Problem):
    """Symmetric Travelling Salesman Problem"""

    def __init__(
        self,
        nodes: int,
        coords: np.ndarray,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """Creates a new Symmetric Travelling Salesman Problem

        Args:
            nodes (int): Number of nodes/cities in the instance to solve
            coords (np.ndarray(N, 2)): Coordinates of each node/city.
        """
        self._nodes = nodes
        if coords.shape[1] != 2:
            raise ValueError(
                f"Expected coordinates shape to be (N, 2). Instead coords has the following shape: {coords.shape}"
            )
        if not isinstance(coords, np.ndarray):
            coords = np.asarray(coords)

        self._coords = coords
        x_min, y_min = np.min(self._coords, axis=0)
        x_max, y_max = np.max(self._coords, axis=0)
        bounds = list(((x_min, y_min), (x_max, y_max)) for _ in range(self._nodes))
        super().__init__(dimension=nodes, bounds=bounds, name="TSP", seed=seed)

        self._distances = np.zeros((self._nodes, self._nodes))
        differences = self._coords[:, np.newaxis, :] - self._coords[np.newaxis, :, :]
        self._distances = np.sqrt(np.sum(differences**2, axis=-1))

    def __evaluate_constraints(self, individual: Sequence | Solution) -> bool:
        counter = Counter(individual)
        if any(counter[c] != 1 for c in counter if c != 0) or (
            individual[0] != 0 or individual[-1] != 0
        ):
            return False
        return True

    def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Travelling Salesmas Problem.

        The fitness of the solution is the fraction of the sum of the distances of the tour
        Args:
            individual (Sequence | Solution): Individual to evaluate

        Returns:
            Tuple[float]: Fitness
        """
        if len(individual) != self._nodes + 1:
            msg = f"Mismatch between individual variables ({len(individual)}) and instance variables ({self._nodes}) in {self.__class__.__name__}. A solution for the TSP must be a sequence of len {self._nodes + 1}"
            raise ValueError(msg)

        penalty: np.float64 = np.float64(0)

        if self.__evaluate_constraints(individual):
            distance: float = 0.0
            for i in range(len(individual) - 2):
                distance += self._distances[individual[i]][individual[i + 1]]

            fitness = 1.0 / distance
        else:
            fitness = 2.938736e-39  # --> 1.0 / np.float.max
            penalty = np.finfo(np.float64).max

        if isinstance(individual, Solution):
            individual.fitness = fitness
            individual.objectives = (fitness,)
            individual.constraints = (penalty,)

        return (fitness,)

    def __call__(self, individual: Sequence | Solution) -> tuple[float]:
        return self.evaluate(individual)

    def __repr__(self):
        return f"TSP<n={self._nodes}>"

    def __len__(self):
        return self._nodes

    def __array__(self, dtype=np.float32, copy: Optional[bool] = True) -> npt.ArrayLike:
        return np.asarray(self._coords, dtype=dtype, copy=copy)

    def create_solution(self) -> Solution:
        items = [0] + list(range(1, self._nodes)) + [0]
        return Solution(variables=items)

    def to_file(self, filename: str = "instance.tsp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\n\n")
            content = "\n".join(f"{x}\t{y}" for (x, y) in self._coords)
            file.write(content)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        # TODO: Improve using np.loadtxt
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        nodes = int(lines[0])
        coords = np.zeros(shape=(nodes, 2), dtype=np.float32)
        for i, line in enumerate(lines[2:]):
            x, y = line.split()
            coords[i] = [np.float32(x), np.float32(y)]

        return cls(nodes=nodes, coords=coords)

    def to_instance(self) -> Instance:
        return Instance(variables=self._coords.flatten())


class TSPDomain(Domain):
    """Domain to generate instances for the Symmetric Travelling Salesman Problem."""

    __FEAT_NAMES = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius".split(
        ","
    )

    def __init__(
        self,
        dimension: int = 100,
        x_range: Tuple[int, int] = (0, 1000),
        y_range: Tuple[int, int] = (0, 1000),
        seed: int = 42,
    ):
        """Creates a new TSPDomain to generate instances for the Symmetric Travelling Salesman Problem

        Args:
            dimension (int, optional): Dimension of the instances to generate. Defaults to 100.
            x_range (Tuple[int, int], optional): Ranges for the Xs coordinates of each node/city. Defaults to (0, 1000).
            y_range (Tuple[int, int], optional): Ranges for the ys coordinates of each node/city. Defaults to (0, 1000).

        Raises:
            ValueError: If dimension is < 0
            ValueError: If x_range OR y_range does not have 2 dimensions each
            ValueError: If minimum ranges are greater than maximum ranges
        """
        if dimension < 0:
            raise ValueError(f"Expected dimension > 0 got {dimension}")
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError(
                f"Expected x_range and y_range to be a tuple with only to integers. Got: x_range = {x_range} and y_range = {y_range}"
            )
        x_min, x_max = x_range
        y_min, y_max = y_range
        if x_min < 0 or x_max <= x_min:
            raise ValueError(
                f"Expected x_range to be (x_min, x_max) where x_min >= 0 and x_max > x_min. Got: x_range {x_range}"
            )
        if y_min < 0 or y_max <= y_min:
            raise ValueError(
                f"Expected y_range to be (y_min, y_max) where y_min >= 0 and y_max > y_min. Got: y_range {y_range}"
            )

        self._x_range = x_range
        self._y_range = y_range
        __bounds = [
            (x_min, x_max) if i % 2 == 0 else (y_min, y_max)
            for i in range(dimension * 2)
        ]

        super().__init__(dimension=dimension, bounds=__bounds, name="TSP", seed=seed)

    def generate_instances(self, n: int = 1) -> List[Instance]:
        """Generates N instances using numpy. It can return the instances in two formats:
        1. A numpy ndarray with the definition of the instances
        2. A list of Instance objects created from the raw numpy generation

        Args:
            n (int, optional): Number of instances to generate. Defaults to 1.
            cast (bool, optional): Whether to cast the raw data to Instance objects. Defaults to False.

        Returns:
            List[Instance]: Sequence of instances
        """
        instances = np.empty(shape=(n, self.dimension * 2), dtype=np.float32)
        instances[:, 0::2] = self._rng.uniform(
            low=self._x_range[0],
            high=self._x_range[1],
            size=(n, (self.dimension)),
        )
        instances[:, 1::2] = self._rng.uniform(
            low=self._y_range[0],
            high=self._y_range[1],
            size=(n, (self.dimension)),
        )
        return list(Instance(coords) for coords in instances)

    def extract_features(self, instances: Sequence[Instance]) -> np.ndarray:
        """Extract the features of the instance based on the TSP domain.
           For the TSP the features are:
            - Size
            - Standard deviation of the distances
            - Centroid coordinates
            - Radius of the instance
            - Fraction of distinct distances
            - Rectangular area
            - Variance of the normalised nearest neighbours distances
            - Coefficient of variation of the nearest neighbours distances
            - Cluster ratio
            - Mean cluster radius
        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple[float]: Values of each feature
        """
        _instances = np.asarray(instances, copy=True)
        N_INSTANCES = len(_instances)
        N_CITIES = len(_instances[0]) // 2  # self.dimension // 2
        assert _instances is not instances
        coords = np.asarray(_instances, copy=True).reshape((N_INSTANCES, N_CITIES, 2))
        xs = coords[:, :, 0]
        ys = coords[:, :, 1]
        areas = (
            (np.max(xs, axis=1) - np.min(xs, axis=1))
            * (np.max(ys, axis=1) - np.min(ys, axis=1))
        ).astype(np.float64)

        # Compute distances for all instances
        distances = np.zeros((N_INSTANCES, N_CITIES, N_CITIES))
        differences = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
        distances = np.sqrt(np.sum(differences**2, axis=-1))
        mask = ~np.eye(N_CITIES, dtype=bool)
        std_distances = np.std(distances[:, mask], axis=1)

        centroids = np.mean(coords, axis=1)
        expanded_centroids = centroids[:, np.newaxis, :]
        centroids_distances = np.linalg.norm(coords - expanded_centroids, axis=-1)
        radius = np.mean(centroids_distances, axis=1)

        fractions = np.array(
            [
                np.unique(d[np.triu_indices_from(d, k=1)]).size
                / (N_CITIES * (N_CITIES - 1) / 2)
                for d in distances
            ]
        )
        # Top five only
        norm_distances = np.sort(distances, axis=2)[:, :, ::-1][:, :, :5] / np.max(
            distances, axis=(1, 2), keepdims=True
        )

        variance_nnds = np.var(norm_distances, axis=(1, 2))
        variation_nnds = variance_nnds / np.mean(norm_distances, axis=(1, 2))

        cluster_ratio = np.empty(shape=N_INSTANCES, dtype=np.float64)
        mean_cluster_radius = np.empty(shape=N_INSTANCES, dtype=np.float64)

        for i in range(N_INSTANCES):
            scale = np.mean(np.std(coords[i], axis=0))
            dbscan = DBSCAN(eps=0.2 * scale, min_samples=1)
            labels = dbscan.fit_predict(coords[i])
            unique_labels = [label for label in set(labels) if label != -1]
            cluster_ratio[i] = len(unique_labels) / N_CITIES
            # Cluster radius
            cluster_radius = np.empty(shape=len(unique_labels), dtype=np.float64)
            for j, label_id in enumerate(unique_labels):
                points_in_cluster = coords[i][labels == label_id]
                cluster_centroid = np.mean(points_in_cluster, axis=0)
                cluster_radius[j] = np.mean(
                    np.linalg.norm(points_in_cluster - cluster_centroid, axis=1)
                )

            mean_cluster_radius[i] = (
                np.mean(cluster_radius) if cluster_radius.size > 0 else 0.0
            )
        return np.column_stack(
            [
                np.full(shape=len(_instances), fill_value=N_CITIES),
                std_distances,
                centroids[:, 0],
                centroids[:, 1],
                radius,
                fractions,
                areas,
                variance_nnds,
                variation_nnds,
                cluster_ratio,
                mean_cluster_radius,
            ]
        ).astype(np.float64)

    def extract_features_as_dict(
        self, instances: Sequence[Instance]
    ) -> List[Dict[str, np.float32]]:
        """Creates a dictionary with the features of the instance.
        The key are the names of each feature and the values are
        the values extracted from instance.

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Mapping[str, float]: Dictionary with the names/values of each feature
        """
        features = self.extract_features(instances)
        named_features: list[dict[str, np.float32]] = [{}] * len(features)
        for i, feats in enumerate(features):
            named_features[i] = {k: v for k, v in zip(TSPDomain.__FEAT_NAMES, feats)}
        return named_features

    def generate_problem_from_instance(self, instance: Instance) -> TSP:
        n_nodes = len(instance) // 2
        coords = np.array([*zip(instance[::2], instance[1::2])])
        return TSP(nodes=n_nodes, coords=coords)

    def generate_problems_from_instances(
        self, instances: Sequence[Instance]
    ) -> List[Problem]:
        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances)

        dimension = instances.shape[1] // 2
        return list(
            TSP(
                nodes=dimension, coords=np.array([*zip(instance[0::2], instance[1::2])])
            )
            for instance in instances
        )
