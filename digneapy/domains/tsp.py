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
from typing import Mapping, Self, Tuple

import numpy as np
from more_itertools import windowed
from sklearn.cluster import DBSCAN

from digneapy._core import Domain, Instance, Problem, Solution


class TSP(Problem):
    """Symmetric Travelling Salesman Problem"""

    def __init__(
        self,
        nodes: int,
        coords: Tuple[Tuple[int, int], ...],
        *args,
        **kwargs,
    ):
        """Creates a new Symmetric Travelling Salesman Problem

        Args:
            nodes (int): Number of nodes/cities in the instance to solve
            coords (Tuple[Tuple[int, int], ...]): Coordinates of each node/city.
        """
        self._nodes = nodes
        self._coords = np.array(coords)
        x_min, y_min = np.min(self._coords, axis=0)
        x_max, y_max = np.max(self._coords, axis=0)
        bounds = list(((x_min, y_min), (x_max, y_max)) for _ in range(self._nodes))
        super().__init__(dimension=nodes, bounds=bounds, name="TSP")

        self._distances = np.zeros((self._nodes, self._nodes))
        for i in range(self._nodes):
            for j in range(i + 1, self._nodes):
                self._distances[i][j] = np.linalg.norm(
                    self._coords[i] - self._coords[j]
                )
                self._distances[j][i] = self._distances[i][j]

    def __evaluate_constraints(self, individual: Sequence | Solution) -> bool:
        counter = Counter(individual)
        if (individual[0] != 0 or individual[-1] != 0) or any(
            counter[c] != 1 for c in counter if c != 0
        ):
            return False
        return True

    def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Travelling Salesmas Problem.

        The fitness of the solution is the fraction of the sum of the distances of the tour
        defined as:
        # TODO: Update with the correct equation (x) = \\frac{\\sum_{k=1}^{N} \\left(\\frac{fill_k}{C}\\right)^2}{N}

        Args:
            individual (Sequence | Solution): Individual to evaluate

        Returns:
            Tuple[float]: Fitness
        """
        if len(individual) != self._nodes + 1:
            msg = f"Mismatch between individual variables ({len(individual)}) and instance variables ({self._nodes}) in {self.__class__.__name__}. A solution for the TSP must be a sequence of len {self._nodes + 1}"
            raise ValueError(msg)

        distance = 0.0
        penalty = 0.0

        if self.__evaluate_constraints(individual):
            for a, b in windowed(individual, n=2):
                distance += self._distances[a][b]
        else:
            distance = np.inf
            penalty = np.inf

        fitness = 1.0 / (distance + penalty)
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

    def create_solution(self) -> Solution:
        items = [0] + list(range(1, self._nodes)) + [0]
        return Solution(chromosome=items)

    def to_file(self, filename: str = "instance.tsp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\n\n")
            content = "\n".join(f"{x}\t{y}" for (x, y) in self._coords)
            file.write(content)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        nodes = int(lines[0])
        coords = []
        for line in lines[2:]:
            x, y = line.split()
            coords.append((int(x), int(y)))

        return cls(nodes=nodes, coords=tuple(coords))

    def to_instance(self) -> Instance:
        return Instance(variables=self._coords.flatten())


class TSPDomain(Domain):
    """Domain to generate instances for the Symmetric Travelling Salesman Problem."""

    def __init__(
        self,
        dimension: int = 100,
        x_range: Tuple[int, int] = (0, 1000),
        y_range: Tuple[int, int] = (0, 1000),
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
        __bounds = [(self._x_range, self._y_range) for _ in range(dimension)]

        super().__init__(dimension=dimension, bounds=__bounds, name="TSP")

    def generate_instance(self) -> Instance:
        """Generates a new instances for the TSP domain

        Returns:
            Instance: New randomly generated instance
        """
        coords = np.random.randint(
            low=(self._x_range[0], self._y_range[0]),
            high=(self._x_range[1], self._y_range[1]),
            size=(self.dimension, 2),
            dtype=int,
        )
        coords = coords.flatten()
        return Instance(coords)

    def extract_features(self, instance: Instance) -> tuple:
        """Extract the features of the instance based on the TSP domain.
           For the TSP the features are:
            - Size
            - Standard deviation of the distances
            - Centroid coordinates
            - Radius of the instance
            - Fraction of distinct instances
            - Rectangular area
            - Variance of the normalised nearest neighbours distances
            - Coefficient of variantion of the nearest neighbours distances
            - Cluster ratio
            - Mean cluster radius
        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple[float]: Values of each feature
        """

        tsp = self.from_instance(instance)
        xs = instance[0::2]
        ys = instance[1::2]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        std_distances = np.std(tsp._distances)
        centroid = (np.mean(xs), np.mean(ys))  # (0.01 * np.sum(xs), 0.01 * np.sum(ys))

        centroid_distance = [np.linalg.norm(city - centroid) for city in tsp._coords]
        radius = np.mean(centroid_distance)

        fraction = len(np.unique(tsp._distances)) / (len(tsp._distances) / 2)
        # Top five only
        norm_distances = np.sort(tsp._distances)[::-1][:5] / np.max(tsp._distances)

        variance_nnNds = np.var(norm_distances)
        variation_nnNds = variance_nnNds / np.mean(norm_distances)

        dbscan = DBSCAN()
        dbscan.fit(tsp._coords)
        cluster_ratio = len(set(dbscan.labels_)) / self.dimension
        # Cluster radius
        mean_cluster_radius = 0.0
        for label_id in dbscan.labels_:
            points_in_cluster = tsp._coords[dbscan.labels_ == label_id]
            cluster_centroid = (
                np.mean(points_in_cluster[:, 0]),
                np.mean(points_in_cluster[:, 1]),
            )
            mean_cluster_radius = np.mean(
                [np.linalg.norm(city - cluster_centroid) for city in tsp._coords]
            )
        mean_cluster_radius /= len(set(dbscan.labels_))

        return (
            self.dimension,
            std_distances,
            centroid[0],
            centroid[1],
            radius,
            fraction,
            area,
            variance_nnNds,
            variation_nnNds,
            cluster_ratio,
            mean_cluster_radius,
        )

    def extract_features_as_dict(self, instance: Instance) -> Mapping[str, float]:
        """Creates a dictionary with the features of the instance.
        The key are the names of each feature and the values are
        the values extracted from instance.

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Mapping[str, float]: Dictionary with the names/values of each feature
        """
        names = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius"
        features = self.extract_features(instance)
        return {k: v for k, v in zip(names.split(","), features)}

    def from_instance(self, instance: Instance) -> TSP:
        n_nodes = len(instance) // 2
        coords = tuple([*zip(instance[::2], instance[1::2])])
        return TSP(nodes=n_nodes, coords=coords)
