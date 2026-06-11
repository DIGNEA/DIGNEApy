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
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from digneapy._core import Domain, Instance, Problem, Solution


class TSP(Problem):
    """Representation of the Symmetric Travelling Salesman Problem (TSP).

    Given a set of cities, each described by a pair of 2D coordinates, the objective
    is to find the shortest possible tour that visits every city exactly once and
    returns to the starting city. This implementation uses the Euclidean distance
    between coordinate pairs as the inter-city travel cost.

    A candidate solution is encoded as a sequence of city indices of length N + 1,
    where N is the number of cities. The first and last elements must both be 0
    (the depot / starting city), and every city index from 1 to N-1 must appear
    exactly once in between.

    The objective value is the reciprocal of the total tour length (1 / distance),
    so higher values correspond to shorter, better tours. Infeasible tours—those
    that violate the visit-once constraint or do not start and end at city 0—are
    assigned the smallest positive float64 value as a penalty fitness.
    """

    def __init__(
        self,
        nodes: np.uint,
        coords: np.ndarray,
        seed: Optional[int | np.random.SeedSequence] = None,
        *args,
        **kwargs,
    ):
        """Create a new Symmetric Travelling Salesman Problem instance.

        The full Euclidean distance matrix between all pairs of cities is
        pre-computed during initialisation so that repeated evaluations are
        fast.

        Args:
            nodes (int): Number of cities (nodes) in the instance.
            coords (np.ndarray): A two-dimensional array of shape (N, 2) containing
                the x and y coordinates of each city. If a plain sequence is
                supplied it is automatically converted to a NumPy array.
            seed (Optional[int | np.random.SeedSequence], optional): Seed used to
                initialise the internal random number generator, which is inherited
                from the parent ``Problem`` class. Defaults to None.

        Raises:
            ValueError: If ``coords`` does not have exactly two columns, i.e., its
                shape is not (N, 2).
        """
        self._nodes = nodes
        if not isinstance(coords, np.ndarray):
            coords = np.asarray(coords)

        if coords.shape[1] != 2:
            raise ValueError(
                "Expected coordinates shape to be (N, 2). "
                f"Instead coords has the following shape: {coords.shape}"
            )

        self._coords = coords
        x_min, y_min = np.min(self._coords, axis=0)
        x_max, y_max = np.max(self._coords, axis=0)
        bounds = list(((x_min, y_min), (x_max, y_max)) for _ in range(self._nodes))

        super().__init__(dimension=nodes, bounds=bounds, name="TSP", seed=seed)

        self._distances = np.zeros((self._nodes, self._nodes))
        differences = self._coords[:, np.newaxis, :] - self._coords[np.newaxis, :, :]
        self._distances = np.sqrt(np.sum(differences**2, axis=-1))

    def __evaluate_constraints(self, individual: Sequence | Solution) -> bool:
        """Check that a candidate tour satisfies the hard feasibility constraints.

        A tour is considered feasible if and only if:
            1. Every city index from 1 to N-1 appears exactly once in the sequence.
            2. The tour starts at city 0 (``individual[0] == 0``).
            3. The tour ends at city 0 (``individual[-1] == 0``).

        Args:
            individual (Sequence | Solution): The candidate tour to validate.

        Returns:
            bool: ``True`` if all constraints are satisfied
                  ``False`` otherwise.
        """
        counter = Counter(individual)
        if any(counter[c] != 1 for c in counter if c != 0) or (
            individual[0] != 0 or individual[-1] != 0
        ):
            return False
        return True

    def evaluate(self, individual: Sequence | Solution | np.ndarray) -> tuple[float]:
        """Evaluate a candidate tour and compute its objective value.

        The fitness of a feasible solution is the reciprocal of the total Euclidean
        tour length (``1.0 / distance``). This formulation turns the minimisation
        problem into a maximisation problem, which is consistent with the Digneapy
        framework's convention. Fitness, always to maximise.

        Infeasible tours—those that repeat cities or do not begin and end
        at city 0— are assigned a penalty fitness of ``2.938736e-39``
        (approximately ``1.0 / float_max``) and a constraint
        value of ``np.finfo(np.float64).max``.

        If the supplied ``individual`` is a ``Solution`` object, its ``fitness``,
        ``objectives``, and ``constraints`` attributes are updated in-place.

        Args:
            individual (Sequence | Solution | np.ndarray): The candidate tour to
                evaluate. Must have length ``N + 1``, where ``N`` is the number of
                cities, with ``individual[0] == individual[-1] == 0``.

        Raises:
            ValueError: If the length of ``individual`` does not equal ``N + 1``.

        Returns:
            Tuple[float]: A one-element tuple containing the objective value.
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

    def __call__(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        """Evaluate a candidate tour and compute its objective value.

        Delegates directly to :meth:`evaluate`. This makes the problem instance
        callable, allowing it to be used wherever a plain function is expected
        (e.g. as an argument to a solver).

        Args:
            individual (Sequence | Solution | np.ndarray): The candidate tour to
                evaluate. Must have length ``N + 1``.

        Returns:
            Tuple[float]: A one-element tuple containing the objective value.
        """
        return self.evaluate(individual)

    def __str__(self):
        return f"TSP(n={self._nodes})"

    def __len__(self):
        return self._nodes

    def __array__(self, dtype=np.float32, copy: Optional[bool] = True) -> np.ndarray:
        """Return a NumPy array representation of the TSP instance.

        The returned array is the (N, 2) coordinate matrix, where each row stores
        the ``[x, y]`` position of one city. This is useful for serialisation and
        for passing the instance to downstream NumPy-based tools.

        Args:
            dtype: NumPy data type for the returned array. Defaults to
                ``np.float32``.
            copy (Optional[bool]): Whether to force a copy of the underlying data.
                Defaults to ``True``.

        Returns:
            npt.ndarray: The coordinate matrix of shape (N, 2).
        """
        return np.asarray(self._coords, dtype=dtype, copy=copy)

    def create_solution(self) -> Solution:
        """Create a trivial initial solution for the TSP.

        The solution visits cities in natural order: 0 → 1 → 2 → … → N-1 → 0.
        This is a feasible but almost certainly non-optimal tour that can be used
        as a starting point for local search or population initialisation.

        Returns:
            Solution: A ``Solution`` object whose ``variables`` encode the tour,
                with zeroed ``objectives`` and ``constraints`` arrays.
        """
        items = [0] + list(range(1, self._nodes)) + [0]
        return Solution(
            variables=items,
            objectives=np.zeros(1),
            constraints=np.zeros(1),
        )

    def to_file(self, filename: str | Path = "instance.tsp"):
        """Serialise the TSP instance to a plain text file.

        The file will follow this format:
        - Line 1: number of cities.
        - Line 2: blank separator.
        - Lines 3+: one city per line as ``x<TAB>y``.

        Args:
            filename (str | Path, optional): Destination path for the serialised instance.
                Defaults to ``"instance.tsp"``.

        Raises:
            RuntimeError: If something goes wrong.
        """
        try:
            with open(filename, "w") as file:
                file.write(f"{len(self)}\n\n")
                content = "\n".join(f"{x}\t{y}" for (x, y) in self._coords)
                file.write(content)

        except Exception as exc:
            raise RuntimeError(
                "Something went wrong when saving the TSP object."
            ) from exc

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """Load a TSP instance from a text file produced by :meth:`to_file`.

        The expected file format is:
        - Line 1: number of cities.
        - Line 2: blank separator.
        - Lines 3+: one city per line as ``x<TAB>y``.

        Args:
            filename (str): Path to the file containing the serialised instance.

        Raises:
            RuntimeError: If something goes wrong.

        Returns:
            TSP: A new ``TSP`` object rebuilt from the stored city coordinates.
        """
        # TODO: Improve using np.loadtxt
        try:
            with open(filename) as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]

            nodes = np.uint64(lines[0])
            coords = np.zeros(shape=(nodes, 2), dtype=np.float32)
            for i, line in enumerate(lines[2:]):
                x, y = line.split()
                coords[i] = [np.float32(x), np.float32(y)]

            return cls(nodes=nodes, coords=coords)
        except Exception as exc:
            raise RuntimeError(
                "Something went wrong when loading the TSP object"
            ) from exc

    def to_instance(self) -> Instance:
        """Convert the TSP problem into an ``Instance`` object used by Digneapy.

        The instance variables are the coordinate array flattened into a
        one-dimensional sequence: ``[x_0, y_0, x_1, y_1, …, x_{N-1}, y_{N-1}]``.

        Returns:
            Instance: An instance object whose ``variables`` contain the
                interleaved x/y coordinates of all cities.
        """
        return Instance(variables=self._coords.flatten().tolist())


class TSPDomain(Domain):
    """Domain for synthesising Symmetric TSP instances.

    This class generates benchmark TSP instances by sampling city coordinates
    uniformly within configurable rectangular bounds. It also provides utilities
    for extracting a rich set of geometric and structural features from the
    generated instances and for converting raw instance data back into
    ``TSP`` problem objects ready for a solver.

    Each generated ``Instance`` contains ``2 * dimension`` variables arranged as
    interleaved x/y coordinates: ``x_0, y_0, x_1, y_1, …, x_{N-1}, y_{N-1}``.

    The eleven descriptive features extracted by this domain are:

    +-----+----------------------+-----------------------------------------------+
    | #   | Name                 | Description                                   |
    +=====+======================+===============================================+
    | 0   | size                 | Number of cities (always equals ``dimension``)|
    +-----+----------------------+-----------------------------------------------+
    | 1   | std_distances        | Standard deviation of all pairwise distances  |
    +-----+----------------------+-----------------------------------------------+
    | 2   | centroid_x           | x coordinate of the geometric centroid        |
    +-----+----------------------+-----------------------------------------------+
    | 3   | centroid_y           | y coordinate of the geometric centroid        |
    +-----+----------------------+-----------------------------------------------+
    | 4   | radius               | Mean distance from each city to the centroid  |
    +-----+----------------------+-----------------------------------------------+
    | 5   | fraction_distances   | Fraction of unique pairwise distances         |
    +-----+----------------------+-----------------------------------------------+
    | 6   | area                 | Bounding-box area of all city coordinates     |
    +-----+----------------------+-----------------------------------------------+
    | 7   | variance_nnNds       | Variance of normalised nearest-neighbour      |
    |     |                      | distances (top-5)                             |
    +-----+----------------------+-----------------------------------------------+
    | 8   | variation_nnNds      | Coefficient of variation of the normalised    |
    |     |                      | nearest-neighbour distances                   |
    +-----+----------------------+-----------------------------------------------+
    | 9   | cluster_ratio        | Ratio of DBSCAN clusters to number of cities  |
    +-----+----------------------+-----------------------------------------------+
    | 10  | mean_cluster_radius  | Mean radius of the DBSCAN-identified clusters |
    +-----+----------------------+-----------------------------------------------+
    """

    def __init__(
        self,
        dimension: np.uint32 = np.uint32(100),
        x_range: Tuple[int, int] = (0, 1000),
        y_range: Tuple[int, int] = (0, 1000),
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Create a new TSPDomain for generating Symmetric TSP instances.

        Args:
            dimension (np.uint32, optional): Number of cities in each generated
                instance. Defaults to 100.
            x_range (Tuple[int, int], optional): Inclusive lower and upper bounds
                for the x coordinates of sampled cities, expressed as
                ``(x_min, x_max)``. Defaults to ``(0, 1000)``.
            y_range (Tuple[int, int], optional): Inclusive lower and upper bounds
                for the y coordinates of sampled cities, expressed as
                ``(y_min, y_max)``. Defaults to ``(0, 1000)``.
            seed (Optional[int | np.random.SeedSequence], optional): Seed used to
                initialise the internal random number generator. Defaults to None.

        Raises:
            ValueError: If either ``x_range`` or ``y_range`` does not contain
                exactly two elements.
            ValueError: If ``x_min < 0`` or ``x_max <= x_min``.
            ValueError: If ``y_min < 0`` or ``y_max <= y_min``.
        """
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
        _bounds = [
            (x_min, x_max) if i % 2 == 0 else (y_min, y_max)
            for i in range(dimension * 2)
        ]
        features_names = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius".split(
            ","
        )
        super().__init__(
            dimension=dimension,
            bounds=_bounds,
            domain_name="TSP",
            features_names=features_names,
            seed=seed,
        )

    def generate_instances(self, n: np.uint32 = np.uint32(1)) -> List[Instance]:
        """Generate a batch of TSP instances by sampling city coordinates at random.

        City x-coordinates are drawn uniformly from ``x_range`` and y-coordinates
        from ``y_range``. Each instance stores the coordinates as a flat vector of
        length ``2 * dimension`` in the interleaved form
        ``[x_0, y_0, x_1, y_1, …, x_{N-1}, y_{N-1}]``.

        Args:
            n (np.uint32, optional): Number of instances to generate. Defaults to 1.

        Returns:
            List[Instance]: A list of ``n`` ``Instance`` objects, each encoding the
                coordinates of ``dimension`` cities.
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

    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        """Compute an eleven-dimensional feature vector for each supplied instance.

        The features capture the geometric structure of the city layout and are
        intended for use in instance-space analysis and algorithm selection.
        All computations are performed in a vectorised, batch-friendly manner
        where possible; the DBSCAN-based cluster features require a per-instance
        loop.

        Feature descriptions:

        * **size** – Number of cities. Constant across all instances generated
          by the same domain, but included for completeness.
        * **std_distances** – Standard deviation of all pairwise Euclidean
          distances, excluding self-distances. Captures the overall spread of
          city separations.
        * **centroid_x / centroid_y** – The x and y coordinates of the geometric
          centroid (mean position of all cities).
        * **radius** – Mean Euclidean distance from each city to the centroid.
          Indicates how tightly the cities cluster around their centre of mass.
        * **fraction_distances** – The number of unique pairwise distances divided
          by the total number of city pairs ``N*(N-1)/2``. Values close to 1
          indicate that few distances are identical.
        * **area** – Bounding-box area, calculated as
          ``(x_max - x_min) * (y_max - y_min)``.
        * **variance_nnNds** – Variance of the top-5 normalised nearest-neighbour
          distances (normalised by the maximum distance in the instance).
        * **variation_nnNds** – Coefficient of variation (variance / mean) of the
          top-5 normalised nearest-neighbour distances.
        * **cluster_ratio** – Ratio of the number of clusters found by DBSCAN to
          the total number of cities. A low ratio indicates dense clustering; a
          ratio close to 1 indicates that every city forms its own cluster.
        * **mean_cluster_radius** – Average radius of the DBSCAN clusters, where
          each cluster radius is the mean distance of its member cities to the
          cluster centroid.

        Args:
            instances (Sequence[Instance] | np.ndarray): The instances to
                characterise. Each instance must encode ``2 * N`` coordinate
                values in interleaved x/y order.

        Returns:
            np.ndarray: A two-dimensional array of shape
                ``(len(instances), 11)`` where each row contains the feature
                values for the corresponding instance. All values are cast to
                ``np.float64``.
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

        fractions = np.asarray([
            np.unique(d[np.triu_indices_from(d, k=1)]).size
            / (N_CITIES * (N_CITIES - 1) / 2)
            for d in distances
        ])
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

        return np.column_stack([
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
        ]).astype(np.float64)

    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Dict[str, np.float32]]:
        """Return the extracted features as a list of named dictionaries.

        This is a convenience wrapper around :meth:`extract_features` that pairs
        each numeric value with its human-readable feature name, making the
        output easier to inspect, log, or pass to downstream tools that expect
        labelled data (e.g. ``pandas.DataFrame``).

        Args:
            instances (Sequence[Instance] | np.ndarray): Instances whose features
                should be extracted.

        Returns:
            List[Dict[str, np.float32]]: One dictionary per instance mapping each
                feature name (``size``, ``std_distances``, ``centroid_x``, etc.)
                to its corresponding value.
        """
        features = self.extract_features(instances)
        named_features: list[dict[str, np.float32]] = [{}] * len(features)
        for i, feats in enumerate(features):
            named_features[i] = {k: v for k, v in zip(self.features_names, feats)}
        return named_features

    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Problem]:
        """Create ``TSP`` problem objects from a collection of raw instances.

        Each instance is converted from its flat interleaved representation into
        an (N, 2) coordinate array and wrapped in a ``TSP`` object.
        The instances variables are expected to be in the interleaved format
        ``[x_0, y_0, x_1, y_1, …]`` produced by :meth:`generate_instances`.
        The number of cities is inferred as ``len(instance) // 2``.

        Args:
            instances (Sequence[Instance] | np.ndarray): Collection of instances
                to transform. If not already a NumPy array, it is converted
                automatically.

        Returns:
            List[Problem]: A list containing one ``TSP`` problem per input
                instance, in the same order as the input.
        """
        if not isinstance(instances, np.ndarray):
            instances = np.asarray(instances)

        dimension = instances.shape[1] // 2
        return list(
            TSP(
                nodes=dimension,
                coords=np.asarray([*zip(instance[0::2], instance[1::2])]),
            )
            for instance in instances
        )
