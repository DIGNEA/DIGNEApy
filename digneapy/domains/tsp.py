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

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from digneapy.core import Domain, Instance, Problem, Solution


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
    so higher values correspond to shorter, better tours. Infeasible tours are
    those that violate the cyclic constraint (meaning that the start and end of the
    tour should be the node number 0) or those that visit a node more than once.

    """

    def __init__(
        self,
        number_of_nodes: np.uint | int,
        coords: np.ndarray,
        penalty_factor: float | np.float64 = 10.0,
        postpone_dist_comp: bool = False,
        save_distances_as_1d: bool = False,
        seed: Optional[int | np.random.SeedSequence] = None,
        *args,
        **kwargs,
    ):
        """Create a new Symmetric Travelling Salesman Problem instance.

        The full Euclidean distance matrix between all pairs of cities is
        pre-computed during initialisation so that repeated evaluations are
        fast.

        Args:
            number_of_nodes (int): Number of cities (nodes) in the instance. It must
                be a positive integer.
            coords (np.ndarray): A two-dimensional array of shape (N, 2) containing
                the x and y coordinates of each city. If a plain sequence is
                supplied it is automatically converted to a NumPy array.
            penalty_factor (np.float64 | float, optional): Penalisation factor used
                to lower the fitness of unfeasible solutions. Defaults to 10.0.
                matrix. Defaults to False.
            postpone_dist_comp (bool, optional): Boolean flag used to indicate that the
                distance matrix should not be precomputed. When True, distances
                are calculated on-the-fly in evaluate, rather than computing all
                the pairwise distances between nodes and creating a flattened 1D distance
                matrix. Defaults to False.
            save_distances_as_1d (bool, optional): Boolean flag to tell the problem
                that the distance matrix must be stored as a flattened 1d array
                using only the upper right triangular matrix. Defaults to False,
                and stores it as a 2d (number_of_nodes x number_of_nodes) matrix.
            seed (Optional[int | np.random.SeedSequence], optional): Seed used to
                initialise the internal random number generator, which is inherited
                from the parent ``Problem`` class. Defaults to None.

        Raises:
            ValueError: If ``coords`` does not have exactly two columns, i.e., its
                shape is not (N, 2).
        """

        try:
            number_of_nodes = int(number_of_nodes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid number_of_nodes in TSP. Got: {number_of_nodes}"
            ) from exc

        try:
            if not isinstance(coords, np.ndarray):
                coords = np.asarray(coords)

            if coords.shape != (number_of_nodes, 2):
                raise ValueError(
                    f"Expected coordinates shape to be ({number_of_nodes}, 2). "
                    f"Instead coords has the following shape: {coords.shape}."
                )

            self._coordinates = np.asarray(coords, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError from exc

        try:
            self._penalty_factor = float(penalty_factor)
            if self._penalty_factor <= 0.0:
                raise ValueError()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "penalty_factor must be a valid positive float value. "
                f"Got {penalty_factor}. From: {exc}"
            ) from exc

        x_min, y_min = np.min(self._coordinates, axis=0)
        x_max, y_max = np.max(self._coordinates, axis=0)
        _bounds = list(((x_min, y_min), (x_max, y_max)) for _ in range(number_of_nodes))

        super().__init__(
            dimension=number_of_nodes, bounds=_bounds, name="TSP", seed=seed
        )
        self._distances = np.empty(1)
        if postpone_dist_comp:
            # If we postpone the computation of the distance
            # matrix, we're essentially working the same as
            # such matrix doesn't fit in memory
            self._too_large_to_fit = True

            self._saved_as_1d = False
        else:
            # This is a Symmetric TSP problem
            # we can store only the upper right triangular matrix
            try:
                self._distances = cdist(self._coordinates, self._coordinates)
                self._saved_as_1d = save_distances_as_1d
                if save_distances_as_1d:
                    self._saved_as_1d = True
                    self._distances = self._distances[
                        np.triu_indices(number_of_nodes, k=1)
                    ]
                self._too_large_to_fit = False
            except Exception:
                # The distance matrix is too large to fit in memory
                # we keep the coordinates and compute the distances
                # on the fly each time
                self._too_large_to_fit = True
                self._saved_as_1d = False

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def distances(self) -> np.ndarray:
        if self._too_large_to_fit or self._distances is None:
            warnings.warn(
                "Distance matrix no calculated in TSP "
                f"for {self.dimension} nodes because it doesn't fit in memory or it was postponed."
                "Returning np.empty()",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.empty(0)
        else:
            return self._distances

    def evaluate(
        self, individual: Sequence | Solution | np.ndarray
    ) -> Tuple[float, ...]:
        """Evaluate a candidate tour and compute its objective value.

        The fitness of a feasible solution is the reciprocal of the total Euclidean
        tour length (``1.0 / distance``). This formulation turns the minimisation
        problem into a maximisation problem, which is consistent with the Digneapy
        framework's convention. Fitness, always to maximise.

        Infeasible tours—those that repeat cities are assigned a penalty
        proportional to the number of repetitions.

        If the supplied ``individual`` is a ``Solution`` object, its ``fitness``,
        ``objectives``, and ``constraints`` attributes are updated in-place.

        Args:
            individual (Sequence | Solution | np.ndarray): The candidate tour to
                evaluate. Must have length ``N``, where ``N`` is the number of
                cities or nodes.

        Raises:
            ValueError: If the length of ``individual`` does not equal ``N ``.

        Returns:
            Tuple[float, float]: A two-element tuple containing the objective value
                and the number of duplicated nodes in the inner tour. This second value
                should be zero for a feasible solution, meaning that all nodes are only
                visited once.
        """
        if len(individual) != self.dimension:
            raise ValueError(
                f"Mismatch between individual variables ({len(individual)})"
                f" and instance variables ({self.dimension}) in TSP. "
                f"A solution for the TSP must be a sequence of len {self.dimension} items. "
                f"Instead got {len(individual)}."
            )

        duplicated_count = len(individual) - len(set(individual))

        # to_node shifts left by 1 to wrap the last node with the first one
        from_node = np.asarray(individual)
        to_node = np.roll(np.asarray(individual), -1)

        if self._too_large_to_fit:
            from_coords = self._coordinates[from_node]
            to_coords = self._coordinates[to_node]
            distance = np.sum(
                np.linalg.norm(from_coords - to_coords, axis=1), dtype=np.float64
            )
        else:
            if self._saved_as_1d:
                # Now calculate from the flatted indices to extract
                # the distances from the flattened 1D distance matrix
                i = np.minimum(from_node, to_node)
                j = np.maximum(from_node, to_node)
                flat_indices = i * self.dimension - (i * (i + 1)) // 2 + (j - i - 1)
                distance = np.sum(self._distances[flat_indices], dtype=np.float64)
            else:
                distance = np.sum(self._distances[from_node, to_node], dtype=np.float64)

        if duplicated_count != 0:
            penalty = np.float64(duplicated_count) * distance * self._penalty_factor
            fitness = 1.0 / (distance + penalty)
        else:
            fitness = 1.0 / distance

        try:
            # We assume that individual is a Solution object
            # and in that case we can update its attributes
            individual.fitness = fitness
            individual.objectives = (fitness,)
            individual.constraints = (duplicated_count,)
        except Exception:
            pass

        return (fitness, duplicated_count)

    def __call__(
        self, individual: Sequence | Solution | np.ndarray
    ) -> Tuple[float, ...]:
        """Evaluate a candidate tour and compute its objective value.

        Delegates directly to :meth:`evaluate`. This makes the problem instance
        callable, allowing it to be used wherever a plain function is expected
        (e.g. as an argument to a solver).

        Args:
            individual (Sequence | Solution | np.ndarray): The candidate tour to
                evaluate. Must have length ``N``.

        Returns:
            Tuple[float, float]: A two-element tuple containing the objective value
                and the number of duplicated nodes in the inner tour. This second value
                should be zero for a feasible solution, meaning that all nodes are only
                visited once.
        """
        return self.evaluate(individual)

    def __str__(self):
        return f"TSP(n={self.dimension})"

    def __len__(self):
        return self.dimension

    def __array__(self, dtype=np.float64, copy: Optional[bool] = None) -> np.ndarray:
        """Return a NumPy array representation of the TSP instance.

        The returned array is the (N, 2) coordinate matrix, where each row stores
        the ``[x, y]`` position of one city. This is useful for serialisation and
        for passing the instance to downstream NumPy-based tools.

        Args:
            dtype: NumPy data type for the returned array. Defaults to
                ``np.float64``.
            copy (Optional[bool]): Whether to force a copy of the underlying data.
                Defaults to ``None``.

        Returns:
            npt.ndarray: The coordinate matrix of shape (N, 2).
        """
        return np.asarray(self._coordinates, dtype=dtype, copy=copy)

    def create_solution(self, random: bool = False, start_node: int = 0) -> Solution:
        """Create a trivial initial solution for the TSP.

        The solution visits cities in natural order: 0 → 1 → 2 → … → N-1 → 0.
        This is a feasible but almost certainly non-optimal tour that can be used
        as a starting point for local search or population initialisation.

        Args:
            random(bool): If True, generate a random permutation of nodes
                instead of the natural-order tour. Defaults to False.
            start_node(int): Starting and endind node of the cyclic tour.
                Defaults to standard initial node (0).
        Returns:
            Solution: A ``Solution`` object whose ``variables`` encode the tour,
                with zeroed ``objectives`` and ``constraints`` arrays.
        """
        items = np.zeros(self.dimension, dtype=np.uint32)
        items[0] = start_node
        remaining_nodes = np.array(
            [c for c in range(self.dimension) if c != start_node], dtype=np.uint32
        )
        if random:
            self._rng.shuffle(remaining_nodes)

        items[1:] = remaining_nodes
        fitness, duplicated = self.evaluate(items)
        return Solution(
            variables=items,
            objectives=(fitness,),
            constraints=(duplicated,),
            fitness=fitness,
        )

    def to_file(self, filename: str | Path = "instance.tsp"):
        """Serialise the TSP instance to a plain text file.

        The file will follow this format:
        - Line 1: number of cities.
        - Line 2: blank separator.
        - Lines3+: one city per line as ``x<TAB>y``.

        Args:
            filename (str | Path, optional): Destination path for the serialised instance.
                Defaults to ``"instance.tsp"``.

        Raises:
            RuntimeError: If something goes wrong.
        """
        try:
            with open(filename, "w") as file:
                file.write(f"{len(self)}\n\n")
                content = "\n".join(f"{x}\t{y}" for (x, y) in self._coordinates)
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
        try:
            with open(filename) as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]

            nodes = np.uint64(lines[0])
            coordinates = [
                [float(coord) for coord in node.split()] for node in lines[2:]
            ]
            coordinates = np.asarray(coordinates, dtype=np.float64)

            return cls(number_of_nodes=nodes, coords=coordinates)
        except Exception as exc:
            raise RuntimeError(
                f"Something went wrong when loading the TSP object: {exc}"
            ) from exc

    def to_instance(self) -> Instance:
        """Convert the TSP problem into an ``Instance`` object used by Digneapy.

        The instance variables are the coordinate array flattened into a
        one-dimensional sequence: ``[x_0, y_0, x_1, y_1, …, x_{N-1}, y_{N-1}]``.

        Returns:
            Instance: An instance object whose ``variables`` contain the
                interleaved x/y coordinates of all cities.
        """
        return Instance(variables=self._coordinates.flatten(), dtype=np.float64)


class TSPDomain(Domain):
    """Domain for synthesising Symmetric TSP instances.

    This class generates benchmark TSP instances by sampling city coordinates
    uniformly within configurable rectangular bounds. It also provides utilities
    for extracting a rich set of geometric and structural features from the
    generated instances and for converting raw instance data back into
    ``TSP`` problem objects ready for a solver.

    Each generated ``Instance`` contains ``2 * number_of_nodes`` variables arranged as
    interleaved x/y coordinates: ``x_0, y_0, x_1, y_1, …, x_{N-1}, y_{N-1}``.

    The eleven descriptive features extracted by this domain are:

    +-----+----------------------+-----------------------------------------------+
    | #   | Name                 | Description                                   |
    +=====+======================+===============================================+
    | 0   | size                 | Number of cities (eq. ``number_of_nodes * 2``)|
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

    features_names = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius".split(
        ","
    )

    def __init__(
        self,
        number_of_nodes: np.uint32 | int = np.uint32(100),
        x_range: Tuple[int, int] = (0, 1000),
        y_range: Tuple[int, int] = (0, 1000),
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Create a new TSPDomain for generating Symmetric TSP instances.

        Args:
            number_of_nodes (np.uint32, optional): Number of cities/nodes in each generated
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
        try:
            if len(x_range) != 2 or len(y_range) != 2:
                raise ValueError(
                    "x_range and y_range must be 2d sequences only to values. "
                    f" Got: x_range = {x_range} and y_range = {y_range}."
                )
            x_min, x_max = x_range
            y_min, y_max = y_range
            if x_min >= x_max:
                raise ValueError(
                    "x_range  must be a 2d sequence (x_min, x_max) "
                    f"where x_min < x_max. Got: x_range {x_range}."
                )
            if y_min >= y_max:
                raise ValueError(
                    "y_range  must be a 2d sequence (y_min, y_max) "
                    f"where y_min < y_max. Got: y_range {y_range}."
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(exc) from exc

        self._x_range = x_range
        self._y_range = y_range
        _bounds = [
            (x_min, x_max) if i % 2 == 0 else (y_min, y_max)
            for i in range(number_of_nodes * 2)
        ]

        super().__init__(
            dimension=number_of_nodes * 2,
            bounds=_bounds,
            domain_name="TSP",
            features_names=self.features_names,
            seed=seed,
        )

    def generate_instances(self, n: np.uint32 | int = np.uint32(1)) -> List[Instance]:
        """Generate a batch of TSP instances by sampling city coordinates at random.

        City x-coordinates are drawn uniformly from ``x_range`` and y-coordinates
        from ``y_range``. Each instance stores the coordinates as a flat vector of
        length ``2 * number_of_nodes`` in the interleaved form
            ``[x_0, y_0, x_1, y_1, …, x_{N-1}, y_{N-1}]``.

        Args:
            n (np.uint32, optional): Number of instances to generate. Defaults to 1.

        Returns:
            List[Instance]: A list of ``n`` ``Instance`` objects, each encoding the
                coordinates of ``dimension`` cities.
        """
        instances = np.empty(shape=(n, self.dimension), dtype=np.float64)
        instances[:, 0::2] = self._rng.uniform(
            low=self._x_range[0],
            high=self._x_range[1],
            size=(n, (self.dimension // 2)),
        )
        instances[:, 1::2] = self._rng.uniform(
            low=self._y_range[0],
            high=self._y_range[1],
            size=(n, (self.dimension // 2)),
        )
        return list(Instance(coords) for coords in instances)

    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[TSP]:
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
        _instances = np.asarray(instances)
        _n_instances, _total_coordinates = _instances.shape
        _n_nodes = _total_coordinates // 2
        # The coordinates of the instances in the batch are reshaped into a 3d matrix
        # (M, 2N) --> (M, N, 2)
        # M = Number of instances
        # N = Number of nodes per instance
        _coordinates = _instances.reshape(_n_instances, _n_nodes, 2)
        return [
            TSP(
                number_of_nodes=_n_nodes,
                coords=coords,
            )
            for coords in _coordinates
        ]

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
        _instances = np.asarray(instances)
        n_instances_batch = len(_instances)
        n_nodes = len(_instances[0]) // 2

        # The coordinates of the instances in the batch are reshaped into a 3d matrix
        # (M, 2N) --> (M, N, 2)
        # M = Number of instances
        # N = Number of nodes per instance
        coords = np.asarray(_instances).reshape((
            n_instances_batch,
            n_nodes,
            2,
        ))
        xs = coords[:, :, 0]
        ys = coords[:, :, 1]
        areas = (
            (np.max(xs, axis=1) - np.min(xs, axis=1))
            * (np.max(ys, axis=1) - np.min(ys, axis=1))
        ).astype(np.float64)

        # Compute distances for all instances
        distances = np.zeros((n_instances_batch, n_nodes, n_nodes))
        differences = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
        distances = np.sqrt(np.sum(differences**2, axis=-1))
        mask = ~np.eye(n_nodes, dtype=bool)
        std_distances = np.std(distances[:, mask], axis=1)

        centroids = np.mean(coords, axis=1)
        expanded_centroids = centroids[:, np.newaxis, :]
        centroids_distances = np.linalg.norm(coords - expanded_centroids, axis=-1)
        radius = np.mean(centroids_distances, axis=1)

        fractions = np.asarray([
            np.unique(d[np.triu_indices_from(d, k=1)]).size
            / (n_nodes * (n_nodes - 1) / 2)
            for d in distances
        ])
        # Top five only
        norm_distances = np.sort(distances, axis=2)[:, :, ::-1][:, :, :5] / np.max(
            distances, axis=(1, 2), keepdims=True
        )

        variance_nnds = np.var(norm_distances, axis=(1, 2))
        variation_nnds = variance_nnds / np.mean(norm_distances, axis=(1, 2))

        cluster_ratio = np.empty(shape=n_instances_batch, dtype=np.float64)
        mean_cluster_radius = np.empty(shape=n_instances_batch, dtype=np.float64)

        for i in range(n_instances_batch):
            scale = np.mean(np.std(coords[i], axis=0))
            eps = 0.2 * scale
            adjacent = distances[i] <= eps
            n_components, labels = connected_components(
                csr_matrix(adjacent), directed=False
            )
            cluster_ratio[i] = n_components / n_nodes

            counts = np.bincount(labels, minlength=n_components)
            cluster_centroids = np.zeros((n_components, 2))
            for dimension in range(2):
                cluster_centroids[:, dimension] = (
                    np.bincount(
                        labels, weights=coords[i][:, dimension], minlength=n_components
                    )
                    / counts
                )
            distances_to_centroid = np.linalg.norm(
                coords[i] - cluster_centroids[labels], axis=1
            )
            radii = (
                np.bincount(
                    labels, weights=distances_to_centroid, minlength=n_components
                )
                / counts
            )
            mean_cluster_radius[i] = np.mean(radii) if radii.size > 0 else 0.0

        return np.column_stack([
            np.full(shape=len(_instances), fill_value=n_nodes),
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
    ) -> List[Dict[str, np.float64]]:
        """Return the extracted features as a list of named dictionaries.

        This is a convenience wrapper around :meth:`extract_features` that pairs
        each numeric value with its human-readable feature name, making the
        output easier to inspect, log, or pass to downstream tools that expect
        labelled data (e.g. ``pandas.DataFrame``).

        Args:
            instances (Sequence[Instance] | np.ndarray): Instances whose features
                should be extracted.

        Returns:
            List[Dict[str, np.float64]]: One dictionary per instance mapping each
                feature name (``size``, ``std_distances``, ``centroid_x``, etc.)
                to its corresponding value.
        """
        features = self.extract_features(instances)
        named_features: list[dict[str, np.float64]] = [{}] * len(features)
        for i, feats in enumerate(features):
            named_features[i] = {k: v for k, v in zip(self.features_names, feats)}
        return named_features
