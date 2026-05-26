#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sphere.py
@Time    :   2026/05/20 15:31:48
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from digneapy._core import Domain, Instance, Problem, Solution


class Sphere(Problem):
    """Minimises the shifted sphere: f(x) = Σ (xᵢ − centerᵢ)²

    Fitness is returned as *−f(x)* so that higher is better, matching the
    maximisation convention used throughout digneapy.
    """

    def __init__(
        self,
        dimension: int,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError(
                f"Dimension must be a positive integer. Got {type(dimension)} = {dimension}"
            )
        bounds = [(-5.12, 5.12)] * dimension
        super().__init__(
            dimension=dimension,
            bounds=bounds,
            name="Sphere",
            dtype=np.float64,
            seed=seed,
        )

    def evaluate(
        self, individual: Sequence | Solution | np.ndarray
    ) -> tuple[np.float64]:
        """Returns (−sphere_value,) so higher fitness = closer to centre."""
        x = np.asarray(individual, dtype=np.float64)
        sphere_val = np.float64(np.sum(x**2))
        return (sphere_val,)

    def create_solution(self) -> Solution:
        """Creates a random feasible solution."""
        variables = self._rng.uniform(self._lbs, self._ubs)
        (fitness,) = self.evaluate(variables)
        return Solution(
            variables=variables,
            objectives=[fitness],
            fitness=fitness,
            dtype=np.float64,
            otype=np.float64,
        )

    def __call__(self, individual: Sequence | Solution | np.ndarray) -> tuple[float]:
        """Alias for evaluate — makes SphereProblem directly callable."""
        return self.evaluate(individual)

    def to_instance(self) -> Instance:
        """Converts this problem back to an Instance (center as variables)."""
        return Instance(variables=self._rng.uniform(self.lbs, self.ubs, size=2))

    def to_file(self, filename: str):
        """Persists the problem centre to a plain text file."""
        raise NotImplementedError("Not implemented here. Sphere is just for testing.")

    def __array__(self, dtype=None, copy: Optional[bool] = None) -> np.ndarray:
        raise NotImplementedError("Not implemented here. Sphere is just for testing.")

    def __repr__(self) -> str:  # pragma: no cover
        raise NotImplementedError("Not implemented here. Sphere is just for testing.")


class SphereDomain(Domain):
    """Domain of 2-D shifted sphere problems.

    Each *instance* is a 2-D point (x₀, x₁) that serves as the centre of a
    sphere. The two coordinates are also the features used as descriptors,
    so they map directly onto a 2-D GridArchive.

    Args:
        dimension (int): Dimensionality of the sphere. Defaults to 2.
        lb (float): Lower bound for each variable. Defaults to −5.12.
        ub (float): Upper bound for each variable. Defaults to +5.12.
        seed (int | None): RNG seed.
    """

    FEAT_NAMES = ["x0", "x1"]

    def __init__(
        self,
        dimension: int = 2,
        lb: float = -5.12,
        ub: float = 5.12,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        bounds = [(lb, ub)] * dimension
        feat_names = [f"x{i}" for i in range(dimension)]
        super().__init__(
            dimension=dimension,
            bounds=bounds,
            name="Sphere",
            feat_names=feat_names,
            dtype=np.float64,
            seed=seed,
        )

    def generate_instances(self, n: int = 1) -> List[Instance]:
        """Generates n random sphere centre points as Instance objects."""
        points = self._rng.uniform(
            low=self._lbs, high=self._ubs, size=(n, self._dimension)
        )
        return [
            Instance(variables=row, otype=np.float64, dtype=np.float64)
            for row in points
        ]

    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List:
        """Creates one SphereProblem per instance (using variables as centre)."""
        return [Sphere(dimension=2) for _ in instances]

    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        """Returns the raw instance coordinates as features.

        Shape: (n_instances, dimension).
        For a 2-D domain these directly populate a 2-D GridArchive.
        """
        arr = np.asarray(instances, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)

    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Dict[str, np.float32]]:
        """Returns a list of {feat_name: value} dicts, one per instance."""
        features = self.extract_features(instances)
        return [
            {name: val for name, val in zip(self.feat_names, row)} for row in features
        ]
