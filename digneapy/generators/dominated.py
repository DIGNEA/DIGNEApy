#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   dominated.py
@Time    :   2026/05/22 12:51:26
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np

from .._core import (
    Domain,
    Instance,
    Solver,
)
from .._core.descriptors import DescriptorPipeline
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import UnstructuredArchive
from ..operators import (
    UCX,
    BinarySelection,
    Crossover,
    Mutation,
    Selection,
    UMut,
)
from ._base_generator import GenResult
from .evolutionary import Evolutionary


def dominated_novelty_search(
    descriptors: np.ndarray,
    performances: np.ndarray,
    k: int,
    force_feasible_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dominated Novelty Search (DNS)
        Bahlous-Boldi, R., Faldor, M., Grillotti, L., Janmohamed, H., Coiffard, L., Spector, L., & Cully, A. (2025).
        Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity. 1.
        https://arxiv.org/abs/2502.00593v1

        Quality-Diversity algorithm that implements local competition through dynamic fitness transformations,
        eliminating the need for predefined bounds or parameters. The competition fitness, also known as the dominated novelty score,
        is calculated as the average distance to the k nearest neighbors with higher fitness.

    The method returns a descending sorted list of instances by their competition fitness value.
    For each instance ``i'' in the sequence, we calculate all the other instances that dominate it.
    Then, we compute the distances between their descriptors using the norm of the difference for each dimension of the descriptors.
    Novel instances will get a competition fitness of np.inf (assuring they will survive).
    Less novel instances will be selected by their competition fitness value. This competition mechanism creates two complementary evolutionary
    pressures: individuals must either improve their fitness or discover distinct behaviors that differ from better-performing
    solutions. Solutions that have no fitter neighbors (D𝑖 = ∅) receive an infinite competition fitness, ensuring their preservation in the
    population.

    Args:
        descriptors (np.ndarray): Numpy array with the descriptors of the instances
        performances (np.ndarray): Numpy array with the performance values of the instances
        k (int): Number of nearest neighbours to calculate the competition fitness. Default to 15.
        force_feasible_only (bool): Allow only instances with performance >= 0 to be considered. Default True.
    Raises:
        ValueError: If len(d) where d is the descriptor of each instance i differs from another

    Returns:
        Tuple[np.ndarray]: Tuple with the descriptors, performances and competition fitness values sorted, plus the sorted indexing (descending order).
    """
    warnings.filterwarnings(
        "ignore", message="Mean of empty slice", category=RuntimeWarning
    )
    if len(performances) != len(descriptors):
        raise ValueError(
            f"Array mismatch between performances and descriptors. len(performance) = {len(performances)} != {len(descriptors)} len(descriptors)"
        )
    num_instances = len(descriptors)
    if num_instances <= k:
        msg = f"Trying to calculate the dominated novelty search with k({k}) > len(instances) = {num_instances}"
        raise ValueError(msg)

    # Try to force only feasible performances to get proper biased instances
    is_unfeasible = (
        performances < 0.0 if force_feasible_only else (performances == -np.inf)
    )
    fitter = performances[:, None] <= performances[None, :]
    fitter = np.where(is_unfeasible[None, :], False, fitter)
    np.fill_diagonal(fitter, False)
    distance = np.linalg.norm(
        descriptors[:, None, :] - descriptors[None, :, :], axis=-1
    )
    distance = np.where(fitter, distance, np.inf)
    neg_dist = -distance
    indices = np.argpartition(neg_dist, -k, axis=-1)[..., -k:]
    values = np.take_along_axis(neg_dist, indices, axis=-1)
    indices = np.argsort(values, axis=-1)[..., ::-1]
    values = np.take_along_axis(values, indices, axis=-1)
    indices = np.take_along_axis(indices, indices, axis=-1)
    with np.errstate(invalid="ignore"):
        distance = np.mean(
            -values,
            where=np.take_along_axis(fitter, indices, axis=1),
            axis=-1,
        )
        distance = np.where(np.isnan(distance), np.inf, distance)
    distance = np.where(is_unfeasible, -np.inf, distance)
    sorted_indices = np.argsort(-distance)
    return (
        descriptors[sorted_indices],
        performances[sorted_indices],
        distance[sorted_indices],
        sorted_indices,
    )


class Dominated(Evolutionary):
    """
    Object to generate instances based on a Evolutionary Algorithn
    with a Dominated Novelty Search approach
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: int = 128,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1000,
        repetitions: int = 1,
        k: int = 15,
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = UCX(),
        mutation: Mutation = UMut(),
        selection: Selection = BinarySelection(),
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Sequence[Solver]): Sequence item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 128.
            offspring_size (int, optional): Number of instances in the offspring population. Defaults to 128.
            generations (int, optional): Number of total generations to perform. Defaults to 1000.
            k (int, optional): Number of neighbours to calculate the sparseness. Defaults to 15.
            descriptor_strategy (str, optional): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies. Defaults to "features".
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            crossover (Crossover, optional): Crossover operator. Defaults to uniform_crossover.
            mutation (Mutation, optional): Mutation operator. Defaults to uniform_one_mutation.
            selection (Selection, optional): Selection operator. Defaults to binary_tournament_selection.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
        """
        super().__init__(
            domain=domain,
            portfolio=portfolio,
            pop_size=pop_size,
            performance_function=performance_function,
            generations=generations,
            repetitions=repetitions,
            cxrate=cxrate,
            mutrate=mutrate,
            crossover=crossover,
            mutation=mutation,
            selection=selection,
            descriptor_pipe=descriptor_pipe,
            seed=seed,
            archive=UnstructuredArchive(threshold=0.1, k=1),
        )
        self._k = k
        self.offspring_size = pop_size
        del self._archive

    def __call__(self, verbose: bool = False) -> GenResult:
        if self._domain is None:
            raise ValueError("You must specify a domain to run the generator.")
        if len(self._portfolio) == 0:
            raise ValueError(
                "The portfolio is empty. To run the generator you must provide a valid portfolio of solvers"
            )
        self._population = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(self._population)
        descriptors = self._descriptor_pipe(
            population=self._population,
            scores=portfolio_scores,
            domain=self._domain,
        )

        # if features is not None:
        #     combined_features = np.empty(
        #         shape=(self._pop_size * 2, features.shape[1]), dtype=np.float32
        #     )
        for pgen in range(self._generations):
            offspring = self.generate(self._pop_size)
            off_perf_biases, off_portfolio_scores = self._evaluate_population(offspring)

            off_descriptors = self._descriptor_pipe(
                population=offspring,
                scores=off_portfolio_scores,
                domain=self._domain,
            )
            combined_descriptors = np.concatenate(
                (descriptors, off_descriptors), axis=0
            )
            combined_performances = np.concatenate(
                (perf_biases, off_perf_biases), axis=0
            )
            combined_port_scores = np.concatenate(
                (portfolio_scores, portfolio_scores), axis=0
            )
            genotypes = np.concatenate(
                (np.asarray(self._population, copy=True), offspring), axis=0
            )
            # if features is not None:
            #     combined_features = np.concatenate((features, off_features), axis=0)

            (
                sorted_descriptors,
                sorted_performances,
                sorted_competition_fitness,
                sorted_indexing,
            ) = dominated_novelty_search(
                descriptors=combined_descriptors,
                performances=combined_performances,
                k=self._k,
                force_feasible_only=True,
            )

            # Keep the top N for the next generation
            sorted_indexing = sorted_indexing[: self._pop_size]
            fitness = sorted_competition_fitness[: self._pop_size]
            descriptors = sorted_descriptors[: self._pop_size]
            perf_biases = sorted_performances[: self._pop_size]
            # Track from the combined arrays based on the indexing
            portfolio_scores = combined_port_scores[sorted_indexing]
            genotypes = genotypes[sorted_indexing]
            # if features is not None:
            #     features = combined_features[sorted_indexing]
            # Both population and offspring are used in the replacement
            # Record the stats and update the performed gens
            self._population = [
                Instance(
                    variables=genotypes[i],
                    fitness=fitness[i],
                    descriptor=descriptors[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                    # Todo: Consider remove explicit features attr features=features[i] if features is not None else (),
                )
                for i in range(self._pop_size)
            ]

            self._logbook.update(
                generation=pgen, population=self._population, feedback=verbose
            )

        if verbose:  # pragma: no cover
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        return GenResult(
            target=self._portfolio[0].__name__,
            instances=self._population,
            history=self._logbook,
        )
