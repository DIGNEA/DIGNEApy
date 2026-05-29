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
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from digneapy.generators._utils import cast_to_instances, extract_solvers_name

from .._core import (
    Domain,
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


@dataclass
class DNSResult:
    """Result of the Dominated Novelty Search computation. The attributes are sorted based on competition fitness."""

    descriptors: np.ndarray
    performances: np.ndarray
    comp_f: np.ndarray


def dominated_novelty_search(
    descriptors: np.ndarray,
    performances: np.ndarray,
    k: np.uint32,
) -> DNSResult:
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
    Raises:
        ValueError: If len(d) where d is the descriptor of each instance i differs from another

    Returns:
        Tuple[np.ndarray]: Tuple with the descriptors, performances and competition fitness values sorted, plus the sorted indexing (descending order).
    """
    if len(performances) != len(descriptors):
        raise ValueError(
            f"Array mismatch between performances and descriptors. len(performance) = {len(performances)} != {len(descriptors)} len(descriptors)"
        )

    num_instances = len(descriptors)
    if num_instances == 0 or k <= 0:
        warnings.warn(
            f"DNS called with either |descriptors| = 0 ({num_instances}) or k <= 0 ({k}). Returning zeros.",
            RuntimeWarning,
            stacklevel=3,
        )
        return DNSResult(
            descriptors,
            performances,
            np.zeros(num_instances),
        )
    k_effective = min(k, num_instances - 1)

    # To penalise unfeasible performances with the hope of getting proper biased instances
    is_feasible = performances >= 0.0
    distances = cdist(descriptors, descriptors)
    np.fill_diagonal(distances, np.inf)

    fitter_mask = (performances[None, :] >= performances[:, None]) & (
        is_feasible[None, :]
    )
    # fitter_mask = np.where(is_unfeasible[None, :], False, fitter_mask)
    distances_fitter = np.where(fitter_mask, distances, np.inf)
    # K smallest distances
    partition = np.partition(distances_fitter, k_effective - 1, axis=1)[:, :k_effective]
    finite_mask = np.isfinite(partition)
    counts = finite_mask.sum(axis=1)
    # Non-dominated instances are set to 1 to avoid ZeroDivisionError
    safe_counts = np.where(counts == 0, 1, counts)
    sums = np.where(finite_mask, partition, 0).sum(axis=1)
    means = sums / safe_counts
    means = np.where(counts == 0, np.inf, means)
    means = np.where(~is_feasible, performances, means)
    return DNSResult(
        descriptors=descriptors,
        performances=performances,
        comp_f=means,
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
        pop_size: np.uint32 = np.uint32(128),
        performance_function: PerformanceFn = max_gap_target,
        generations: np.uint32 = np.uint32(1000),
        repetitions: np.uint16 = np.uint16(1),
        k: np.uint32 = np.uint32(15),
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

        for pgen in range(self._generations):
            offspring = self.generate(self._pop_size)
            off_perf_biases, off_portfolio_scores = self._evaluate_population(offspring)

            off_descriptors = self._descriptor_pipe(
                population=offspring,
                scores=off_portfolio_scores,
                domain=self._domain,
            )
            combined_descriptors = np.concatenate((descriptors, off_descriptors))
            combined_performances = np.concatenate((perf_biases, off_perf_biases))
            combined_port_scores = np.concatenate((portfolio_scores, portfolio_scores))
            genotypes = np.concatenate((np.asarray(self._population), offspring))

            dns_result: DNSResult = dominated_novelty_search(
                descriptors=combined_descriptors,
                performances=combined_performances,
                k=self._k,
            )

            # Keep the top N for the next generation
            sorted_indices = np.argpartition(-dns_result.comp_f, self._pop_size - 1)[
                : self._pop_size
            ]
            sorted_indices = sorted_indices[
                np.argsort(-dns_result.comp_f[sorted_indices])
            ]
            fitness = dns_result.comp_f[sorted_indices]
            descriptors = dns_result.descriptors[sorted_indices]
            perf_biases = dns_result.performances[sorted_indices]
            portfolio_scores = combined_port_scores[sorted_indices]
            genotypes = genotypes[sorted_indices]
            # Both population and offspring are used in the replacement
            # Record the stats and update the performed gens
            self._population = cast_to_instances(
                genotypes=genotypes,
                descriptors=descriptors,
                fitness=fitness,
                portfolio_scores=portfolio_scores,
                diversity_scores=np.zeros_like(fitness),
                bias_score=perf_biases,
            )

            self._logbook.update(
                generation=pgen, population=self._population, feedback=verbose
            )

        if verbose:  # pragma: no cover
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        return GenResult(
            solvers=tuple(extract_solvers_name(self._portfolio)),
            instances=self._population,
            history=self._logbook,
        )
