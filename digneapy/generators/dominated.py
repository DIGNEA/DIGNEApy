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

from digneapy.core import (
    DescriptorPipeline,
    Domain,
    Solver,
    maximise_perf_gap_easy,
)
from digneapy.typing import PerformanceFn

from ..archives import UnstructuredArchive
from ..operators import (
    UCX,
    BinarySelection,
    Crossover,
    Mutation,
    Selection,
    UMut,
)
from ._base_generator import GenerationResult
from ._utils import (
    build_instances_from_attributes,
    extract_solvers_name,
)
from .evolutionary import Evolutionary


@dataclass
class DNSResult:
    """Container for the outputs of a Dominated Novelty Search computation.

    Bundles together the descriptors, performance values, and competition
    fitness scores produced by ``dominated_novelty_search`` for a pooled set
    of individuals (e.g. a population merged with its offspring). All three
    arrays are aligned by index — that is, ``descriptors[i]``,
    ``performances[i]``, and ``competition_fitness[i]`` all refer to the same
    individual — and are sorted in descending order of ``competition_fitness``,
    so the most competitive (highest-fitness) individual is always at index 0.
    """

    descriptors: np.ndarray
    """np.ndarray: Descriptor vectors of each individual, used to measure
    novelty/diversity relative to its neighbours. Shape is
    ``(n_individuals, descriptor_dim)``."""

    performances: np.ndarray
    """np.ndarray: Performance bias (or raw solver performance score) of each
    individual. Shape is ``(n_individuals,)``."""

    competition_fitness: np.ndarray
    """np.ndarray: Combined fitness score (``comp_f``) for each individual,
    derived from both performance and local novelty against its ``k``
    nearest neighbours in descriptor space. This is the value used to rank
    and select individuals during survival/replacement. Shape is
    ``(n_individuals,)``."""


def dominated_novelty_search(
    descriptors: np.ndarray,
    performances: np.ndarray,
    k: np.uint32 | int,
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
        competition_fitness=means,
    )


class Dominated(Evolutionary):
    """Quality-Diversity instance generator using Dominated Novelty Search.

    ``Dominated`` is a variant of :class:`Evolutionary` that replaces the
    archive-based novelty mechanism with Dominated Novelty Search (DNS): at
    each generation, parents and offspring are pooled together and ranked
    using a combined fitness (``comp_f``) computed by
    ``dominated_novelty_search`` from each individual's descriptor and
    performance bias relative to its ``k`` nearest neighbours in the pooled
    set. The top ``pop_size`` individuals by combined fitness survive into
    the next generation, acting simultaneously as both selection and
    replacement.

    Because DNS computes novelty directly from the pooled population/offspring
    rather than from a persistent archive, this class does not maintain an
    ``Archive`` for novelty purposes: the inherited ``self._archive`` is
    explicitly deleted after initialisation, and no ``solution_set`` support
    is exposed.
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: np.uint32 | int = np.uint32(128),
        performance_function: PerformanceFn = maximise_perf_gap_easy,
        generations: np.uint32 | int = np.uint32(1000),
        repetitions: np.uint16 | int = np.uint16(1),
        k: np.uint32 | int = np.uint32(15),
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = UCX(),
        mutation: Mutation = UMut(),
        selection: Selection = BinarySelection(),
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search

        Internally this delegates to ``Evolutionary.__init__`` with a
        throwaway ``UnstructuredArchive`` (since the parent class requires
        one), which is then discarded via ``del self._archive`` once
        construction completes, as ``Dominated`` computes novelty directly
        from the pooled population/offspring rather than from a persistent
        archive.

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

        Raises:
            ValueError: Propagated from ``Evolutionary.__init__`` if ``cxrate``,
                ``mutrate``, or ``phi`` (fixed internally) are invalid.
            TypeError: Propagated from ``Evolutionary.__init__`` if the
                internally constructed placeholder archive is invalid (not
                expected to occur under normal use).
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
            archive=UnstructuredArchive(novelty_threshold=0.1, k=1),
        )
        self._k = k
        del self._archive

    def __str__(self):
        """Return a human-readable summary of the generator's configuration.

        Includes the population size, number of generations, domain name, and
        the names of the solvers in the portfolio.

        Returns:
            str: A formatted string describing the generator instance.
        """
        solvers_names = tuple(extract_solvers_name(self._portfolio))
        return (
            "Dominated Novelty Search Generator:\n"
            f"- Domain {self._domain}\n"
            f"- Portfolio: {solvers_names}\n"
            f"- Population Size: {self._pop_size}\n"
            f"- Nearest Neighbours (k): {self._k}\n"
            f"- Generations: {self._generations:,}\n"
            f"- Crossover rate: {self.cxrate}\n"
            f"- Crossover: {self.crossover}\n"
            f"- Mutation rate: {self.mutrate}\n"
            f"- Mutation: {self.mutation}\n"
            f"- Selection: {self.selection}\n"
            f"- Repetitions: {self._repetitions}\n"
            f"- {self._descriptor_pipe}\n"
            f"- Performance Function: {self._performance_fn.__name__}\n"
            f"- Seed: {self.seed}\n"
        )

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the generator.

        Reuses :meth:`__str__` but swaps the surrounding parentheses for angle
        brackets, following the convention used elsewhere in the framework.

        Returns:
            str: A compact representation suitable for debugging/logging.
        """
        solvers_names = tuple(extract_solvers_name(self._portfolio))
        return (
            "Dominated Novelty Search Generator:\n"
            f"- Domain {self._domain}\n"
            f"- Portfolio: {solvers_names!r}\n"
            f"- Population Size: {self._pop_size}\n"
            f"- Nearest Neighbours (k): {self._k}\n"
            f"- Generations: {self._generations:,}\n"
            f"- Crossover rate: {self.cxrate}\n"
            f"- Crossover: {self.crossover}\n"
            f"- Mutation rate: {self.mutrate}\n"
            f"- Mutation: {self.mutation}\n"
            f"- Selection: {self.selection}\n"
            f"- Repetitions: {self._repetitions}\n"
            f"- {self._descriptor_pipe}\n"
            f"- Performance Function: {self._performance_fn.__name__}\n"
            f"- Seed: {self.seed}"
        )

    def __call__(self, verbose: bool = False) -> GenerationResult:
        """Run the Dominated Novelty Search evolutionary process.

        The algorithm proceeds as follows:
        1. Validates that a domain and a non-empty solver portfolio were
           provided.
        2. Samples an initial population of ``pop_size`` instances and
           evaluates it against the portfolio to obtain performance biases
           and per-solver scores, then computes descriptors for it.
        3. For each of the ``self._generations`` generations:
            - Generates an offspring population of size ``pop_size`` via
              :meth:`Evolutionary.generate` (inherited selection + crossover
              + mutation).
            - Evaluates the offspring and computes its descriptors.
            - Concatenates the current population and the offspring into a
              single pool of descriptors, performance biases, portfolio
              scores, and genotypes.
            - Runs ``dominated_novelty_search`` over the pooled descriptors
              and performances (using ``self._k`` neighbours) to obtain a
              combined fitness (``comp_f``) for every pooled individual.
            - Selects the indices of the top ``pop_size`` individuals by
              combined fitness (via ``np.argpartition`` followed by a sort of
              just that subset, for efficiency) to form the next generation.
            - Rebuilds the survivors as ``Instance`` objects (with diversity
              scores set to zero, since DNS folds novelty into ``comp_f``
              rather than tracking it separately) and assigns them to
              ``self._population``.
            - Records progress for the generation in ``self._logbook``.
        4. After all generations, returns a :class:`GenerationResult`
           containing the solver names, the final surviving population, and
           the evolutionary history.

        Args:
            verbose (bool, optional): Whether to print/log progress during the
                run (forwarded to ``self._logbook.update``). Defaults to False.

        Raises:
            ValueError: If ``self._domain`` is ``None``.
            ValueError: If ``self._portfolio`` is empty.

        Returns:
            GenerationResult: The final surviving population together with
                solver names and the recorded evolutionary history.
        """

        self._population = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(self._population)
        descriptors = self._descriptor_pipe(
            population=self._population,
            scores=portfolio_scores,
            domain=self._domain,
        )

        for generation in range(self._generations):
            offs_genotypes = self.generate(self.offspring_size)
            off_perf_biases, off_portfolio_scores = self._evaluate_population(
                offs_genotypes
            )

            off_descriptors = self._descriptor_pipe(
                population=offs_genotypes,
                scores=off_portfolio_scores,
                domain=self._domain,
            )
            # We need to combine the offspring and old population
            # attributes to compute the dominated fitness score
            combined_descriptors = np.concatenate((descriptors, off_descriptors))
            combined_performances = np.concatenate((perf_biases, off_perf_biases))
            combined_port_scores = np.concatenate((
                portfolio_scores,
                off_portfolio_scores,
            ))
            combined_genotypes = np.concatenate((
                np.asarray(self._population),
                offs_genotypes,
            ))

            # Result of the computation of the DNS algorithm
            dominated_result: DNSResult = dominated_novelty_search(
                descriptors=combined_descriptors,
                performances=combined_performances,
                k=self._k,
            )

            # We keep the top N best instances
            # based on the competition fitness
            # for the next generation
            best_indices = np.argpartition(
                -dominated_result.competition_fitness, self._pop_size - 1
            )[: self._pop_size]
            sorted_indices = best_indices[
                np.argsort(-dominated_result.competition_fitness[best_indices])
            ]
            # We now extract all the attributes of the best instances
            # from both the offspring and population
            fitness = dominated_result.competition_fitness[sorted_indices]
            descriptors = dominated_result.descriptors[sorted_indices]
            perf_biases = dominated_result.performances[sorted_indices]
            portfolio_scores = combined_port_scores[sorted_indices]
            genotypes = combined_genotypes[sorted_indices]
            # Both population and offspring are used in the replacement
            # Record the stats and update the performed gens
            self._population = build_instances_from_attributes(
                genotypes=genotypes,
                descriptors=descriptors,
                fitness=fitness,
                portfolio_scores=portfolio_scores,
                diversity_scores=np.zeros_like(fitness),
                bias_score=perf_biases,
            )

            self._logbook.update(
                generation=generation, instances=self._population, feedback=verbose
            )

        if verbose:  # pragma: no cover
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        return GenerationResult(
            solvers=tuple(extract_solvers_name(self._portfolio)),
            instances=self._population,
            history=self._logbook,
        )
