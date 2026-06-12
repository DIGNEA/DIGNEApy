#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   evolutionary.py
@Time    :   2026/03/25 12:20:42
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Optional, Tuple

import numpy as np

from digneapy._core.descriptors import DescriptorPipeline

from .._core import (
    Domain,
    Instance,
    Solver,
)
from .._core.scores import PerformanceFn, maximise_perf_gap_easy
from ..archives import Archive
from ..operators import (
    UCX,
    BinarySelection,
    CrossoverLike,
    Generational,
    MutationLike,
    ReplacementLike,
    SelectionLike,
    UMut,
)
from ._base_generator import BaseGenerator, GenResult
from ._utils import cast_to_instances, extract_solvers_name


class Evolutionary(BaseGenerator):
    """Object to generate instances based on a Evolutionary Algorithn with set of diverse solutions"""

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: np.uint32,
        archive: Archive,
        solution_set: Optional[Archive] = None,
        performance_function: PerformanceFn = maximise_perf_gap_easy,
        generations: np.uint32 = np.uint32(1000),
        repetitions: np.uint16 = np.uint16(1),
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: CrossoverLike = UCX(),
        mutation: MutationLike = UMut(),
        selection: SelectionLike = BinarySelection(),
        replacement: ReplacementLike = Generational(),
        phi: float = 0.85,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search
        The generator uses a set of solvers to evaluate the instances and
        a novelty search algorithm to guide the evolution of the instances.

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Sequence[Solver]): Sequence item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            archive (Archive): Container to store diverse instances.
            solution_set (Optional[Archive], optional): Solution set to store the instances. Defaults to None.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            describe_by (DESCRIPTORS, optional): _Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.DESCRIPTORS. Defaults to "features".
            cxrate (float, optional): Crossover rate. Defaults to 0.5.
            mutrate (float, optional): Mutation rate. Defaults to 0.8.
            crossover (Crossover, optional): Crossover operator. Defaults to uniform_crossover.
            mutation (Mutation, optional): Mutation operator. Defaults to uniform_one_mutation.
            selection (Selection, optional): Selection operator. Defaults to binary_tournament_selection.
            replacement (Replacement, optional): Replacement operator. Defaults to generational_replacement.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            phi (float, optional): Phi balance value for the weighted fitness function. Defaults to 0.85.
            seed (int, optional): Seed for the RNG protocol. Defaults to 42.

        Raises:
            ValueError: Raises error if phi is not a floating point value or it is not in the range [0.0-1.0]
            KeyError: Raises error if the descriptor strategy is not available in the DESCRIPTORS dictionary
        """
        super().__init__(
            domain,
            portfolio,
            pop_size,
            performance_function,
            descriptor_pipe,
            generations=generations,
            repetitions=repetitions,
        )

        if any(
            type(param) not in (float,) or (param < 0.0 or param > 1.0)
            for param in (cxrate, mutrate, phi)
        ):
            msg = f"Invalid parameters. cxrate, mutrate and phi must be a float number in the range [0.0-1.0]. Got: cxrate={cxrate}, mutrate={mutrate}, phi={phi}."
            raise ValueError(msg)

        if not isinstance(archive, Archive):
            raise TypeError("archive must be a subclass of Archive")

        if solution_set is not None and not isinstance(solution_set, Archive):
            raise TypeError("solution_set must be a subclass of Archive")

        self.phi = phi
        self._archive = archive
        self._solution_set = solution_set
        self.offspring_size = self._pop_size
        self.cxrate = cxrate
        self.mutrate = mutrate
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.replacement = replacement
        self.seed = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )

        self._rng = np.random.default_rng(self.seed)

    def __call__(self, verbose: bool = False) -> GenResult:

        self._population = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(self._population)
        descriptors = self._descriptor_pipe(
            population=self._population, scores=portfolio_scores, domain=self._domain
        )
        for pgen in range(self._generations):
            offspring = self.generate(self._pop_size)
            perf_biases, portfolio_scores = self._evaluate_population(offspring)
            descriptors = self._descriptor_pipe(
                population=offspring,
                scores=portfolio_scores,
                domain=self._domain,
            )
            novelty_scores = self._archive(descriptors=descriptors)
            offspring_fitness = self._compute_fitness(perf_biases, novelty_scores)

            # Update to include this
            # 1. Novelty Scores --> novelty_scores
            # 2. Performance bias --> perf_biases
            # 3. Fitness --> oiffspring_fitness
            # 4. Descriptor --> descriptors
            offspring = cast_to_instances(
                genotypes=offspring,
                descriptors=descriptors,
                fitness=offspring_fitness,
                portfolio_scores=portfolio_scores,
                diversity_scores=novelty_scores,
                bias_score=perf_biases,
            )

            # Only the feasible instances are considered to be included
            # in the archive and the solution set.
            # feasible_indices = np.where(perf_biases > 0)[0]
            # self._archive.extend(
            #     instances=[offspring[i] for i in feasible_indices],
            #     descriptors=descriptors[feasible_indices],
            #     novelty_scores=novelty_scores[feasible_indices],
            #     objectives=perf_biases[feasible_indices],
            # )
            self._archive.extend(
                instances=offspring,
                descriptors=descriptors,
                novelty_scores=novelty_scores,
                objectives=perf_biases,
            )

            if self._solution_set is not None:
                novelty_solution_set = self._solution_set(descriptors=descriptors)
                self._solution_set.extend(
                    instances=offspring,
                    descriptors=descriptors,
                    novelty_scores=novelty_solution_set,
                    objectives=perf_biases,
                )

            # However the whole offspring population is used in the replacement operator
            self._population = self.replacement(self._population, offspring)
            # Record the stats and update the performed gens
            self._logbook.update(
                generation=pgen, population=self._population, feedback=verbose
            )

        if verbose:  # pragma: no cover
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        _instances = (
            self._solution_set if self._solution_set is not None else self._archive
        )
        return GenResult(
            solvers=tuple(extract_solvers_name(self._portfolio)),
            instances=_instances,
            history=self._logbook,
        )

    def generate(self, pop_size: np.uint32) -> np.ndarray:
        """Generates a offspring population of size |offspring_size| from the current population

        Args:
            offspring_size (int): offspring size. Defaults to pop_size.

        Returns:
            Sequence[Instance]  Returns a sequence with the instances definitions, the offspring population.
        """
        offspring = [None] * pop_size  # np.empty(offspring_size, dtype=Instance)
        for i in range(pop_size):
            p_1 = self.selection(self._population)
            p_2 = self.selection(self._population)
            child = self.__reproduce(p_1, p_2)
            offspring[i] = child

        return np.asarray(offspring, copy=True)

    def __reproduce(self, parent_1: Instance, parent_2: Instance) -> Instance:
        """Generates a new offspring instance from two parent instances

        Args:
            parent_1 (Instance): First Parent
            parent_1 (Instance): Second Parent

        Returns:
            Instance: New offspring
        """
        offspring = parent_1.clone()
        if self._rng.random() < self.cxrate:
            offspring = self.crossover(offspring, parent_2)
            return self.mutation(offspring, self._domain.lbs, self._domain.ubs)
        else:
            return self.mutation(offspring, self._domain.lbs, self._domain.ubs)

    def _compute_fitness(
        self, performance_biases: np.ndarray, novelty_scores: np.ndarray
    ) -> np.ndarray:
        """Calculates the fitness of each instance in the population

        Args:
            performance_biases (np.ndarray): Performance biases or scores of each instance
            novelty_scores (np.ndarray): Novelty scores of each instance

        Returns:
            fitness of each instance (np.ndarray)
        """
        phi_r = 1.0 - self.phi
        fitness = np.zeros(len(performance_biases))
        fitness = (performance_biases * self.phi) + (novelty_scores * phi_r)
        return fitness


class ES(BaseGenerator):
    """Object to generate instances based on a Evolutionary Stategy with set of diverse solutions"""

    def __init__(
        self,
        generator_dimension: int,
        domain: Domain,
        portfolio: Sequence[Solver],
        lambda_: np.uint32,
        archives: Sequence[Archive],
        keep_only_feasible: bool = True,
        sigma: float = 0.5,
        performance_function: PerformanceFn = maximise_perf_gap_easy,
        generations: np.uint32 = np.uint32(1_000),
        repetitions: np.uint16 = np.uint16(1),
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("instance"),
        seed: Optional[int | np.random.SeedSequence] = None,
        workers: int = 1,
    ):
        """Creates a Evolutionary Instance Generator based on Novelty Search
        The generator uses a set of solvers to evaluate the instances and
        a novelty search algorithm to guide the evolution of the instances.

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Sequence[Solver]): Sequence item of callable objects that can evaluate a instance.
            pop_size (int, optional): Number of instances in the population to evolve. Defaults to 100.
            archives (Sequence[Archive]): Containers to store diverse instances.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            generations (int, optional): Number of generations to perform. Defaults to 1000.
            repetitions (int, optional): Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            describe_by (DESCRIPTORS, optional): _Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.DESCRIPTORS. Defaults to "features".
            seed (int, optional): Seed for the RNG protocol. Defaults to 42.

        Raises:
            KeyError: Raises error if the descriptor strategy is not available in the DESCRIPTORS dictionary
        """
        if len(descriptor_pipe._transformers) == 0:
            raise ValueError(
                "ES is expected to be use with a DescriptorPipeline that includes at least one transformer object."
            )
        if descriptor_pipe._key != "instance":
            raise ValueError(
                "ES is expected to be use with a DescriptorPipeline that uses instance as key. Performance or Features not allowed yet!"
            )

        super().__init__(
            domain=domain,
            portfolio=portfolio,
            pop_size=lambda_,
            performance_function=performance_function,
            descriptor_pipe=descriptor_pipe,
            generations=generations,
            repetitions=repetitions,
        )
        if any(param < 0 for param in (workers,)):
            raise ValueError(
                f"These parameters cannot be negative:\n\t- workers. Got {workers}"
            )
        if any(not isinstance(archive, Archive) for archive in archives):
            raise TypeError("archives must be a subclass of Archive")
        self._generator_dimension = generator_dimension
        self._archives = archives
        self._sigma = sigma
        self._workers = workers
        self.seed = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )

        self._rng = np.random.default_rng(self.seed)
        self._keep_feasible = keep_only_feasible

    def _compute_fitness(
        self, performance_biases: np.ndarray, diversity_scores: np.ndarray
    ) -> np.ndarray:
        """Calculates the fitness of each instance in the population

        Args:
            performance_biases (np.ndarray): Performance biases or scores of each instance
            novelty_scores (np.ndarray): Novelty scores of each instance

        Returns:
            fitness of each instance (np.ndarray)
        """
        phi_r = 1.0 - 0.5
        fitness = np.zeros(len(performance_biases))
        fitness = (performance_biases * 0.5) + (diversity_scores * phi_r)
        return fitness

    def _update_archive(
        self,
        archive: Archive,
        individuals: np.ndarray,
        descriptors: np.ndarray,
        portfolio_scores: np.ndarray,
        diversity: np.ndarray,
        bias_score: np.ndarray,
        fitness: np.ndarray,
    ):

        instances = cast_to_instances(
            genotypes=individuals,
            descriptors=descriptors,
            fitness=fitness,
            portfolio_scores=portfolio_scores,
            diversity_scores=diversity,
            bias_score=bias_score,
        )
        archive.extend(
            instances=instances,
            descriptors=individuals,
            novelty_scores=diversity,
            objectives=bias_score,
        )
        return instances

    def __call__(self, verbose: bool = False) -> Tuple[GenResult, Sequence[Archive]]:
        import cma

        _x0 = self._rng.uniform(
            size=(self._generator_dimension),
        )
        strategy = cma.CMAEvolutionStrategy(
            _x0,
            sigma0=self._sigma,
            inopts={
                "popsize": self._pop_size,
            },
        )
        _current_generation = 0
        mn, mx = np.asarray(self._domain.bounds).T
        while _current_generation < self._generations:
            # Here descriptors have shape (lambda_, generator_dimension)
            descriptors = np.asarray(strategy.ask())
            genotypes = self._descriptor_pipe(descriptors, domain=self._domain)
            # Some genotypes may be outside of the bounds of the domain
            # For instance, KPDecoder could generate negative Qs and items
            valid_mask = ((genotypes >= mn) & (genotypes <= mx)).all(axis=1)
            valid_genotypes = genotypes[valid_mask]
            valid_descriptors = descriptors[valid_mask]
            perf_biases, portfolio_scores = self._evaluate_population(valid_genotypes)
            try:
                diversity_scores = self._archives[0](descriptors=valid_descriptors)
            except NotImplementedError:
                diversity_scores = np.zeros(shape=(len(valid_descriptors), 1))

            fitness = self._compute_fitness(perf_biases, diversity_scores)
            instances = self._update_archive(
                self._archives[0],
                individuals=valid_descriptors,  # Valid genotypes extracted from the transformed
                descriptors=valid_descriptors,
                portfolio_scores=portfolio_scores,
                diversity=diversity_scores,
                bias_score=perf_biases,
                fitness=fitness,
            )
            if len(self._archives) > 1:
                _valid_indices = (
                    np.where(perf_biases > 0)[0]
                    if self._keep_feasible
                    else np.arange(len(valid_descriptors))
                )
                if len(_valid_indices) > 1:
                    feasible_descriptors = descriptors[_valid_indices]
                    feasible_performances = perf_biases[_valid_indices]
                    feasible_scores = portfolio_scores[_valid_indices]
                    feasible_fitness = fitness[_valid_indices]
                    for archive in self._archives[1:]:
                        try:
                            feasible_div_scores = archive(
                                descriptors=descriptors[_valid_indices]
                            )
                        except NotImplementedError:
                            feasible_div_scores = np.zeros(
                                shape=(len(descriptors[_valid_indices]), 1)
                            )
                        _ = self._update_archive(
                            archive,
                            individuals=feasible_descriptors,
                            descriptors=feasible_descriptors,
                            portfolio_scores=feasible_scores,
                            diversity=feasible_div_scores,
                            bias_score=feasible_performances,
                            fitness=feasible_fitness,
                        )

            self._logbook.update(
                generation=_current_generation, population=instances, feedback=verbose
            )
            # Tell the descriptors and their corresponding fitness
            full_fitness = np.full(len(descriptors), -np.inf)  # Invalid ones get -INF
            full_fitness[valid_mask] = fitness
            # CMA-ES minimises, so -INF becomes large unfeasible individuals
            strategy.tell(descriptors, -full_fitness)
            _current_generation += 1

        _instances = (
            self._archives[0] if len(self._archives) == 1 else self._archives[1]
        )
        return (
            GenResult(
                solvers=tuple(extract_solvers_name(self._portfolio)),
                instances=_instances,
                history=self._logbook,
            ),
            self._archives,
        )
