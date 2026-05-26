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
from typing import Optional

import numpy as np

from digneapy._core.descriptors import DescriptorPipeline

from .._core import (
    Domain,
    Instance,
    Solver,
)
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import Archive
from ..operators import (
    UCX,
    BinarySelection,
    Crossover,
    Generational,
    Mutation,
    Replacement,
    Selection,
    UMut,
)
from ._base_generator import BaseGenerator, GenResult


class Evolutionary(BaseGenerator):
    """Object to generate instances based on a Evolutionary Algorithn with set of diverse solutions"""

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: int,
        archive: Archive,
        solution_set: Optional[Archive] = None,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1000,
        repetitions: int = 1,
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        cxrate: float = 0.5,
        mutrate: float = 0.8,
        crossover: Crossover = UCX(),
        mutation: Mutation = UMut(),
        selection: Selection = BinarySelection(),
        replacement: Replacement = Generational(),
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

            offspring = [
                Instance(
                    variables=offspring[i],
                    fitness=offspring_fitness[i],
                    descriptor=descriptors[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                    s=novelty_scores[i],
                    # Todo: Consider remove features as explicit attribute features=features[i] if features is not None else None,
                )
                for i in range(len(offspring))
            ]
            # Only the feasible instances are considered to be included
            # in the archive and the solution set.
            feasible_indeces = np.where(perf_biases > 0)[0]
            self._archive.extend(
                instances=[offspring[i] for i in feasible_indeces],
                descriptors=descriptors[feasible_indeces],
                novelty_scores=novelty_scores[feasible_indeces],
            )

            if self._solution_set:
                novelty_solution_set = self._solution_set(descriptors=descriptors)
                self._solution_set.extend(
                    instances=[offspring[i] for i in feasible_indeces],
                    descriptors=descriptors[feasible_indeces],
                    novelty_scores=novelty_solution_set[feasible_indeces],
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
            target=self._portfolio[0].__name__,
            instances=_instances,
            history=self._logbook,
        )

    def generate(self, pop_size: int) -> np.ndarray:
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


def cast_to_instances(
    genotypes, descriptors, fitness, portfolio_scores, diversity_scores, bias_score
) -> list[Instance]:
    """Creates objects of type Instance from a collection of np.ndarray

    Args:
        genotypes (_type_): Genotypes of the instances
        descriptors (_type_): Descriptors of the instances
        fitness (_type_): Fitness values of the instances
        portfolio_scores (_type_): Scores of the instances
        diversity_scores (_type_): Diversity scores of the instances
        bias_score (_type_): Performance bias scores of the instances

    Raises:
        RuntimeError: If the len() of any list differs from the rest

    Returns:
        list[Instance]: List of Instance objects ready to be inserted in the archives
    """
    expected = len(genotypes)
    if any(
        len(l) != expected
        for l in (
            genotypes,
            descriptors,
            fitness,
            portfolio_scores,
            diversity_scores,
            fitness,
            bias_score,
        )
    ):
        raise RuntimeError("Length mismatch")
    return [
        Instance(
            variables=genotypes[i],
            fitness=fitness[i],
            descriptor=descriptors[i],
            p=bias_score[i],
            portfolio_scores=portfolio_scores[i],
            s=diversity_scores[i],
        )
        for i in range(expected)
    ]


class ES(BaseGenerator):
    """Object to generate instances based on a Evolutionary Stategy with set of diverse solutions"""

    def __init__(
        self,
        generator_dimension: int,
        domain: Domain,
        portfolio: Sequence[Solver],
        lambda_: int,
        archives: Sequence[Archive],
        keep_only_feasible: bool = True,
        sigma: float = 0.5,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1_000,
        repetitions: int = 1,
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
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
        if descriptor_pipe._key == "performance":
            raise ValueError(
                "ES is expected to be use with a DescriptorPipeline that uses either features or instance as key. Performance not allowed yet!"
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
            descriptors,
            individuals,
            fitness,
            portfolio_scores=portfolio_scores,
            diversity_scores=diversity,
            bias_score=bias_score,
        )
        archive.extend(
            instances=instances,
            descriptors=individuals,
            novelty_scores=diversity,
        )
        return instances

    def __call__(self, verbose: bool = False) -> GenResult:
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
        while _current_generation < self._generations:
            descriptors = np.asarray(strategy.ask())
            individual_genotypes = self._descriptor_pipe(
                descriptors, scores=None, domain=self._domain
            )
            perf_biases, portfolio_scores = self._evaluate_population(
                individual_genotypes
            )
            diversity_scores = np.zeros(shape=(len(descriptors), 1))
            try:
                diversity_scores = self._archives[0](descriptors=descriptors)
            except NotImplementedError:
                diversity_scores = np.zeros(shape=(len(descriptors), 1))

            fitness = self._compute_fitness(perf_biases, diversity_scores)
            instances = self._update_archive(
                self._archives[0],
                individual_genotypes,
                descriptors,
                portfolio_scores,
                diversity_scores,
                perf_biases,
                fitness,
            )
            if len(self._archives) > 1:
                _valid_indices = (
                    np.where(perf_biases > 0)[0]
                    if self._keep_feasible
                    else np.arange(len(individual_genotypes))
                )
                if len(_valid_indices) > 1:
                    # feasible_indeces = np.where(perf_biases > 0)[0]
                    feasible_individuals = individual_genotypes[_valid_indices]
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
                            feasible_individuals,
                            feasible_descriptors,
                            feasible_scores,
                            feasible_div_scores,
                            feasible_performances,
                            feasible_fitness,
                        )

            self._logbook.update(
                generation=_current_generation, population=instances, feedback=verbose
            )
            # Tell the descriptors and their corresponding fitness
            strategy.tell(descriptors, -fitness)

            _current_generation += 1

        _instances = (
            self._archives[0] if len(self._archives) == 1 else self._archives[1]
        )
        return GenResult(
            target=self._portfolio[0].__name__,
            instances=_instances,
            history=self._logbook,
        ), self._archives
