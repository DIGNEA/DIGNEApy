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
from typing import Optional, Protocol, Tuple

import numpy as np
from cma.evolution_strategy import CMAEvolutionStrategy

from digneapy.core import (
    DescriptorPipeline,
    Domain,
    Instance,
    Solver,
    maximise_perf_gap_easy,
)
from digneapy.typing import PerformanceFn

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
from ._base_generator import BaseGenerator, GenerationResult
from ._utils import build_instances_from_attributes, extract_solvers_name


class Evolutionary(BaseGenerator):
    """Quality-Diversity instance generator based on a Genetic Algorithm.

    This generator evolves a population of ``Instance`` genotypes using
    standard genetic operators (selection, crossover, mutation, replacement)
    combined with a Novelty Search mechanism: each candidate instance is
    scored both on solver performance (via the portfolio) and on how
    different it is from previously seen instances (via an ``Archive`` of
    descriptors).
    The two scores are blended into a single fitness value using the ``phi``
    balance parameter, which drives the evolutionary
    process towards instances that are simultaneously challenging for the
    solvers and diverse with respect to the archive.

    Optionally, a second ``Archive`` (``solution_set``) can be maintained to
    keep a curated set of the best/most representative instances separately
    from the exploratory archive used to drive novelty. This ``solution_set``
    only stores feasible instances, those for whom the performance of the
    target solver is better than the rest of the portfolio.
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: np.uint32 | int,
        archive: Archive,
        solution_set: Optional[Archive] = None,
        performance_function: PerformanceFn = maximise_perf_gap_easy,
        generations: np.uint32 | int = np.uint32(1000),
        repetitions: np.uint16 | int = np.uint16(1),
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        cxrate: float | np.float64 = 0.5,
        mutrate: float | np.float64 = 0.8,
        crossover: CrossoverLike = UCX(),
        mutation: MutationLike = UMut(),
        selection: SelectionLike = BinarySelection(),
        replacement: ReplacementLike = Generational(),
        phi: float | np.float64 = 0.85,
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
            TypeError: If ``archive`` is not an instance of ``Archive``.
            TypeError: If ``solution_set`` is provided and is not an instance of ``Archive``.
        """
        super().__init__(
            domain,
            portfolio,
            pop_size,
            performance_function,
            descriptor_pipe,
            generations=generations,
            repetitions=repetitions,
            seed=seed,
        )

        try:
            cxrate = float(cxrate)
            mutrate = float(mutrate)
            phi = float(phi)
            if any(p <= 0.0 or p > 1.0 for p in (cxrate, mutrate, phi)):
                raise ValueError()

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cxrate ({cxrate}), mutrate ({mutrate}) "
                f"and phi ({phi}) must be floating point values in the "
                f"range (0.0, 1.0]. {exc}"
            ) from exc

        if not isinstance(archive, Archive):
            raise TypeError(f"archive ({archive}) must be a subclass of Archive")

        if solution_set is not None and not isinstance(solution_set, Archive):
            raise TypeError(
                f"solution_set ({solution_set}) must be a subclass of Archive"
            )

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

    def __str__(self):
        """Return a human-readable summary of the generator's configuration.

        Includes the population size, number of generations, domain name, and
        the names of the solvers in the portfolio.

        Returns:
            str: A formatted string describing the generator instance.
        """
        solvers_names = tuple(extract_solvers_name(self._portfolio))
        return (
            "Genetic-Based Novelty Search Generator:\n"
            f"- Domain {self._domain}\n"
            f"- Portfolio: {solvers_names}\n"
            f"- Archive: {self._archive}\n"
            f"- Solution set: {self._solution_set}\n"
            f"- Population Size: {self._pop_size}\n"
            f"- Generations: {self._generations:,}\n"
            f"- Crossover rate: {self.cxrate}\n"
            f"- Crossover: {self.crossover}\n"
            f"- Mutation rate: {self.mutrate}\n"
            f"- Mutation: {self.mutation}\n"
            f"- Selection: {self.selection}\n"
            f"- Replacement: {self.replacement}\n"
            f"- Phi: {self.phi}\n"
            f"- Repetitions: {self._repetitions}\n"
            f"- {self._descriptor_pipe}\n"
            f"- Performance Function: {self._performance_fn.__name__}\n"
            f"- Seed (entropy): {self.seed.entropy}\n"
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
            "Genetic-Based Novelty Search Generator:\n"
            f"- Domain {self._domain}\n"
            f"- Portfolio: {solvers_names!r}\n"
            f"- Population Size: {self._pop_size}\n"
            f"- Generations: {self._generations:,}\n"
            f"- Crossover rate: {self.cxrate}\n"
            f"- Crossover: {self.crossover}\n"
            f"- Mutation rate: {self.mutrate}\n"
            f"- Mutation: {self.mutation}\n"
            f"- Selection: {self.selection}\n"
            f"- Repetitions: {self._repetitions}\n"
            f"- {self._descriptor_pipe}\n"
            f"- Performance Function: {self._performance_fn.__name__}\n"
            f"- Seed (entropy): {self.seed.entropy}"
        )

    def __call__(self, verbose: bool = False) -> GenerationResult:
        """Run the full evolutionary process and return the generated instances.

        The algorithm proceeds as follows:
        1. An initial population of ``pop_size`` instances is sampled from
           ``self._domain`` and evaluated against the solver portfolio to
           obtain performance biases and per-solver scores.
        2. Descriptors are computed for the initial population via
           ``self._descriptor_pipe`` (used later for novelty comparisons).
        3. For each of the ``self._generations`` generations:
            - An offspring population of size ``pop_size`` is produced via
              :meth:`generate` (selection + crossover + mutation).
            - The offspring is evaluated against the portfolio and its
              descriptors are computed.
            - Novelty scores are obtained by querying ``self._archive`` with
              the offspring descriptors.
            - A combined fitness is computed via :meth:`_compute_fitness`,
              blending performance bias and novelty according to ``self.phi``.
            - The offspring genotypes are wrapped into ``Instance`` objects
              carrying their fitness, descriptors, and scores.
            - The offspring (all of it, not just feasible individuals — see
              the commented-out filtering logic) is added to ``self._archive``,
              and optionally to ``self._solution_set`` if one was provided.
            - The current population is replaced using ``self.replacement``.
            - Progress for the generation is recorded in ``self._logbook``.
        4. After all generations, the resulting instances (from
           ``self._solution_set`` if present, otherwise from ``self._archive``)
           are packaged into a :class:`GenerationResult` together with the
           solver names and the evolutionary history.

        Args:
            verbose (bool, optional): Whether to print/log progress during the
                run (forwarded to ``self._logbook.update``). Defaults to False.

        Returns:
            GenerationResult: The final set of generated instances together
                with solver names and the recorded evolutionary history.
        """

        self._population = self._domain.generate_instances(n=self._pop_size)
        _, portfolio_scores = self._evaluate_population(self._population)
        _ = self._descriptor_pipe(
            population=self._population,
            scores=portfolio_scores,
            domain=self._domain,
        )
        for generation in range(self._generations):
            offs_genotypes = self.generate(self._pop_size)
            offs_perf_bias, offs_portfolio_scores = self._evaluate_population(
                offs_genotypes
            )
            offs_descriptors = self._descriptor_pipe(
                population=offs_genotypes,
                scores=offs_portfolio_scores,
                domain=self._domain,
            )
            offs_novelty_scores = self._archive(descriptors=offs_descriptors)
            offs_fitness = self._compute_fitness(offs_perf_bias, offs_novelty_scores)

            # Update to include this
            # 1. Novelty Scores --> novelty_scores
            # 2. Performance bias --> perf_biases
            # 3. Fitness --> oiffspring_fitness
            # 4. Descriptor --> descriptors
            offspring = build_instances_from_attributes(
                genotypes=offs_genotypes,
                descriptors=offs_descriptors,
                fitness=offs_fitness,
                portfolio_scores=offs_portfolio_scores,
                diversity_scores=offs_novelty_scores,
                bias_score=offs_perf_bias,
            )

            self._archive.extend(
                instances=offspring,
                descriptors=offs_descriptors,
                novelty_scores=offs_novelty_scores,
                objectives=offs_perf_bias,
            )

            if self._solution_set is not None:
                # Only the feasible instances are considered to be included
                # in the archive and the solution set.
                feasible_indices = np.where(offs_perf_bias > 0)[0]
                offs_feasible_perf_bias = offs_perf_bias[feasible_indices]
                offs_feasible_descriptors = offs_descriptors[feasible_indices]
                offs_feasible_novelty_scores = self._solution_set(
                    descriptors=offs_feasible_descriptors
                )
                self._solution_set.extend(
                    instances=[offspring[i] for i in feasible_indices],
                    descriptors=offs_feasible_descriptors,
                    novelty_scores=offs_feasible_novelty_scores,
                    objectives=offs_feasible_perf_bias,
                )

            # However the whole offspring population is used in the replacement operator
            self._population = self.replacement(self._population, offspring)

            # Record the stats and update the performed gens
            self._logbook.update(
                generation=generation, instances=self._population, feedback=verbose
            )

        if verbose:  # pragma: no cover
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        _instances = (
            self._solution_set if self._solution_set is not None else self._archive
        )
        return GenerationResult(
            solvers=tuple(extract_solvers_name(self._portfolio)),
            instances=_instances,
            history=self._logbook,
        )

    def generate(self, pop_size: np.uint32 | int) -> np.ndarray:
        """Generates a offspring population of size |offspring_size| from the current population

        For each offspring to produce, two parents are drawn from the current
        population using ``self.selection`` and recombined/mutated via the
        private :meth:`__reproduce` helper.

        Args:
            offspring_size (int): offspring size. Defaults to pop_size.

        Returns:
            Sequence[Instance]  Returns a sequence with the instances definitions, the offspring population.
        """
        offspring = [None] * pop_size  # np.empty(offspring_size, dtype=Instance)
        for i in range(pop_size):
            p_1 = self.selection(self._population)
            p_2 = self.selection(self._population)
            child = self._reproduce(p_1, p_2)
            offspring[i] = child

        return np.asarray(offspring)

    def _reproduce(self, parent_1: Instance, parent_2: Instance) -> Instance:
        """Generates a new offspring instance from two parent instances

        ``parent_1`` is cloned to form the base of the offspring. With
        probability ``self.cxrate`` the clone undergoes crossover with
        ``parent_2`` via ``self.crossover``; in either case, the (possibly
        recombined) offspring is then mutated via ``self.mutation`` using the
        domain's lower/upper bounds to keep values within range.

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

        Fitness is a convex combination of performance bias and novelty:
        ``fitness = phi * performance_bias + (1 - phi) * novelty_score``.
        A higher ``self.phi`` favours instances that are harder/more
        discriminating for the solver portfolio, while a lower ``self.phi``
        favours instances that are more novel with respect to the archive.

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


##
class AskFn(Protocol):
    def __call__(
        self,
        strategy: CMAEvolutionStrategy,
        descriptor_pipeline: DescriptorPipeline,
        domain: Domain,
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class CMAAsker:
    def __init__(
        self,
        strategy: CMAEvolutionStrategy,
        descriptor_pipeline: DescriptorPipeline,
        domain: Domain,
        ask_fn: AskFn,
    ):
        self.strategy = strategy
        self.descriptor_pipeline = descriptor_pipeline
        self.domain = domain
        self.ask_fn = ask_fn

    def ask(self):
        return self.ask_fn(self.strategy, self.descriptor_pipeline, self.domain)


def ask_for_descriptors(
    strategy: CMAEvolutionStrategy,
    descriptor_pipeline: DescriptorPipeline,
    domain: Domain,
):
    descriptors = np.asarray(strategy.ask())
    instances = descriptor_pipeline(descriptors, domain=domain)
    return instances, descriptors


def ask_for_instances(
    strategy: CMAEvolutionStrategy,
    descriptor_pipeline: DescriptorPipeline,
    domain: Domain,
):
    instances = np.asarray(strategy.ask())
    descriptors = descriptor_pipeline(instances, domain=domain)
    return instances, descriptors


class ES(BaseGenerator):  # pragma: no cover
    """Quality-Diversity instance generator based on an Evolutionary Strategy (CMA-ES).

    Unlike :class:`Evolutionary`, which evolves ``Instance`` genotypes
    directly with genetic operators, ``ES`` evolves a lower-dimensional
    latent representation (of size ``generator_dimension``) using CMA-ES
    (Covariance Matrix Adaptation Evolution Strategy, via the ``cma``
    package). Each sampled latent vector is decoded into one or more
    instance descriptors/genotypes through ``descriptor_pipe`` (which, for
    this generator, must operate with ``"instance"`` as its key), evaluated
    against the solver portfolio, and scored for diversity against one or
    more archives.

    ``ES`` supports maintaining several archives simultaneously: the first
    archive in ``archives`` is always updated, while any additional archives
    are updated only with the feasible subset of individuals (when
    ``keep_only_feasible`` is True) or with all valid individuals otherwise.
    CMA-ES is told the negated fitness of each sampled point (since CMA-ES
    minimises by convention), with infeasible/out-of-bounds points penalised
    with negative infinity fitness so they are effectively never favoured.
    """

    def __init__(
        self,
        ask_fn: AskFn,
        dimension: int,
        domain: Domain,
        portfolio: Sequence[Solver],
        lambda_: np.uint32 | int,
        archives: Sequence[Archive],
        keep_only_feasible: bool = True,
        sigma: float = 0.5,
        performance_function: PerformanceFn = maximise_perf_gap_easy,
        generations: np.uint32 | int = 1_000,
        repetitions: np.uint16 | int = 1,
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("instance"),
        seed: Optional[int | np.random.SeedSequence] = None,
        phi: float | np.float64 = 0.85,
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
            phi (float, optional): Phi balance value for the weighted fitness function. Defaults to 0.85.

        Raises:
            ValueError: If ``descriptor_pipe`` has no configured transformers.
            ValueError: If ``descriptor_pipe`` is not configured with
                ``"instance"`` as its key (the only mode currently supported
                by ``ES``).
            ValueError: If ``workers`` is negative.
            TypeError: If any element of ``archives`` is not an instance of
                ``Archive``.
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
        if any(param < 0 for param in (workers, dimension)):
            raise ValueError(
                f"dimension ({dimension}) and workers ({workers}) cannot be negative"
            )
        if any(not isinstance(archive, Archive) for archive in archives):
            raise TypeError("archives must be a subclass of Archive")

        self._ask_fn = ask_fn
        self._generator_dimension = dimension
        self._archives = archives
        self._sigma = sigma
        self._phi = float(phi)
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

        Fitness is a fixed 50/50 convex combination of performance bias and
        diversity score: ``fitness = 0.5 * performance_bias + 0.5 * diversity_score``.
        Unlike :meth:`Evolutionary._compute_fitness`, the balance here is not
        configurable via a ``phi`` attribute and is hard-coded to an equal
        weighting.

        Args:
            performance_biases (np.ndarray): Performance biases or scores of each instance
            novelty_scores (np.ndarray): Novelty scores of each instance

        Returns:
            fitness of each instance (np.ndarray)
        """
        phi_r = 1.0 - self._phi
        fitness = np.zeros(len(performance_biases))
        fitness = (performance_biases * self._phi) + (diversity_scores * phi_r)
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
        """Build ``Instance`` objects from raw attributes and insert them into an archive.

        This helper centralises the pattern of packaging the per-individual
        attributes produced during a generation step (genotype, descriptor,
        fitness, solver scores, diversity, and performance bias) into
        ``Instance`` objects via ``build_instances_from_attributes``, and then
        extending the given ``archive`` with those instances.

        Args:
            archive (Archive): The archive to update.
            individuals (np.ndarray): Genotypes of the individuals to add.
            descriptors (np.ndarray): Descriptors associated with each
                individual, used both to build the instances and as the
                archive's novelty-comparison key.
            portfolio_scores (np.ndarray): Per-solver scores obtained when
                evaluating each individual.
            diversity (np.ndarray): Novelty/diversity score of each
                individual with respect to ``archive``.
            bias_score (np.ndarray): Performance bias of each individual.
            fitness (np.ndarray): Combined fitness of each individual.

        Returns:
            np.ndarray: The ``Instance`` objects built from the supplied
                attributes (the same objects inserted into ``archive``).
        """

        instances = build_instances_from_attributes(
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

    def __call__(
        self, verbose: bool = False
    ) -> Tuple[GenerationResult, Sequence[Archive]]:
        """Run the CMA-ES-driven generation process and return the results.

        The algorithm proceeds as follows:
        1. A random initial mean vector ``_x0`` of dimension
           ``self._generator_dimension`` is sampled, and a
           ``cma.CMAEvolutionStrategy`` is initialised with it, the configured
           ``sigma`` (step size), and a population size equal to
           ``self._pop_size`` (i.e. ``lambda_``).
        2. The domain's variable bounds are extracted once as ``mn``/``mx``.
        3. For each generation (up to ``self._generations``):
            - CMA-ES proposes a batch of latent vectors via ``strategy.ask()``;
              these are treated as raw descriptors.
            - The descriptor pipeline decodes the descriptors into concrete
              instance genotypes via ``self._descriptor_pipe``.
            - Genotypes outside the domain's bounds are filtered out via
              ``valid_mask``, since decoders (e.g. for the Knapsack domain)
              may produce out-of-range values.
            - The valid genotypes are evaluated against the solver portfolio
              to obtain performance biases and per-solver scores.
            - Diversity scores are computed against ``self._archives[0]``; if
              that archive does not support novelty scoring
              (``NotImplementedError``), diversity defaults to zero.
            - Fitness is computed via :meth:`_compute_fitness` and the
              primary archive is updated via :meth:`_update_archive`.
            - If more than one archive is configured, the remaining archives
              are updated with either the feasible subset (performance bias
              > 0, when ``self._keep_feasible`` is True) or all valid
              individuals, each scored for diversity against its own archive
              before insertion.
            - Progress is recorded in ``self._logbook``.
            - CMA-ES is informed of the fitness of every sampled point
              (including invalid ones, which receive ``-inf`` fitness so they
              are never preferred) via ``strategy.tell``, using the negated
              fitness since CMA-ES minimises.
        4. After all generations, the resulting instances are taken from
           ``self._archives[0]`` if only one archive was configured, or from
           ``self._archives[1]`` otherwise, and packaged into a
           :class:`GenerationResult`.

        Args:
            verbose (bool, optional): Whether to print/log progress during the
                run (forwarded to ``self._logbook.update``). Defaults to False.

        Returns:
            Tuple[GenerationResult, Sequence[Archive]]: A two-element tuple
                containing the packaged generation result (solver names,
                selected instances, and history) and the full sequence of
                archives maintained throughout the run.
        """
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
        _cma_asker = CMAAsker(
            strategy=strategy,
            descriptor_pipeline=self._descriptor_pipe,
            domain=self._domain,
            ask_fn=self._ask_fn,
        )

        _current_generation = 0
        mn, mx = np.asarray(self._domain.bounds).T
        while _current_generation < self._generations:
            # With this new strategy pattern we have genotypes and descriptors at the same time
            genotypes, descriptors = _cma_asker.ask()

            # Here descriptors have shape (lambda_, generator_dimension)
            # descriptors = np.asarray(strategy.ask())
            # genotypes = self._descriptor_pipe(descriptors, domain=self._domain)
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
                    # If we have more than one archive
                    # we only store the feasible instances
                    # in those result sets.
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
                generation=_current_generation, instances=instances, feedback=verbose
            )
            # Tell the descriptors and their corresponding fitness
            full_fitness = np.full(len(instances), -np.inf)  # Invalid ones get -INF
            full_fitness[valid_mask] = fitness
            # CMA-ES minimises, so -INF becomes large unfeasible individuals
            strategy.tell(descriptors, -full_fitness)
            _current_generation += 1

        _instances = (
            self._archives[0] if len(self._archives) == 1 else self._archives[1]
        )
        return (
            GenerationResult(
                solvers=tuple(extract_solvers_name(self._portfolio)),
                instances=_instances,
                history=self._logbook,
            ),
            self._archives,
        )
