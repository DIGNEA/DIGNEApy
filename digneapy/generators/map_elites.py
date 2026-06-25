#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   map_elites.py
@Time    :   2026/03/25 12:21:11
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Optional, Sequence

import numpy as np

from digneapy.core import (
    DescriptorPipeline,
    Domain,
    Solver,
    maximise_perf_gap_easy,
)
from digneapy.generators._utils import (
    build_instances_from_attributes,
    extract_solvers_name,
)
from digneapy.typing import PerformanceFn

from ..archives import CVTArchive, GridArchive
from ..operators import (
    BatchUMut,
    Mutation,
)
from ..visualize import ArchivePlotter
from ._base_generator import BaseGenerator, GenerationResult


class MapElites(BaseGenerator):
    """Quality-Diversity instance generator based on the MAP-Elites algorithm.

    MAP-Elites maintains a discretised archive (either a ``GridArchive`` or a
    ``CVTArchive``) where each occupied cell holds the single best instance
    found so far for that region of descriptor space. At every generation, a
    batch of parents is sampled uniformly from the currently filled cells,
    mutated to produce offspring, evaluated against the solver portfolio, and
    inserted back into the archive — each offspring either claims an empty
    cell, replaces the current occupant of its cell if it scores higher, or
    is discarded if it does not improve on the existing elite.

    Unlike :class:`Evolutionary`, fitness in MAP-Elites is simply the
    performance bias of the instance (there is no separate novelty term),
    since diversity is enforced structurally by the archive's discretisation
    rather than by a blended fitness score.
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: np.uint32 | int,
        archive: GridArchive | CVTArchive,
        mutation: Mutation = BatchUMut(seed=None),
        repetitions: np.uint16 | int = np.uint16(1),
        descriptor_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        performance_function: PerformanceFn = maximise_perf_gap_easy,
        generations: np.uint32 | int = np.uint32(1_000),
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Creates a MAP-Elites instance generator.
        The generator uses a set of solvers to evaluate the instances and MAP-Elites
        to guide the evolution of the instances.

        Args:
            domain (Domain): Domain for which the instances are generated for.
            portfolio (Sequence[Solver]): Sequence item of callable objects that can evaluate a instance.
            pop_size (int): Number of instances in the population to evolve. Defaults to 100.
            archive (GridArchive | CVTArchive): Archive to store the instances. It can be a GridArchive or a CVTArchive.
            mutation (Mutation): Mutation operator
            repetitions (int):  Number times a solver in the portfolio must be run over the same instance. Defaults to 1.
            descriptor_pipe (str): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            generations (int): Number of generations to perform. Defaults to 1000.
            seed (int, optional): Seed for the RNG protocol. Defaults to 42.

        Raises:
            ValueError: If the archive is not a GridArchive or CVTArchive

        Note:
            Despite the docstring above, the actual implementation raises a
            ``TypeError`` (not a ``ValueError``) when ``archive`` is not a
            ``GridArchive`` or ``CVTArchive`` instance.
        """
        super().__init__(
            domain,
            portfolio,
            pop_size,
            performance_function,
            descriptor_pipe=descriptor_pipe,
            generations=generations,
            repetitions=repetitions,
            seed=seed,
        )
        if not isinstance(archive, (GridArchive, CVTArchive)):
            raise TypeError(
                f"MapElites expects an archive of class GridArchive or CVTArchive and got {archive.__class__.__name__}"
            )

        self._archive = archive
        self._mutation = mutation

    def __str__(self):
        """Return a human-readable summary of the generator's configuration.

        Includes the population size, number of generations, domain name, and
        the names of the solvers in the portfolio.

        Returns:
            str: A formatted string describing the generator instance.
        """
        solvers_names = tuple(extract_solvers_name(self._portfolio))
        return (
            "Map-Elites Generator:\n"
            f"- Domain {self._domain}\n"
            f"- Portfolio: {solvers_names}\n"
            f"- {self._archive}\n"
            f"- Population Size: {self._pop_size}\n"
            f"- Generations: {self._generations:,}\n"
            f"- Mutation: {self._mutation}\n"
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
            "Map-Elites Generator:\n"
            f"- Domain {self._domain!r}\n"
            f"- Portfolio: {solvers_names!r}\n"
            f"- {self._archive!r}\n"
            f"- Population Size: {self._pop_size!r}\n"
            f"- Generations: {self._generations:,}\n"
            f"- Mutation: {self._mutation}\n"
            f"- Repetitions: {self._repetitions}\n"
            f"- {self._descriptor_pipe}\n"
            f"- Performance Function: {self._performance_fn.__name__}\n"
            f"- Seed: {self.seed}\n"
        )

    @property
    def archive(self):
        """Return the archive backing this generator.

        Returns:
            GridArchive | CVTArchive: The discretised archive that stores the
                current MAP-Elites population, one elite per occupied cell.
        """
        return self._archive

    def _run_generation(self, generation: int, verbose: bool):
        """Perform one MAP-Elites generation: sample, mutate, evaluate, insert.

        A batch of ``pop_size`` parent indices is sampled uniformly (with
        replacement) from the archive's currently filled cells. The
        corresponding parent genotypes are mutated in a single batched call
        via ``self._mutation``, respecting the domain's lower/upper bounds.
        The resulting offspring are evaluated against the solver portfolio
        and their descriptors are computed, then wrapped into ``Instance``
        objects — note that for MAP-Elites, ``fitness`` is set equal to the
        performance bias ``p``, since there is no separate novelty
        component. Finally, the offspring are inserted into the archive,
        where each one may replace its cell's current occupant, fill an
        empty cell, or be rejected, depending on the archive's insertion
        rules. Progress for this generation is recorded in ``self._logbook``.

        Args:
            generation (int): Index of the generation about to be recorded
                (the logbook is updated with ``generation + 1``, since
                generation 0 is reserved for the initial grid).
            verbose (bool): Whether to forward progress feedback to
                ``self._logbook.update``.
        """
        filled_cells_indices = self._rng.choice(
            list(self._archive.filled_cells), size=self._pop_size
        )
        parents_genotypes = np.asarray(
            self._archive.retrieve_filled_cells(filled_cells_indices)
        )
        offs_genotypes = self._mutation(
            population=parents_genotypes,
            lb=self._domain._lbs,
            ub=self._domain._ubs,
        )
        offs_perf_biases, offs_portfolio_scores = self._evaluate_population(
            offs_genotypes
        )
        offs_descriptors = self._descriptor_pipe(
            offs_genotypes, offs_portfolio_scores, self._domain
        )

        offspring = build_instances_from_attributes(
            genotypes=offs_genotypes,
            descriptors=offs_descriptors,
            fitness=offs_perf_biases,
            portfolio_scores=offs_portfolio_scores,
            diversity_scores=np.zeros_like(offs_perf_biases),
            bias_score=offs_perf_biases,
        )
        self._archive.extend(
            instances=offspring,
            descriptors=offs_descriptors,
            objectives=offs_perf_biases,
        )
        # Record the stats and update the performed gens
        self._logbook.update(
            generation=generation + 1,
            instances=self._archive,
            feedback=verbose,
        )

    def _initialise_grid(self, verbose: bool):
        """Seed the archive with an initial batch of randomly generated instances.

        Samples ``pop_size`` fresh instances directly from the domain (rather
        than mutating existing archive members, since the archive starts
        empty), evaluates them against the solver portfolio, computes their
        descriptors, and inserts all of them into the archive unconditionally
        — including infeasible ones, which are expected to be purged later
        via ``self._archive.purge_unfeasible()``. Records generation 0 in
        ``self._logbook``.

        Args:
            verbose (bool): Whether to forward progress feedback to
                ``self._logbook.update``.
        """
        genotypes = np.asarray(self._domain.generate_instances(n=self._pop_size))
        perf_biases, portfolio_scores = self._evaluate_population(genotypes)
        descriptors = self._descriptor_pipe(genotypes, portfolio_scores, self._domain)

        initial_instances = build_instances_from_attributes(
            genotypes=genotypes,
            descriptors=descriptors,
            fitness=perf_biases,
            portfolio_scores=portfolio_scores,
            diversity_scores=np.zeros_like(perf_biases),
            bias_score=perf_biases,
        )

        self._archive.extend(instances=initial_instances, descriptors=descriptors)
        self._logbook.update(
            generation=0, instances=initial_instances, feedback=verbose
        )

    def __call__(self, verbose: bool = False) -> GenerationResult:
        """Run the full MAP-Elites process and return the generated instances.

        The algorithm proceeds as follows:
        1. The archive is seeded with an initial batch of instances via
           :meth:`_initialise_grid` (recorded as generation 0).
        2. For each of the ``self._generations`` subsequent generations,
           :meth:`_run_generation` samples parents from the filled archive
           cells, mutates them, evaluates the offspring, and updates the
           archive in place.
        3. After all generations, any infeasible instances that slipped into
           the archive during initialisation or evolution are removed via
           ``self._archive.purge_unfeasible()``.
        4. The final archive contents are packaged into a
           :class:`GenerationResult` together with the solver names and the
           recorded evolutionary history.

        Args:
            verbose (bool, optional): Whether to print/log progress during the
                run (forwarded to ``self._logbook.update`` at every
                generation). Defaults to False.

        Returns:
            GenerationResult: The final feasible archive contents together
                with solver names and the recorded evolutionary history.
        """

        self._initialise_grid(verbose)
        for generation in range(self._generations):
            # Refactored to use plotted version
            self._run_generation(generation, verbose)

            if verbose:  # pragma: no cover
                # Clear the terminal
                blank = " " * 80
                print(f"\r{blank}\r", end="")

        return GenerationResult(
            solvers=tuple(extract_solvers_name(self._portfolio)),
            instances=self._archive,
            history=self._logbook,
        )


class PlottedMapElites:
    """Wraps a MapElites generator and injects live plotting after each generation.

    It replaces the generator's internal loop with an equivalent one that calls
    ``plotter.update()`` every *refresh_every* generations.  Because it accesses
    the generator's internals (all prefixed ``_``), pin your digneapy version.

    This class does not subclass ``BaseGenerator``: it is a thin orchestration
    wrapper that drives an already-constructed (but not yet called)
    ``MapElites`` instance through the same initialisation and generation
    steps it would normally run on its own, interleaving calls to an
    ``ArchivePlotter`` so that progress can be visualised as a live heatmap
    while the search is running.

    Args:
        generator:       A ``MapElites`` instance (not yet called).
        feat_names:      Labels for the two archive axes.
        attr:            Instance attribute to plot. Default ``"p"``.
        cmap:            Matplotlib colourmap. Default ``"viridis"``.
        vmin / vmax:     Fixed colour limits (``None`` = auto).
        refresh_every:   Redraw every N generations. Default 1.
        save_final:      If not ``None``, saves the final frame to this path.
    """

    def __init__(
        self,
        generator,
        feat_names: Optional[Sequence[str]] = None,
        attr: str = "p",
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        refresh_every: int = 1,
        save_final: Optional[str] = None,
    ):
        """Store the wrapped generator and the plotting configuration.

        No plotting or generation work happens here; everything is deferred
        to :meth:`__call__`. ``refresh_every`` is clamped to a minimum of 1
        to avoid division/modulo issues during the generation loop.

        Args:
            generator: A ``MapElites`` instance that has not yet been called.
            feat_names (Optional[Sequence[str]], optional): Labels for the two
                archive axes shown on the plot. Defaults to None (axes left
                unlabelled or auto-labelled by ``ArchivePlotter``).
            attr (str, optional): Name of the ``Instance`` attribute used to
                colour each cell of the heatmap. Defaults to ``"p"``
                (performance bias).
            cmap (str, optional): Matplotlib colormap name used for the
                heatmap. Defaults to ``"viridis"``.
            vmin (Optional[float], optional): Fixed lower bound for the
                colour scale. ``None`` lets the plotter infer it
                automatically. Defaults to None.
            vmax (Optional[float], optional): Fixed upper bound for the
                colour scale. ``None`` lets the plotter infer it
                automatically. Defaults to None.
            refresh_every (int, optional): Redraw the plot every this many
                generations. Clamped to at least 1. Defaults to 1.
            save_final (Optional[str], optional): If provided, the final
                plotted frame is saved to this path. Defaults to None.
        """
        self._gen = generator
        self._feat_names = feat_names
        self._attr = attr
        self._cmap = cmap
        self._vmin = vmin
        self._vmax = vmax
        self._refresh_every = max(1, refresh_every)
        self._save_final = save_final

    def __call__(self, verbose: bool = False):
        """Runs the full MAP-Elites loop and shows the live heatmap.

        Mirrors the generation sequence performed by ``MapElites.__call__``
        (initial grid seeding followed by the configured number of
        generations), but additionally creates an ``ArchivePlotter`` bound to
        the wrapped generator's archive and refreshes it: once immediately
        after initialisation, every ``refresh_every`` generations during the
        loop, and once more after the final generation. After the loop
        completes, infeasible instances are purged from the archive (as in
        the plain ``MapElites``), the final frame is optionally saved to
        ``save_final``, and the plot window is shown interactively.

        Returns the same ``GenResult`` the underlying generator would return.

        Args:
            verbose (bool, optional): Whether to print/log progress during the
                run (forwarded to the wrapped generator's internal logbook
                updates). Defaults to False.

        Returns:
            GenerationResult: The final feasible archive contents from the
                wrapped generator, together with solver names and the
                recorded evolutionary history.
        """
        self._gen._initialise_grid(verbose)
        plotter = ArchivePlotter(
            self._gen._archive,
            attr=self._attr,
            feat_names=self._feat_names,
            cmap=self._cmap,
            vmin=self._vmin,
            vmax=self._vmax,
            title=f"MAP-Elites — {self._gen._domain.__name__}",
        )
        plotter.update(generation=0)

        for generation in range(self._gen._generations):
            self._gen._run_generation(generation, verbose)
            if (generation + 1) % self._refresh_every == 0:
                plotter.update(generation=generation + 1)

        plotter.update(generation=self._gen._generations)
        self._gen._archive.purge_unfeasible()
        if self._save_final:
            plotter.save(self._save_final)

        plotter.show()
        return GenerationResult(
            solvers=tuple(extract_solvers_name(self._gen._portfolio)),
            instances=self._gen._archive,
            history=self._gen._logbook,
        )
