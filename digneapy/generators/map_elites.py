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

from .._core import (
    Domain,
    Instance,
    Solver,
)
from .._core.descriptors import DescriptorPipeline
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import CVTArchive, GridArchive
from ..operators import (
    BatchUMut,
    Mutation,
)
from ..visualize import ArchivePlotter
from ._base_generator import BaseGenerator, GenResult


class MapElites(BaseGenerator):
    """Object to generate instances based on MAP-Elites algorithm."""

    def __init__(
        self,
        domain: Domain,
        portfolio: Sequence[Solver],
        pop_size: int,
        archive: GridArchive | CVTArchive,
        mutation: Mutation = BatchUMut(seed=None),
        repetitions: int = 1,
        describe_pipe: DescriptorPipeline = DescriptorPipeline("features"),
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1_000,
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
            describe_pipe (str): Descriptor used to calculate the diversity. The options available are defined in the dictionary digneapy.qd.descriptor_strategies.
            performance_function (PerformanceFn, optional): Performance function to calculate the performance score. Defaults to max_gap_target.
            generations (int): Number of generations to perform. Defaults to 1000.
            seed (int, optional): Seed for the RNG protocol. Defaults to 42.

        Raises:
            ValueError: If the archive is not a GridArchive or CVTArchive
        """
        super().__init__(
            domain,
            portfolio,
            pop_size,
            performance_function,
            descriptor_pipe=describe_pipe,
            generations=generations,
            repetitions=repetitions,
        )
        if not isinstance(archive, (GridArchive, CVTArchive)):
            raise TypeError(
                f"MapElites expects an archive of class GridArchive or CVTArchive and got {archive.__class__.__name__}"
            )

        self._archive = archive
        self._mutation = mutation
        self._seed_sequence = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        me_seed, mut_seed = self._seed_sequence.spawn(2)
        self._rng = np.random.default_rng(me_seed)
        self.__mut_seed = mut_seed

    @property
    def archive(self):
        return self._archive

    def _run_generation(self, generation: int, verbose: bool):
        indices = self._rng.choice(
            list(self._archive.filled_cells), size=self._pop_size
        )
        parents = np.asarray(self._archive[indices], copy=True)
        offspring = self._mutation(
            parents,
            self._domain._lbs,
            ub=self._domain._ubs,
        )
        perf_biases, portfolio_scores = self._evaluate_population(offspring)
        descriptors = self._descriptor_pipe(offspring, portfolio_scores, self._domain)

        offspring_population = [
            Instance(
                variables=offspring[i],
                fitness=perf_biases[i],  # In MapElites fitness == performance_bias (p)
                descriptor=descriptors[i],
                portfolio_scores=portfolio_scores[i],
                p=perf_biases[i],
            )
            for i in range(len(offspring))
        ]
        self._archive.extend(instances=offspring_population, descriptors=descriptors)
        # Record the stats and update the performed gens
        self._logbook.update(
            generation=generation + 1, population=self._archive, feedback=verbose
        )

    def _initialise_grid(self, verbose: bool):
        instances = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(instances)
        descriptors = self._descriptor_pipe(instances, portfolio_scores, self._domain)
        initial_instances = [
            Instance(
                variables=instances[i].variables,
                fitness=perf_biases[i],
                descriptor=descriptors[i],
                portfolio_scores=portfolio_scores[i],
                p=perf_biases[i],
            )
            for i in range(len(instances))
        ]
        # Here we do not care for p >= 0. We are starting the archive
        # Must be removed later on
        self._archive.extend(instances=initial_instances, descriptors=descriptors)
        self._logbook.update(generation=0, population=instances, feedback=verbose)

    def __call__(self, verbose: bool = False) -> GenResult:
        # Here we do not care for p >= 0. We are starting the archive
        # Must be removed later on
        self._initialise_grid(verbose)
        for generation in range(self._generations):
            # Refactored to use plotted version
            self._run_generation(generation, verbose)
        if verbose:  # pragma: no cover
            # Clear the terminal
            blank = " " * 80
            print(f"\r{blank}\r", end="")

        self._archive.purge_unfeasible()
        return GenResult(
            target=self._portfolio[0].__name__,
            instances=self._archive,
            history=self._logbook,
        )


class PlottedMapElites:
    """Wraps a MapElites generator and injects live plotting after each generation.

    It replaces the generator's internal loop with an equivalent one that calls
    ``plotter.update()`` every *refresh_every* generations.  Because it accesses
    the generator's internals (all prefixed ``_``), pin your digneapy version.

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

        Returns the same ``GenResult`` the underlying generator would return.
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

        self._gen._archive.purge_unfeasible()
        plotter.update(generation=self._gen._generations)

        if self._save_final:
            plotter.save(self._save_final)

        plotter.show()
        return GenResult(
            target=self._gen._portfolio[0].__name__,
            instances=self._gen._archive,
            history=self._gen._logbook,
        )
