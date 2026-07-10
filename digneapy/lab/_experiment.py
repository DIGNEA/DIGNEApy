#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _experiment.py
@Time    :   2026/06/25 12:53:37
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import concurrent.futures
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping

import numpy as np
import tqdm

from digneapy.generators import GenerationResult

from ._run import RunConfig


class GenerationExperiment:
    def __init__(
        self,
        experiment_name: str,
        base_dir: Path,
        runs_to_do: Sequence[RunConfig],
        max_workers: int,
        root_seed: np.random.SeedSequence,
    ):
        """Run a collection of experiment configurations in parallel.

        The experiment creates a shared base directory, stores the root seed,
        and dispatches each configured run through a process pool. Results are
        collected and returned keyed by the run name.

            Args:
                experiment_name (str): Name of the experiment to perform.
                    It will be used to create a sub-directory in the given base_dir directory.
                base_dir (Path): Base directory to store the results.
                runs_to_do (Sequence[RunConfig]): Sequence of RunConfig object that define the
                    runs the experiment must do.
                max_workers (int): Number of processes launched in parallel.
                root_seed (np.random.SeedSequence): Root seed of the experiment. Save in the base_dir
                    directory for reproducibility.

            Raises:
                TypeError: If not all runs_to_do are RunConfig
                ValueError: If base_dir is not a valid directory path
        """
        if not all(isinstance(run, RunConfig) for run in runs_to_do):
            raise TypeError("all runs_to_do must be of type RunConfig")

        self._experiment_name = experiment_name
        self._base_dir = base_dir / experiment_name
        self._root_seed = root_seed
        self._runs_to_do = runs_to_do
        self._max_workers = max_workers

        if self._base_dir.exists() and not self._base_dir.is_dir():
            raise ValueError(
                "base_dir must be a (possibly non-existing) directory, not a file."
            )
        elif not self._base_dir.exists():
            self._base_dir.mkdir(parents=True, exist_ok=True)

        # Logging configuration
        self._logger = logging.getLogger("digneapy.lab")
        logging.basicConfig(
            filename=self._base_dir / f"{experiment_name}.log", level=logging.INFO
        )

        np.savetxt(
            self._base_dir / "root_seed_entropy.txt",
            np.asarray([self._root_seed.entropy]),
            fmt="%i",
        )
        self._logger.info("Structure of directories created successfully.")
        self._logger.info(f"Root seed is {root_seed.entropy}")

    def __call__(self) -> Mapping[str, GenerationResult]:
        """Execute all configured runs and return their generated results.

        Returns:
            A mapping from each run name to the corresponding generation result.
        """
        self._logger.info("Starting experiment.")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            self._futures = {}
            for run_number, run in enumerate(self._runs_to_do):
                future = executor.submit(run, result_directory=self._base_dir)
                self._futures[future] = (run_number, run)

            self._logger.info("All runs submitted!")
            done_run_it = tqdm.tqdm(
                concurrent.futures.as_completed(self._futures),
                total=len(self._runs_to_do),
            )
            results = {}
            for future in done_run_it:
                try:
                    index, run = self._futures[future]
                    if future.cancelled():
                        msg = (
                            f"\t- Run {run.name} was cancelled. "
                            f" Reason: {future.exception}. "
                            "Other runs may continue normally."
                        )
                        self._logger.exception(msg)
                        tqdm.tqdm.write(msg)

                    else:
                        run_result = future.result()
                        results[run.name] = run_result
                        msg = f"\t- Run: {run.name} completed!"
                        self._logger.info(msg)
                        tqdm.tqdm.write(msg)
                except Exception as e:
                    msg = (
                        f"Something went wrong in run {run.name}.\n"
                        f"Reason: {e}. "
                        "Other runs may continue normally."
                    )
                    self._logger.exception(msg)
                    tqdm.tqdm.write(msg)

            return results
