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
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping, Tuple

import tqdm

from digneapy.generators import GenerationResult

from ._run import Run, RunFn


class GenerationExperiment:
    def __init__(
        self,
        experiment_name: str,
        base_dir: Path,
        runs_to_do: Sequence[Run] | Sequence[Tuple[str, RunFn]],
        max_workers: int,
    ):
        self._experiment_name = experiment_name
        self._base_dir = base_dir
        self._runs_to_do = runs_to_do
        self._max_workers = max_workers

        if self._base_dir.exists() and not self._base_dir.is_dir():
            raise ValueError(
                "base_dir must be a (possibly non-existing) directory, not a file."
            )
        elif not self._base_dir.exists():
            self._base_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, verbose: bool = True) -> Mapping[Run | RunFn, GenerationResult]:
        if verbose:
            print(f"Starting experiment: {self._experiment_name}")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            self._futures = {}
            for run_number, run in enumerate(self._runs_to_do):
                future = executor.submit(run)
                self._futures[future] = (run_number, run)

            print("All runs submitted!")
            done_run_it = tqdm.tqdm(
                concurrent.futures.as_completed(self._futures),
                total=len(self._runs_to_do),
            )
            results = {}
            for future in done_run_it:
                try:
                    index, run = self._futures[future]
                    if future.cancelled():
                        raise RuntimeError(f"Run {run._name} was cancelled.")
                    tqdm.tqdm.write(f"\t- Run: {run._name} completed! Results saved.")
                    run_result = future.result()
                    results[run] = run_result
                except Exception as e:
                    raise RuntimeError(f"Something went wrong: {e}")

            return results
