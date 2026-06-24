#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   save_data.py
@Time    :   2025/04/03 10:02:16
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Optional

import polars as pl

from digneapy.generators import GenerationResult

type SavingFn = Callable[[pl.DataFrame | pl.LazyFrame, str, bool, ...], None]


def __save_as_csv(
    frames: pl.DataFrame | pl.LazyFrame, filename_pattern: str, lazily: bool
):
    if lazily:
        frames.sink_csv(f"instances_{filename_pattern}.csv")
    elif not lazily and frames.height > 0:
        frames.write_csv(
            f"instances_{filename_pattern}.csv",
        )


def __save_as_parquet(
    frames: pl.DataFrame | pl.LazyFrame,
    filename_pattern: str,
    lazily: bool,
):
    if lazily:
        frames.sink_parquet(
            f"instances_{filename_pattern}.parquet", compression_level=22
        )
    elif not lazily and frames.height > 0:
        frames.write_parquet(
            f"instances_{filename_pattern}.parquet", compression_level=22
        )


__saving_methods: Mapping[str, SavingFn] = {
    "csv": __save_as_csv,
    "parquet": __save_as_parquet,
}


def save_results_to_files(
    filename_pattern: str,
    result: GenerationResult,
    variables_names: Optional[Sequence[str]] = None,
    descriptor_names: Optional[Sequence[str]] = None,
    lazily: bool = False,
    only_instances: bool = False,
    files_format: Literal["csv", "parquet"] = "parquet",
):
    """Saves the results of the generation to CSV files.
    Args:
        filename_pattern (str): Pattern for the filenames.
        result (GenResult): Result of the generation.
        variables_names (Sequence[str]): Names of the variables.
        lazily (bool, optional): Whether the instances inside the result object
            should be collected as a LazyFrame or a DataFrame. Defaults ot False.
        only_instances (bool): Generate only the files with the resulting instances.
            Default True. If False, it would generate an history and arhice_metrics files.
        files_format (Literal[str] = "csv", "parquet"): Format to store the resulting instances file. Parquet is the most efficient for large datasets.
    """
    if files_format not in ("csv", "parquet"):
        warnings.warn(
            f"Unrecognised file format: {files_format}. Selecting parquet as fallback.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        files_format = "parquet"
    _saving_fn = __saving_methods[files_format]
    if len(result.instances) != 0:
        frames = pl.concat(
            [
                instance.to_df(
                    variables_names=variables_names,
                    descriptor_names=descriptor_names,
                    portfolio_names=result.solvers,
                    lazy=lazily,
                )
                for instance in result.instances
            ],
            how="vertical_relaxed",
        )
        _saving_fn(frames=frames, filename_pattern=filename_pattern, lazily=lazily)

        if not only_instances:
            result.history.to_df().write_csv(f"{filename_pattern}_history.csv")
            if result.metrics is not None:
                result.metrics.write_csv(f"{filename_pattern}_archive_metrics.csv")
    else:
        warnings.warn(
            "Archive in Generation result is empty. Nothing to do in save_results_to_files.",
            category=RuntimeWarning,
            stacklevel=2,
        )
