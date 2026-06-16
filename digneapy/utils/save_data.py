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

from collections.abc import Sequence
from typing import Literal, Optional

import polars as pl

from digneapy.generators import GenerationResult


def save_results_to_files(
    filename_pattern: str,
    result: GenerationResult,
    variables_names: Optional[Sequence[str]] = None,
    descriptor_names: Optional[Sequence[str]] = None,
    only_instances: bool = False,
    files_format: Literal["csv", "parquet"] = "parquet",
):
    """Saves the results of the generation to CSV files.
    Args:
        filename_pattern (str): Pattern for the filenames.
        result (GenResult): Result of the generation.
        variables_names (Sequence[str]): Names of the variables.
        only_instances (bool): Generate only the files with the resulting instances. Default True. If False, it would generate an history and arhice_metrics files.
        files_format (Literal[str] = "csv" or "parquet"): Format to store the resulting instances file. Parquet is the most efficient for large datasets.
    """
    if files_format not in ("csv", "parquet"):
        print(f"Unrecognised file format: {files_format}. Selecting parquet.")
        files_format = "parquet"

    df = pl.concat(
        [
            instance.to_df(
                variables_names=variables_names,
                descriptor_names=descriptor_names,
                portfolio_names=result.solvers,
            )
            for instance in result.instances
        ],
        how="vertical_relaxed",
    )
    if df.height > 0:
        if files_format == "csv":
            df.write_csv(
                f"{filename_pattern}_instances.csv",
            )
        elif files_format == "parquet":
            df.write_parquet(
                f"{filename_pattern}_instances.parquet", compression_level=22
            )

    if not only_instances:
        result.history.to_df().write_csv(f"{filename_pattern}_history.csv")
        if result.metrics is not None:
            result.metrics.write_csv(f"{filename_pattern}_archive_metrics.csv")
