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

import pandas as pd

from digneapy.generators import GenResult


def save_results_to_files(
    filename_pattern: str,
    result: GenResult,
    solvers_names: Optional[Sequence[str]],
    features_names: Optional[Sequence[str]],
    vars_names: Optional[Sequence[str]],
    files_format: Literal["csv", "parquet"] = "parquet",
):
    """Saves the results of the generation to CSV files.
    Args:
        filename_pattern (str): Pattern for the filenames.
        result (GenResult): Result of the generation.
        solvers_names (Sequence[str]): Names of the solvers.
        features_names (Sequence[str]): Names of the features.
        vars_names (Sequence[str]): Names of the variables.
        files_format (Literal[str] = "csv" or "parquet"): Format to store the resulting instances file.
            Parquet is the most efficient for large datasets.
    """
    if files_format not in ("csv", "parquet"):
        print(f"Unrecognised file format: {files_format}. Selecting parquet.")
        files_format = "parquet"
    df = pd.DataFrame(
        [
            i.to_series(
                variables_names=vars_names,
                features_names=features_names,
                score_names=solvers_names,
            )
            for i in result.instances
        ]
    )
    if not df.empty:
        df.insert(0, "target", result.target)
        print(df.head())
    if files_format == "csv":
        df.to_csv(f"{filename_pattern}_instances.csv", index=False)
    elif files_format == "parquet":
        df.to_parquet(f"{filename_pattern}_instances.parquet", index=False)

    result.history.to_df().to_csv(f"{filename_pattern}_history.csv", index=False)
    if result.metrics is not None:
        result.metrics.to_csv(f"{filename_pattern}_archive_metrics.csv")
