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

from typing import Optional
from collections.abc import Sequence
from digneapy.generators import GenResult
import pandas as pd


def save_results_to_files(
    filename_pattern: str,
    result: GenResult,
    solvers_names: Optional[Sequence[str]],
    features_names: Optional[Sequence[str]],
    vars_names: Optional[Sequence[str]],
):
    """Saves the results of the generation to CSV files.
    Args:
        filename_pattern (str): Pattern for the filenames.
        result (GenResult): Result of the generation.
        solvers_names (Sequence[str]): Names of the solvers.
        features_names (Sequence[str]): Names of the features.
        vars_names (Sequence[str]): Names of the variables.
    """
    df = pd.DataFrame(
        list(
            i.to_series(
                variables_names=vars_names,
                features_names=features_names,
                score_names=solvers_names,
            )
            for i in result.instances
        )
    )
    df.insert(0, "target", result.target)
    df.to_csv(f"{filename_pattern}_instances.csv", index=False)
    result.history.to_df().to_csv(f"{filename_pattern}_history.csv", index=False)
    if result.metrics is not None:
        result.metrics.to_csv(f"{filename_pattern}_archive_metrics.csv")
