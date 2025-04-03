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


def save_instances(
    filename: str,
    result: GenResult,
    solvers_names: Optional[Sequence[str]],
    features_names: Optional[Sequence[str]],
    vars_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Saves the results from a Generator into a pd.DataFrame and writes it to a CSV file.

    Args:
        filename (str): filename to store the results.
        result (GenResult): Results from a Generator object.
        solvers_names (Optional[Sequence[str]]): Optionally you can provide the names of the solvers used in the experiment. Default to solver_i.
        features_names (Optional[Sequence[str]]): Optionally you can provide the names of the features of the instances if used. Default to fi.
        vars_names (Optional[Sequence[str]]): Optionally you can provide custom names for the variables of the instances. Default to vi.

    Returns:
        pd.DataFrame: _description_
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
    df.to_csv(filename, index=False)
    return df
