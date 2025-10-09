#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _metrics.py
@Time    :   2025/02/05 14:40:24
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from operator import attrgetter
from typing import Sequence

import numpy as np
import pandas as pd
from deap import tools

from ._instance import Instance


def qd_score(instances_fitness: np.ndarray) -> np.float64:
    """Calculates the Quality Diversity score of a set of instances fitness.

    Args:
        instances (Sequence[float]): List with the fitness of several instances to calculate the QD score.

    Returns:
        float: Sum of the fitness of all instances.
    """
    return np.sum(instances_fitness)


def qd_score_auc(qd_scores: np.ndarray, batch_size: int) -> np.float64:
    """Calculates the Quantifying Efficiency in Quality Diversity Optimization
    In quality diversity (QD) optimization, the QD score is a holistic
    metric which sums the objective values of all cells in the archive.
    Since the QD score only measures the performance of a QD algorithm at a single point in time, it fails to reflect algorithm efficiency.
    Two algorithms may have the same QD score even though one
    algorithm achieved that score with fewer evaluations. We propose
    a metric called “QD score AUC” which quantifies this efficiency.

    Args:
        qd_scores (Sequence[float]): Sequence of QD scores.
        batch_size (int): Number of instances evaluated in each generation.

    Returns:
        np.float64: QD score AUC metric.
    """
    return np.sum(qd_scores) * batch_size


class Statistics:
    def __init__(self):
        self._stats_s = tools.Statistics(key=attrgetter("s"))
        self._stats_p = tools.Statistics(key=attrgetter("p"))
        self._stats_f = tools.Statistics(key=attrgetter("fitness"))

        self._stats = tools.MultiStatistics(
            s=self._stats_s, p=self._stats_p, fitness=self._stats_f
        )
        self._stats.register("mean", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)
        self._stats.register("qd_score", np.sum)

    def __call__(
        self, population: Sequence[Instance], as_series: bool = False
    ) -> dict | pd.Series:
        """Calculates the statistics of the population.
        Args:
            population (Sequence[Instance]): List of instances to calculate the statistics.
        Returns:
            dict: Dictionary with the statistics of the population.
        """
        if len(population) == 0:
            raise ValueError(
                "Error: Trying to calculate the metrics with an empty population"
            )
        if not all(isinstance(ind, Instance) for ind in population):
            raise TypeError("Error: Population must be a sequence of Instance objects")

        record = self._stats.compile(population)
        if as_series:
            _flatten_record = {}
            for key, value in record.items():
                if isinstance(value, dict):  # Flatten nested dicts
                    for sub_key, sub_value in value.items():
                        _flatten_record[f"{key}_{sub_key}"] = sub_value
                else:
                    _flatten_record[key] = value
            return pd.Series(_flatten_record)
        else:
            return record


class Logbook:
    def __init__(self):
        self._statistics = Statistics()
        self._logbook = tools.Logbook()
        self._logbook.header = "gen", "s", "p", "fitness"
        self._headers = (
            "min",
            "mean",
            "std",
            "max",
            "qd_score",
        )
        self._logbook.chapters["s"].header = self._headers[:-1]
        self._logbook.chapters["p"].header = self._headers
        self._logbook.chapters["fitness"].header = self._headers

    def __len__(self):
        return len(self._logbook)

    @property
    def logbook(self):
        return self._logbook

    def update(
        self, generation: int, population: Sequence[Instance], feedback: bool = False
    ):
        if generation < 0:
            raise ValueError(
                f"generation value {generation} must be greater than zero in Logbook.update()"
            )
        self._logbook.record(gen=generation, **self._statistics(population))
        if feedback:
            print(self._logbook.stream)

    def to_df(self):
        df_f = pd.DataFrame(
            self._logbook.chapters["fitness"], columns=["gen", *self._headers]
        )
        df_f.columns = ["gen", *[f"{h}_f" for h in self._headers]]
        df_s = pd.DataFrame(
            self._logbook.chapters["s"], columns=["gen", *self._headers]
        )
        df_s.columns = ["gen", *[f"{h}_s" for h in self._headers]]
        df_p = pd.DataFrame(
            self._logbook.chapters["p"], columns=["gen", *self._headers]
        )
        df_p.columns = ["gen", *[f"{h}_p" for h in self._headers]]
        df = pd.merge(df_f, df_s, on=["gen"])
        df = pd.merge(df, df_p, on=["gen"])

        return df
