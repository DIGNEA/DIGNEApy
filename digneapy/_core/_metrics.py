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
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from deap import tools

from ..archives import Archive
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
        self, population: Sequence[Instance] | Archive, as_dataframe: bool = False
    ) -> Mapping | pl.DataFrame:
        """Calculates the statistics of the population.
        Args:
            as_dataframe (Sequence[Instance]): List of instances to calculate the statistics.
            as_series (bool): Whether to build a Polars DataFrame or not. Default False.
        Returns:
            Mapping | pl.DataFrame: Statistics of the population.
        """

        if len(population) == 0:
            raise ValueError(
                "Error: Trying to calculate the metrics with an empty population"
            )
        if any(not isinstance(ind, Instance) for ind in population):
            raise TypeError(
                f"Error: Population must be a sequence of Instance objects got: {population}\n{type(population[0])}"
            )
        with np.errstate(invalid="ignore"):
            record = self._stats.compile(
                population
            )  # ignore np.inf values which can occur in early steps
            if as_dataframe:
                _flatten_record = {}
                for key, value in record.items():
                    if isinstance(value, dict):  # Flatten nested dicts
                        for sub_key, sub_value in value.items():
                            _flatten_record[f"{key}_{sub_key}"] = sub_value
                    else:
                        _flatten_record[key] = value
                return pl.from_dict(_flatten_record)
            else:
                return record


class Logbook(tools.Logbook):
    def __init__(self):
        super().__init__()
        self._statistics = Statistics()
        self.header = "gen", "s", "p", "fitness"
        self._chapters_headers = (
            "min",
            "mean",
            "std",
            "max",
            "qd_score",
        )
        self.chapters["s"].header = self._chapters_headers[:-1]
        self.chapters["p"].header = self._chapters_headers
        self.chapters["fitness"].header = self._chapters_headers

    def update(
        self,
        generation: int,
        population: Sequence[Instance] | Archive,
        feedback: bool = False,
    ):
        if generation < 0:
            raise ValueError(
                f"generation value {generation} must be greater than zero in Logbook.update()"
            )
        self.record(gen=generation, **self._statistics(population, as_dataframe=False))
        if feedback:  # pragma: no cover
            print(self.stream)

    def to_df(self) -> pl.DataFrame:
        fitness = pl.from_dicts(self.chapters["fitness"], schema=self._chapters_headers)
        diversity = pl.from_dicts(self.chapters["s"], schema=self._chapters_headers)
        performance = pl.from_dicts(self.chapters["p"], schema=self._chapters_headers)
        generations = pl.Series("generation", self.select("gen"))
        df = (
            fitness
            .rename({h: f"{h}_fitness" for h in self._chapters_headers})
            .with_columns(generations)
            .join(
                diversity.rename({
                    h: f"{h}_diversity" for h in self._chapters_headers
                }).with_columns(generations),
                on="generation",
            )
            .join(
                performance.rename({
                    h: f"{h}_performance" for h in self._chapters_headers
                }).with_columns(generations),
                on="generation",
            )
        )

        return df.select(["generation", *[c for c in df.columns if c != "generation"]])
