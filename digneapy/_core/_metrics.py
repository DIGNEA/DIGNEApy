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


def qd_score(instances_fitness: Sequence[float]) -> float:
    """Calculates the Quality Diversity score of a set of instances fitness.

    Args:
        instances (Sequence[float]): List with the fitness of several instances to calculate the QD score.

    Returns:
        float: Sum of the fitness of all instances.
    """
    return np.sum(instances_fitness)


class QDSCoreAUC:
    """
        Quantifying Efficiency in Quality Diversity Optimization
    In quality diversity (QD) optimization, the QD score is a holistic
    metric which sums the objective values of all cells in the archive.
    Since the QD score only measures the performance of a QD algorithm at a single point in time, it fails to reflect algorithm efficiency.
    Two algorithms may have the same QD score even though one
    algorithm achieved that score with fewer evaluations. We propose
    a metric called “QD score AUC” which quantifies this efficiency.
    """

    def __init__(self, batch_size: int):
        if batch_size < 0:
            raise ValueError(
                f"batch_size in {self.__class__.__name__} must be greater than one"
            )

        self._batch_size = batch_size
        self._scores: list[float] = []

    @property
    def scores(self):
        return self._scores

    def __update(self, instance_fitness: Sequence[float]) -> float:
        """Updates the scores after each generation of the algorithm

        Args:
            instances (Sequence[Instance]): List of instances to calculate the QD score.

        Returns:
            float: QD score of the generation.
        """
        generation_score = qd_score(instance_fitness) * self._batch_size
        self._scores.append(generation_score)
        return generation_score

    def auc(self) -> float:
        """Calculates the Area Under the Curve of the QD score --> Efficiency

        Args:
            batch_size (int): Number of instances evaluated in each generation.

        Returns:
            float: QD score AUC metric.
        """
        return sum(self._scores)

    def __call__(self, instance_fitness: Sequence[float]) -> float:
        return self.__update(instance_fitness)


class Logbook:
    def __init__(self, batch_size: int):
        if batch_size < 0:
            raise ValueError("batch_size in Logbook must be greater than zero.")

        self._batch_size = batch_size
        self._stats_s = tools.Statistics(key=attrgetter("s"))
        self._stats_p = tools.Statistics(key=attrgetter("p"))
        self._stats_f = tools.Statistics(key=attrgetter("fitness"))

        self._stats = tools.MultiStatistics(
            s=self._stats_s, p=self._stats_p, fitness=self._stats_f
        )
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)
        self._stats.register("QD score", np.sum)
        self._stats.register("QD score AUC", QDSCoreAUC(self._batch_size))

        self._logbook = tools.Logbook()
        self._logbook.header = "gen", "fitness", "s", "p"
        self._headers = (
            "min",
            "avg",
            "std",
            "max",
            "QD score",
            "QD score AUC",
        )
        self._logbook.chapters["fitness"].header = self._headers
        self._logbook.chapters["s"].header = self._headers
        self._logbook.chapters["p"].header = self._headers

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
        if len(population) == 0:
            raise ValueError(
                "Error: Trying to calculate the metrics with an empty population"
            )

        record = self._stats.compile(population)
        self._logbook.record(gen=generation, **record)
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
