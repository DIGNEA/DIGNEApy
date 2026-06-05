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
        self._stats_novelty = tools.Statistics(key=attrgetter("novelty"))
        self._stats_perf_bias = tools.Statistics(key=attrgetter("performance_bias"))
        self._stats_fitness = tools.Statistics(key=attrgetter("fitness"))

        self._stats = tools.MultiStatistics(
            novelty=self._stats_novelty,
            performance_bias=self._stats_perf_bias,
            fitness=self._stats_fitness,
        )
        self._stats.register("mean", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)
        self._stats.register("qd_score", np.sum)

    def __call__(self, instances: Sequence[Instance] | Archive) -> Mapping:
        """Compiles the statistics of the population or Archive.

        Args:
            instances (Sequence[Instance] | Archive): Instances to extract metrics.

        Raises:
            ValueError: If instances or archive are empty.
            TypeError: If any object in instances/archive are not an Instance or subclass of Instance.

        Returns:
            Mapping: Dict with the metrics of the instances
        """

        if instances is None or len(instances) == 0:
            raise ValueError(
                f"Trying to calculate the metrics with an empty instances or None. Got: {instances}."
            )
        if any(not isinstance(ind, Instance) for ind in instances):
            raise TypeError(
                f"Instances must be a sequence of Instance objects got: {instances}\n{type(instances[0])}."
            )
        # ignore np.inf values which can occur in early steps
        with np.errstate(invalid="ignore"):
            record = self._stats.compile(instances)
            return record
            """ if as_dataframe:
                _flatten_record = {}
                for key, value in record.items():
                    if isinstance(value, dict):  # Flatten nested dicts
                        for sub_key, sub_value in value.items():
                            _flatten_record[f"{key}_{sub_key}"] = sub_value
                    else:
                        _flatten_record[key] = value
                return pl.from_dict(_flatten_record)
            else:
                return record """


class Logbook(tools.Logbook):
    """Extends Deap.tools.Logbook to include additionaly features such as to_df."""

    def __init__(self):
        super().__init__()
        self._statistics = Statistics()
        self.header = "generation", "novelty", "performance bias", "fitness"
        self._chapters_headers = (
            "min",
            "mean",
            "std",
            "max",
            "qd_score",
        )
        self.chapters["novelty"].header = self._chapters_headers[:-1]
        self.chapters["performance_bias"].header = self._chapters_headers
        self.chapters["fitness"].header = self._chapters_headers

    def update(
        self,
        generation: int,
        instances: Sequence[Instance] | Archive,
        feedback: bool = False,
    ):
        if generation < 0:
            raise ValueError(
                f"generation value {generation} must be greater than zero in Logbook update."
            )
        # Checks for None, empty and not isinstance() delegated to Statistics
        self.record(generation=generation, **self._statistics(instances))
        if feedback:  # pragma: no cover
            print(self.stream)

    def to_df(self) -> pl.DataFrame:
        fitness = pl.from_dicts(self.chapters["fitness"], schema=self._chapters_headers)
        novelty = pl.from_dicts(self.chapters["novelty"], schema=self._chapters_headers)
        performance = pl.from_dicts(
            self.chapters["performance_bias"], schema=self._chapters_headers
        )
        generations = pl.Series("generation", self.select("generation"))

        df = (
            fitness
            .rename({h: f"{h}_fitness" for h in self._chapters_headers})
            .with_columns(generations)
            .join(
                novelty.rename({
                    h: f"{h}_novelty" for h in self._chapters_headers
                }).with_columns(generations),
                on="generation",
            )
            .join(
                performance.rename({
                    h: f"{h}_performance_bias" for h in self._chapters_headers
                }).with_columns(generations),
                on="generation",
            )
        )

        return df.select(["generation", *[c for c in df.columns if c != "generation"]])
