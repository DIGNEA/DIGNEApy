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

from typing import Sequence
from ._instance import Instance


def qd_score(instances: Sequence[Instance]) -> float:
    """Calculates the Quality Diversity score of a set of instances.

    Args:
        instances (Sequence[Instance]): List of instances to calculate the QD score.

    Returns:
        float: Sum of the fitness of all instances.
    """
    return sum(ind.fitness for ind in instances)


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

    def __init__(self):
        self._scores = []

    @property
    def scores(self):
        return self._scores

    def update(self, instances: Sequence[Instance]) -> float:
        """Updates the scores after each generation of the algorithm

        Args:
            instances (Sequence[Instance]): List of instances to calculate the QD score.

        Returns:
            float: QD score of the generation.
        """
        generation_score = qd_score(instances)
        self._scores.append(generation_score)
        return generation_score

    def auc(self, batch_size: int) -> float:
        """Calculates the Area Under the Curve of the QD score --> Efficiency

        Args:
            batch_size (int): Number of instances evaluated in each generation.

        Returns:
            float: QD score AUC metric.
        """
        return sum(batch_size * score for score in self._scores)
