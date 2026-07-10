#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/06/17 12:14:28
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Mapping, Tuple

import numpy as np
from numpy.random import SeedSequence


def make_seed_sequences(
    root_seed: int | Sequence[int] | SeedSequence, n_repetitions: int, n_solvers: int
) -> Mapping[Tuple[int, int], SeedSequence]:
    """Generates n_repetitions * n_solvers seeds from a master seed

    Args:
        root_seed (int | Sequence[int]): Root seed to spawn the rest. It
            must be an integer or a sequence of integers.
        n_repetitions (int): Number of repetitions the experiment will perform.
        n_solvers (int): Number of solvers in the portfolio.

    Returns:
        Mapping[Tuple[int, int], SeedSequence]: Dictionary with the seeds generated.
            The keys of the dictionary are tuples of (rep, solver) and the values
            are the seeds for that combination.
    """
    if not isinstance(root_seed, SeedSequence):
        root_seed = np.random.SeedSequence(entropy=root_seed)

    children_seeds = root_seed.spawn(n_repetitions * n_solvers)
    return {
        (rep, solv): children_seeds[rep * n_solvers + solv]
        for rep in range(n_repetitions)
        for solv in range(n_solvers)
    }
