#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   experiments.py
@Time    :   2025/04/02 17:46:02
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bin_packing_novelty_search.py
@Time    :   2025/04/02 15:40:48
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from multiprocessing.pool import Pool
from dataclasses import dataclass
from collections.abc import Sequence, Callable, Iterable
from digneapy import SupportsSolve, P, GenResult, Generator, Domain


@dataclass
class Experiment:
    """Class to keeping the information of an experiment"""

    filename: str
    portfolio: Sequence[SupportsSolve[P]]
    domain: Domain
    generator: Generator
    n_jobs: int


def run_experiment(exp_config: Experiment) -> GenResult:
    print(
        f"Running {exp_config.generator.__name__} for domain {exp_config.domain.__name__}"
    )
    pool = Pool(exp_config.n_jobs)
    results: GenResult = pool.map(
        exp_config.experiment_fn,
        exp_config.portfolio,
    )
    pool.close()
    pool.join()
    return results
