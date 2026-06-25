#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   parallel_exc.py
@Time    :   2026/06/25 12:18:43
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from pathlib import Path
from typing import Sequence

import numpy as np

from digneapy.archives import UnstructuredArchive
from digneapy.core import DescriptorKey, DescriptorPipeline, Solver
from digneapy.domains import KnapsackDomain
from digneapy.generators import Evolutionary
from digneapy.lab import GenerationExperiment, Run
from digneapy.operators import UCX, BinarySelection, Generational, UMut
from digneapy.solvers import (
    default_kp,
    map_kp,
    miw_kp,
    mpw_kp,
)


def create_job(
    base_dir: Path,
    portfolio: Sequence[Solver],
    number_of_items: np.uint32,
    pop_size: np.uint32,
    generations: np.uint32,
    archive_threshold: float,
    ss_threshold: float,
    k: np.uint32,
    descriptor: DescriptorKey,
    seed: np.random.SeedSequence,
    rep_index: int,
):
    # Seed here is the master seed for this job
    # We need to generate several need seeds for the
    # components of the experiment: domain, EA, operators, etc.
    domain_seed, generator_seed, cx_seed, mut_seed, sel_seed, rep_seed = seed.spawn(6)
    domain = KnapsackDomain(number_of_items=number_of_items, seed=domain_seed)
    generator = Evolutionary(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        archive=UnstructuredArchive(novelty_threshold=archive_threshold, k=k),
        solution_set=UnstructuredArchive(novelty_threshold=ss_threshold, k=1),
        repetitions=1,  # Portfolio of determinist heuristics
        descriptor_pipe=DescriptorPipeline(descriptor),
        cxrate=0.85,
        mutrate=(1.0 / number_of_items),
        selection=BinarySelection(seed=sel_seed),
        crossover=UCX(seed=cx_seed),
        mutation=UMut(seed=mut_seed),
        replacement=Generational(seed=rep_seed),
        seed=generator_seed,
    )
    target_name = portfolio[0].__name__
    return Run(
        run_name=f"evolutionary_{target_name}_repetition_{rep_index}",
        generator=generator,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    # The solvers are fixed to Knapsack Heuristics available
    portfolio = [map_kp, mpw_kp, default_kp, miw_kp]
    repetitions = 1
    # Best practice for achieving reproducible bit streams
    # is to use the default None for the initial entropy,
    # and then use SeedSequence.entropy to log/pickle
    # the entropy for reproducibility:
    # sq1 = np.random.SeedSequence()
    # sq1.entropy
    # sq2 = np.random.SeedSequence(sq1.entropy)
    # np.all(sq1.generate_state(10) == sq2.generate_state(10))
    entropy = np.random.SeedSequence().entropy
    root = np.random.SeedSequence(entropy=entropy)
    repetitions_seeds = root.spawn(repetitions)
    jobs = []
    base_dir = Path(__file__).parent / "my_sample_experiment" / "map"
    for rep in range(repetitions):
        seed = repetitions_seeds[rep]

        generator = create_job(
            base_dir=base_dir,
            portfolio=portfolio,
            number_of_items=50,
            pop_size=32,
            generations=1_000,
            archive_threshold=0.001,
            ss_threshold=0.0001,
            k=3,
            descriptor="features",
            seed=seed,
            rep_index=rep,
        )
        jobs.append(generator)

    experiment = GenerationExperiment(
        experiment_name="Sample Knapsack Experiment",
        base_dir=base_dir,
        runs_to_do=jobs,
        max_workers=4,
    )
    results = experiment()
    print(results)
