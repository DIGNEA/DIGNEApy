#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   tuning.py
@Time    :   2025/05/07 08:51:16
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from functools import partial
from multiprocessing.pool import Pool

import joblib
import numpy as np
import pandas as pd
from irace import Experiment, ParameterSpace, Real, Scenario, irace
from numpy import linalg as nplinalg
from scipy import stats as spstat
from sklearn.pipeline import Pipeline

from digneapy import NS, Archive
from digneapy.domains import BPPDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import best_fit, first_fit, next_fit, worst_fit

BINS = 20
LIMITS = (-10_000, 10_000)


def create_2d_histograms(x, y, x_min, y_min, x_max, y_max, bins):
    gridx = np.linspace(x_min, x_max, bins)
    gridy = np.linspace(y_min, y_max, bins)
    grid, _, _ = np.histogram2d(x, y, bins=[gridx, gridy])
    return grid


def JSD(P, Q):
    P = P / nplinalg.norm(P, ord=1)
    Q = Q / nplinalg.norm(Q, ord=1)
    M = (P + Q) * 0.5
    return (spstat.entropy(P, M) + spstat.entropy(Q, M)) * 0.5


def compute_uniform_dist_score(data, xmin, ymin, xmax, ymax, bins):
    hist = create_2d_histograms(data[:, 0], data[:, 1], xmin, ymin, xmax, ymax, bins)
    hist1D = [0 for _ in range(len(hist) * len(hist))]
    sum = 0
    i = 0
    for line in hist:
        for elt in line:
            sum += elt
            hist1D[i] = elt
            i += 1
    uni_ref_val = 1.0 / float((len(hist) * len(hist)))
    uni_hist = [uni_ref_val for _ in range(len(hist) * len(hist))]
    return 1 - JSD(uni_hist, hist1D)


def generate_instancess(
    portfolio,
    dimension: int,
    pop_size: int,
    generations: int,
    archive_threshold: float,
    ss_threshold: float,
    k: int,
    descriptor: str,
    pipeline: Pipeline,
    verbose,
):
    domain = BPPDomain(
        dimension=120,
        min_i=20,
        max_i=100,
        max_capacity=150,
        capacity_approach="fixed",
    )
    eig = EAGenerator(
        pop_size=pop_size,
        generations=generations,
        domain=domain,
        portfolio=portfolio,
        novelty_approach=NS(Archive(threshold=archive_threshold), k=k),
        solution_set=Archive(threshold=ss_threshold),
        repetitions=1,
        descriptor_strategy=descriptor,
        replacement=generational_replacement,
    )

    result = eig()
    if len(result.instances) == 0:
        return 0.0

    df = pd.DataFrame([pd.Series(i) for i in result.instances])
    # Reduce the dataset and keep only the first two dimensions
    reduced_df = pipeline.transform(df)
    score = compute_uniform_dist_score(
        reduced_df,
        xmin=LIMITS[0],
        ymin=LIMITS[0],
        xmax=LIMITS[1],
        ymax=LIMITS[1],
        bins=BINS,
    )
    print(len(result.instances), score)
    return -1.0 * score


def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    portfolios = [
        [best_fit, first_fit, next_fit, worst_fit],
        [first_fit, best_fit, next_fit, worst_fit],
        [next_fit, best_fit, first_fit, worst_fit],
        [worst_fit, best_fit, first_fit, next_fit],
    ]

    with Pool(4) as pool:
        pipeline = joblib.load("pipeline_bpp_N_120.pkl")
        results = pool.map(
            partial(
                generate_instancess,
                dimension=120,
                pop_size=128,
                generations=1000,
                archive_threshold=experiment.configuration["t_a"],
                ss_threshold=experiment.configuration["t_ss"],
                k=15,
                descriptor="features",
                pipeline=pipeline,
                verbose=False,
            ),
            portfolios,
        )

    pool.close()
    pool.join()
    return np.mean(np.asarray(results))


if __name__ == "__main__":
    parameter_space = ParameterSpace(
        [
            Real("t_a", 0.0, 10.0),
            Real("t_ss", 0.0, 10.0),
        ]
    )

    scenario = Scenario(max_experiments=512, verbose=100, seed=42)
    result = irace(target_runner, parameter_space, scenario, return_df=True)
    print(result)
    result.to_csv("irace_results_bpp_N_120.csv", index=False)
