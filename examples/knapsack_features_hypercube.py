#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_features_hypercube.py
@Time    :   2024/06/20 14:25:08
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import dask.dataframe as dd
import numpy as np

from digneapy.archives import GridArchive

features = ["capacity", "max_p", "max_w", "min_p", "min_w", "avg_eff", "mean", "std"]
features_dict = {f: [] for f in features}


def calculate_features(X):
    X_np = X.to_numpy()
    idx = X_np.nonzero()[0][-1]
    X_np = X_np[: idx + 1]

    weights = X_np[0::2]
    profits = X_np[1::2]

    avg_eff = sum([p / w for p, w in zip(profits, weights)]) / len(profits)
    return [
        X_np[0],  # Capacity
        np.max(profits),
        np.max(weights),
        np.min(profits),
        np.min(weights),
        avg_eff,
        np.mean(X_np),
        np.std(X_np),
    ]


def calculate_coverage():
    features = [
        "capacity",
        "max_p",
        "max_w",
        "min_p",
        "min_w",
        "avg_eff",
        "mean",
        "std",
    ]
    features_ranges = [
        (3.0, 188190307.0),
        (10.0, 100099.0),
        (300.0, 188190307.0),
        (0.0, 100005.0),
        (0.0, 1000.0),
        (0.024096446510133377, 91415),
        (63.30284857571215, 149746.5024986119),
        (41.07985038925373, 4314244.913937186),
    ]
    # print(archive.n_cells)
    # print(archive.bounds, archive.bounds.shape)
    df = dd.read_csv(
        "/home/amarrero/knapsack_auto_encoder/results/kp_ns_2D_ae_combined.csv"
    )
    df = df.iloc[:, 4:]
    df = df.fillna(0)
    print(df.head())
    hypercube = [np.linspace(start, stop, 20) for start, stop in features_ranges]
    coverage = [set() for _ in range(len(features))]
    print(hypercube)
    for _, row in df.iterrows():
        features = calculate_features(row)
        print(features)
        for i, f_ik in enumerate(features):
            coverage[i].add(np.digitize(f_ik, hypercube[i]))

    f = sum(len(s) for s in coverage)
    print(f"Filled cells --> {f}")
    # NS_AE2D --> 12 cells / 160?


if __name__ == "__main__":
    calculate_coverage()
