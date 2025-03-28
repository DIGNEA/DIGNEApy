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

import itertools
from collections.abc import Iterable

import dask.dataframe as dd
import numpy as np

feat_names = ["capacity", "max_p", "max_w", "min_p", "min_w", "avg_eff", "mean", "std"]


def calculate_features(X):
    X_np = X.to_numpy()
    # idx = X_np.nonzero()[0][-1]
    # X_np = X_np[: idx + 1]

    weights = X_np[0::2]
    profits = X_np[1::2]

    avg_eff = 0.0
    for p, w in zip(profits, weights):
        if w > 0.0:
            avg_eff += p / w
    avg_eff /= len(profits)

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


def calculate_len_sets(sets):
    return [len(s) for s in sets]


def calculate_overlap(method_a: tuple[str, Iterable], method_b: tuple[str, Iterable]):
    name_a, data_a = method_a
    name_b, data_b = method_b
    difference_ab = [e - n for e, n in zip(data_a, data_b)]
    difference_ba = [e - n for e, n in zip(data_b, data_a)]
    intersection = [e & n for e, n in zip(data_a, data_b)]
    overlap = calculate_len_sets(intersection)
    print(
        f"Difference {name_a} - {name_b} -> {calculate_len_sets(difference_ab)} :\n{difference_ab}"
    )
    print(
        f"Difference {name_b} - {name_a} -> {calculate_len_sets(difference_ba)} :\n{difference_ba}"
    )
    print(
        f"{name_a} & {name_b} -> {calculate_len_sets(intersection)} :\n{intersection}"
    )
    total_overlap = sum(overlap)
    for f, o_i, diff_ab, diff_ba in zip(
        feat_names, overlap, difference_ab, difference_ba
    ):
        print(f"Overlap {name_a} & {name_b} for {f} = {o_i} cells")
        print(f"Difference {name_a} - {name_b} for {f} = {diff_ab} cells")
        print(f"Difference {name_b} - {name_a} for {f} = {diff_ba} cells")

    print("=" * 50)
    print(f"The total overlap between methods is {total_overlap}/160 cells")


def calculate_coverage():
    features_ranges = [
        (0, 4.24424000e05),
        (7.84000000e02, 9.99000000e02),
        (1.27190000e04, 4.25053000e05),
        (0.0, 2.41000000e02),
        (0.0, 2.62000000e02),
        (5.11066068e-01, 5.05759291e01),
        (4.70225033e02, 8.66089109e02),
        (1.24641283e03, 9.47888253e03),
    ]
    filenames = [
        "/home/amarrero/DIGNEApy/knapsack_ae_results/kp_ns_2D_ae_combined.csv",
        "/home/amarrero/DIGNEApy/knapsack_ae_results/kp_ns_8D_ae_combined.csv",
        "/home/amarrero/DIGNEApy/knapsack_ae_results/kp_nsf_8D_combined.csv",
    ]
    coverage_data = []
    for file in filenames:
        df = dd.read_csv(file)
        df = df.iloc[:, 4:]
        df = df.fillna(0)
        hypercube = [np.linspace(start, stop, 20) for start, stop in features_ranges]
        coverage = [set() for _ in range(len(feat_names))]

        for _, row in df.iterrows():  # Calculate features of each instance
            f_row = calculate_features(row)
            for i, f_ik in enumerate(f_row):
                coverage[i].add(np.digitize(f_ik, hypercube[i]))

        f = sum(len(s) for s in coverage)  # Coverage of the whole method

        print(f"File --> {file} filled {f} cells of 160")
        coverage_data.append((file, coverage))

    for method_a, method_b in itertools.combinations(coverage_data, 2):
        calculate_overlap(method_a, method_b)


if __name__ == "__main__":
    calculate_coverage()
