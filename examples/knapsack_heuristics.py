#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_heuristics.py
@Time    :   2024/06/19 08:19:47
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from digneapy.domains import Knapsack
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = [
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]
plt.rcParams["font.size"] = 16


def main(path: str):
    results = {}
    for file in os.listdir(path):
        msg = f"\rSolving {file}"
        print(msg, flush=True, end="")
        knapsack = Knapsack.from_file(os.path.join(path, file))
        default_f = default_kp(knapsack)[0].fitness
        map_f = map_kp(knapsack)[0].fitness
        miw_f = miw_kp(knapsack)[0].fitness
        mpw_f = mpw_kp(knapsack)[0].fitness

        results.setdefault("default", []).append(default_f)
        results.setdefault("map", []).append(map_f)
        results.setdefault("miw", []).append(miw_f)
        results.setdefault("mpw", []).append(mpw_f)

    df = pd.DataFrame.from_dict(results)
    blank = " " * 80
    print(f"\r{blank}\r", end="")
    print(df.head())
    df.to_csv("knapsack_results.csv", index=False)
    plt.figure(figsize=(12, 8))
    axes = sns.boxplot(data=df)
    axes.set_ylabel("Profit")
    axes.set_xlabel("Solver")
    axes.get_figure().savefig("knapsack_results.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="knapsack_heuristics",
        description="Python script to exemplify how to solve Knapsack instances using digneapy",
    )
    parser.add_argument(
        "path", help="Path where to find the Knapsack instances to solve."
    )
    args = parser.parse_args()
    main(args.path)
