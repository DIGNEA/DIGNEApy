#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bin_packing_solvers.py
@Time    :   2024/06/18 14:02:58
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

from digneapy.domains import BPP
from digneapy.solvers import best_fit, first_fit, next_fit, worst_fit

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
        bpp = BPP.from_file(os.path.join(path, file))
        best_f = best_fit(bpp)[0].fitness
        next_f = next_fit(bpp)[0].fitness
        first_f = first_fit(bpp)[0].fitness
        worst_f = worst_fit(bpp)[0].fitness

        results.setdefault("best_fit", []).append(best_f)
        results.setdefault("first_fit", []).append(first_f)
        results.setdefault("next_fit", []).append(next_f)
        results.setdefault("worst_fit", []).append(worst_f)

    df = pd.DataFrame.from_dict(results)
    blank = " " * 80
    print(f"\r{blank}\r", end="")
    print(df.head())
    df.to_csv("bpp_results.csv", index=False)
    plt.figure(figsize=(12, 8))
    axes = sns.boxplot(data=df)
    axes.set_ylabel("Falkenauer Fitness")
    axes.set_xlabel("Solver")
    axes.get_figure().savefig("bpp_results.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="bin_packing_solvers",
        description="Python script to exemplify how to solve BPP instances using digneapy",
    )
    parser.add_argument("path", help="Path where to find the BPP instances to solve.")
    args = parser.parse_args()
    main(args.path)
