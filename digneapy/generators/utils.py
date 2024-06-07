#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/06/07 14:22:35
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from typing import Optional


def plot_generator_logbook(logbook=None, filename: Optional[str] = ""):
    df = pd.DataFrame(logbook.chapters["s"].select("avg"), columns=[r"$s$"])
    df[r"$p$"] = logbook.chapters["p"].select("avg")
    df["Generations"] = logbook.select("gen")

    # Plot configuration
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        data=df,
        x="Generations",
        y=r"$s$",
        marker="o",
        color="blue",
        markersize=5,
        legend=False,
    )

    ax2 = plt.twinx()
    sns.lineplot(
        data=df,
        x="Generations",
        y=r"$p$",
        marker="X",
        markersize=5,
        color="red",
        legend=False,
        ax=ax2,
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="blue", label=r"$s$"),
            Line2D([], [], marker="X", color="red", label=r"$p$"),
        ],
        loc="center right",
    )

    plt.title(r"Evolution of $s$ and $p$")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
