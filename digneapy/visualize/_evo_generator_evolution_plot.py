#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _evo_generator_evolution_plot.py
@Time    :   2024/09/18 11:03:49
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D

from digneapy import Logbook


def ea_generator_evolution_plot(logbook: Logbook, filename: Optional[str | Path] = ""):
    generations = pl.Series("generation", values=logbook.select("gen"))
    avg_s = pl.Series(name=r"$s$", values=logbook.chapters["s"].select("avg"))
    avg_p = pl.Series(name=r"$p$", values=logbook.chapters["p"].select("avg"))

    df = pl.DataFrame([generations, avg_s, avg_p])

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
    else:  # pragma: no cover
        plt.show()
    plt.close()
