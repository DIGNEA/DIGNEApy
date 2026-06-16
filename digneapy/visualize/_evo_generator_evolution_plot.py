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
    mean_novelty = pl.Series(
        name="novelty", values=logbook.chapters["novelty"].select("avg")
    )
    mean_perf_bias = pl.Series(
        name="performance_bias",
        values=logbook.chapters["performance_bias"].select("avg"),
    )

    df = pl.DataFrame([generations, mean_novelty, mean_perf_bias])

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
        x="generation",
        y="novelty",
        marker="o",
        color="blue",
        markersize=5,
        legend=False,
    )

    ax2 = plt.twinx()
    sns.lineplot(
        data=df,
        x="generation",
        y="performance_bias",
        marker="X",
        markersize=5,
        color="red",
        legend=False,
        ax=ax2,
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="blue", label="novelty"),
            Line2D([], [], marker="X", color="red", label="performance bias"),
        ],
        loc="center right",
    )

    plt.title(r"Evolution of novelty and performance bias")
    if filename:
        plt.savefig(filename)
    else:  # pragma: no cover
        plt.show()
    plt.close()
