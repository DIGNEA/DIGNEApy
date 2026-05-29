#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _map_elites_evolution_plot.py
@Time    :   2024/09/18 11:04:14
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

from digneapy import Logbook


def map_elites_evolution_plot(logbook: Logbook, filename: Optional[str | Path] = ""):
    generations = pl.Series("generation", values=logbook.select("gen"))
    avg = pl.Series(name="avg", values=logbook.select("avg"))
    max_values = pl.Series(name="max", values=logbook.select("max"))
    min_values = pl.Series(name="min", values=logbook.select("min"))
    df = pl.DataFrame([generations, min_values, avg, max_values])

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
    for key, color in zip(["avg", "min", "max"], ["blue", "green", "red"]):
        sns.lineplot(data=df, x="Generation", y=key, color=color, label=key)
    plt.legend(loc="best")
    plt.title(r"Evolution of fitness in the Map-Elites generator")
    plt.ylabel("Fitness")
    if filename:
        plt.savefig(filename)
    else:  # pragma: no cover
        plt.show()
    plt.close()
