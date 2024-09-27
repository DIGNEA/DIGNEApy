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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def map_elites_evolution_plot(logbook=None, filename: Optional[str] = ""):
    df = pd.DataFrame(logbook.select("avg"), columns=["avg"])
    df["min"] = logbook.select("min")
    df["max"] = logbook.select("max")
    df["Generation"] = logbook.select("gen")
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
    else:
        plt.show()
