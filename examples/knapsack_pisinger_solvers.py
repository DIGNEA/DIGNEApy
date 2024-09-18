#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_pisinger_solvers.py
@Time    :   2024/05/29 10:24:58
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from digneapy.domains import Knapsack
from digneapy.solvers.pisinger import combo, expknap, minknap


def main():
    n = 1000
    c = np.random.randint(1e3, 1e5)
    w = np.random.randint(1000, 5000, size=n, dtype=np.int32)
    p = np.random.randint(1000, 5000, size=n, dtype=np.int32)
    kp = Knapsack(profits=p, weights=w, capacity=c)
    minknap_time = minknap(kp)
    combo_time = combo(kp)
    expknap_time = expknap(kp)
    print(
        f"MinKnap time {minknap_time[0]}\n Combo time: {combo_time[0]}\n ExpKnap time: {expknap_time[0]}\n"
    )


if __name__ == "__main__":
    main()
