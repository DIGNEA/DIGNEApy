#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   concat.py
@Time    :   2024/06/20 09:00:08
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import os

import pandas as pd

PATH = "."


def read_files(pattern: str, path: str = PATH):
    for file in os.listdir(PATH):
        if file.startswith(pattern):
            print(f"Reading file: {file}")
            yield pd.read_csv(os.path.join(PATH, file))


if __name__ == "__main__":
    for size in (100,):
        pattern = f"instances_nsf_N_{size}_"
        df = pd.concat(list(read_files(pattern=pattern)))
        df.to_csv(f"kp_nsf_N_{size}_merged.csv", index=False)
