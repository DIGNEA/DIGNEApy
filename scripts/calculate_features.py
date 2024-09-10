#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   calculate_features.py
@Time    :   2024/07/18 09:10:34
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
'''

import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd

feat_names = ["max_p", "max_w", "min_p", "min_w", "avg_eff", "mean", "std"]


def calculate_features(X):
    X_np = X.to_numpy()[5:]
    weights = X_np[0::2]
    profits = X_np[1::2]
    
    avg_eff = 0.0
    for p, w in zip(profits, weights):
        if w > 0.0: 
            avg_eff += (p / w)
    avg_eff /= len(profits)   
     
    return (
        #X_np[0],  # Capacity
        np.max(profits),
        np.max(weights),
        np.min(profits),
        np.min(weights),
        avg_eff,
        np.mean(X_np),
        np.std(X_np),
    )

def main():
    df_original = pd.read_csv('../knapsack_ae_results/kp_ns_AE50_combined.csv')
    df_original[feat_names] = 0.0
    print(df_original.head())

    for i, row in df_original.iterrows():
        f_data = calculate_features(row)
        print(f'Row: {i} --> {f_data}')
        df_original.loc[i, feat_names] = f_data

    df_original.to_csv('kp_ns_AE50_combined_with_features.csv', index=False)
if __name__ == "__main__":
    main()
