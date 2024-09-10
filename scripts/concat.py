#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   concat.py
@Time    :   2024/06/20 09:00:08
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
'''

import os

import pandas as pd

PATH = '../knapsack_ae_results/kpae_n_50'

df = pd.DataFrame()
for file in os.listdir(PATH):
    _df = pd.read_csv(os.path.join(PATH, file))
    df = pd.concat([df, _df])

print(df.head())
df.to_csv('kp_ns_AE50_combined.csv', index=False)