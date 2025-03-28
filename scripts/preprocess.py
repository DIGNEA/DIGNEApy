#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   preprocess.py
@Time    :   2024/06/20 09:59:37
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import itertools
import pickle

import pandas as pd
import umap
from keras.utils import pad_sequences
from sklearn.decomposition import PCA

MAX_LENGTH = 2001
instance_cols = list(itertools.chain(*[(f"w_{i}", f"p_{i}") for i in range(1000)]))


def transform_encode(X, scaler, encoder):
    # We pad the data to maximum allowed length
    X_padded = pad_sequences(X, padding="post", dtype="float32", maxlen=MAX_LENGTH)
    X_padded_scaled = scaler.transform(X_padded)
    X_encoded = encoder.predict(X_padded_scaled)

    return X_encoded


def preprocess_df():
    # with open('kp_scaler_for_ae_different_N.pkl', 'rb') as f:
    #     scaler = pickle.load(f)

    # encoder = keras.models.load_model('best_kp_ae_bayesian_latent_dim_2D_lr_one_cycle_training_encoder.keras')

    dd_nsf = pd.read_csv("kp_ns_2D_ae_combined.csv")
    dd_nsf = dd_nsf.fillna(0)
    dd_nsf = dd_nsf[["target", "N", "x_0", "x_1"]]

    # df_data = dd_nsf[['capacity', *instance_cols]].to_numpy()
    # dd_nsf_encoded = transform_encode(df_data, scaler, encoder)
    # dd_nsf_encoded = pd.DataFrame(dd_nsf_encoded, columns=['x0', 'x1'],
    #                            index=dd_nsf.index)

    # dd_nsf_encoded['target'] = dd_nsf['target']
    # dd_nsf_encoded['N'] = dd_nsf['N']
    # dd_nsf_encoded['method'] = '$NS_f$'
    # dd_nsf_encoded.to_csv('kp_nsf_8D_combined_encoded_by_AE_2D.csv', index=False)
    dd_nsf["method"] = "$NS_{AE2D}$"
    dd_nsf["x0"] = dd_nsf["x_0"]
    dd_nsf["x1"] = dd_nsf["x_1"]
    dd_nsf.drop(columns=["x_0", "x_1"], inplace=True)
    dd_nsf.to_csv("kp_ns_2D_ae_combined_encoded_by_AE_2D.csv", index=False)


def transform_encode_others(X, scaler, reducer):
    # We pad the data to maximum allowed length
    X_padded = pad_sequences(X, padding="post", dtype="float32", maxlen=MAX_LENGTH)
    X_padded_scaled = scaler.transform(X_padded)
    X_encoded = reducer.fit_transform(X_padded_scaled)

    return X_encoded


def combined_dfs():
    with open("kp_scaler_for_ae_different_N.pkl", "rb") as f:
        scaler = pickle.load(f)

    df_nsf = pd.read_csv("kp_nsf_8D_combined.csv")
    df_ns2d = pd.read_csv("kp_ns_2D_ae_combined.csv")
    df_ns8d = pd.read_csv("kp_ns_8D_ae_combined.csv")

    df_nsf["method"] = "$NS_f$"
    df_ns2d["method"] = "$NS_{AE2D}$"
    df_ns8d["method"] = "$NS_{AE8D}$"

    df_combined = pd.concat([df_nsf, df_ns2d, df_ns8d])
    df_combined = df_combined.fillna(0)

    del df_nsf
    del df_ns2d
    del df_ns8d

    df_combined = df_combined[["target", "N", "method", "capacity", *instance_cols]]
    pca = PCA(n_components=2)
    umap_reducer = umap.UMAP()
    for encoder, tag in zip([umap_reducer], ["UMAP"]):
        dd_encoded = transform_encode_others(
            df_combined[["capacity", *instance_cols]].to_numpy(), scaler, encoder
        )
        dd_encoded = pd.DataFrame(
            dd_encoded, columns=["x0", "x1"], index=df_combined.index
        )
        dd_encoded["target"] = df_combined["target"]
        dd_encoded["N"] = df_combined["N"]
        dd_encoded["method"] = df_combined["method"]
        dd_encoded.to_csv(f"kp_all_methods_combined_encoded_by_{tag}.csv", index=False)


if __name__ == "__main__":
    # preprocess_df()
    combined_dfs()
