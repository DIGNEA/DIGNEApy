#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   nn_transformer_gecco_23.py
@Time    :   2023/11/10 14:09:41
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""
from collections import deque
from digneapy.transformers import HyperCMA, NN
from digneapy.generator import EIG
from digneapy.solvers.heuristics import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.domains.knapsack import KPDomain
from digneapy.operators.replacement import first_improve_replacement
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def ns_kp_domain_work(transformer: NN):
    kp_domain = KPDomain(dimension=50, capacity_approach="percentage")
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    gen_instances = dict()
    for i in range(len(portfolio)):
        portfolio.rotate(i)
        eig = EIG(
            10,
            10000,
            domain=kp_domain,
            portfolio=portfolio,
            t_a=3,
            t_ss=3,
            k=3,
            repetitions=1,
            descriptor="features",
            replacement=first_improve_replacement,
            transformer=transformer,
        )
        _, solution_set = eig()
        descriptors = [i.features for i in solution_set]
        gen_instances[portfolio[0].__name__] = descriptors
    # Combinar las instancias
    # Calcular el cubrimiento con respecto al espacio de referencia


def main():
    shapes = (11, 5, 2)
    activations = ("relu", "relu", None)
    transformer = NN("nn_transformer_bpp.keras", shape=shapes, activations=activations)

    weights = np.random.random_sample(size=204)
    transformer.update_weights(weights)

    dimension = 204
    nn = NN(
        name="NN_transformer_kp_domain.keras",
        shape=(11, 5, 2),
        activations=("relu", "relu"),
    )
    cma_es = HyperCMA(
        dimension=dimension,
        direction="maximise",
        transformer=nn,
        experiment_work=ns_kp_domain_work,
    )
    best_nn_weights, population, logbook = cma_es()
    nn.update_weights(best_nn_weights)
    nn.save()


if __name__ == "__main__":
    main()
