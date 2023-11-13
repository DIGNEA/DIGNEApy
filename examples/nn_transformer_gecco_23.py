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

from digneapy.nn_novelty_search import HyperCMA


def main():
    dimension = 10
    cma_es = HyperCMA(
        dimension=dimension,
        lambda_=5 * dimension,
        generations=100,
        direction="minimise",
    )
    r = cma_es()
    print(r)


if __name__ == "__main__":
    main()
