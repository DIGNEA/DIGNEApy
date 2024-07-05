#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _bpp_heuristics.py
@Time    :   2024/06/18 11:54:17
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from digneapy.core import Solution
from digneapy.domains.bin_packing import BPP


def best_fit(problem: BPP, *args, **kwargs) -> list[Solution]:
    if problem is None:
        raise ValueError("No problem found in best_fit heuristic")
    assignment = np.zeros(len(problem), dtype=int)
    bin_capacities = []
    items = problem._items
    # Starts the algorithm
    # It keeps a list of open bins, which is initially empty.
    for i, item in enumerate(items):
        bin = None
        max_loaded_bin = np.iinfo(np.int64).min
        for j, b_c in enumerate(bin_capacities):
            # Checks that the item fits in the jth bin
            # and its the bin with maximum load
            if item + b_c <= problem._capacity and b_c > max_loaded_bin:
                max_loaded_bin = b_c
                bin = j
        if bin is not None:
            assignment[i] = bin
            bin_capacities[bin] += item
        else:
            # If no such bin is found, open a new bin and put the item
            # into it

            assignment[i] = len(bin_capacities)
            bin_capacities.append(item)

    _fitness = problem.evaluate(assignment)[0]
    return [Solution(chromosome=assignment, objectives=(_fitness,), fitness=_fitness)]


def first_fit(problem: BPP, *args, **kwargs) -> list[Solution]:
    if problem is None:
        raise ValueError("No problem found in first_fit heuristic")
    assignment = np.zeros(len(problem), dtype=int)
    open_bins = []
    items = problem._items

    for i, item in enumerate(items):
        placed = False
        for j, o_b in enumerate(open_bins):
            if o_b + item <= problem._capacity:
                open_bins[j] += item
                assignment[i] = j
                placed = True
                break

        if not placed:
            open_bins.append(item)
            assignment[i] = len(open_bins) - 1

    _fitness = problem.evaluate(assignment)[0]
    return [Solution(chromosome=assignment, objectives=(_fitness,), fitness=_fitness)]


def next_fit(problem: BPP, *args, **kwargs) -> list[Solution]:
    if problem is None:
        raise ValueError("No problem found in next_fit heuristic")

    assignment = np.zeros(len(problem), dtype=int)
    items = problem._items
    bin_counter = 0
    remaining_capacity = problem._capacity
    for i, item in enumerate(items):
        if item <= remaining_capacity:
            remaining_capacity -= item
        else:
            bin_counter += 1
            remaining_capacity = problem._capacity

        assignment[i] = bin_counter

    _fitness = problem.evaluate(assignment)[0]
    return [Solution(chromosome=assignment, objectives=(_fitness,), fitness=_fitness)]


def worst_fit(problem: BPP, *args, **kwargs) -> list[Solution]:
    if problem is None:
        raise ValueError("No problem found in worst_fit heuristic")

    assignment = np.zeros(len(problem), dtype=int)
    bin_capacities: list[int] = []
    items = problem._items
    # Starts the algorithm
    # It keeps a list of open bins, which is initially empty.
    for i, item in enumerate(items):
        bin = None
        min_load = np.iinfo(np.int64).max
        for j, b_c in enumerate(bin_capacities):
            if item + b_c <= problem._capacity:
                if b_c < min_load:
                    min_load = b_c
                    bin = j

        if bin is not None:
            assignment[i] = bin
            bin_capacities[bin] += item
        else:
            # If no such bin is found, open a new bin and put the item
            # into it
            assignment[i] = len(bin_capacities)
            bin_capacities.append(item)

    _fitness = problem.evaluate(assignment)[0]
    return [Solution(chromosome=assignment, objectives=(_fitness,), fitness=_fitness)]
