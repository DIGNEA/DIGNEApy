#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   heuristics.py
@Time    :   2024/4/11 11:14:36
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from typing import Callable
from digneapy.core import Solution
import numpy as np
from deap import creator, base, tools, algorithms

direction = ("Min", "Max")


def gen_dignea_ind(icls, size: int, min_value, max_value):
    """Auxiliar function to generate individual based on
    the Solution class of digneapy
    """
    chromosome = icls(np.random.randint(low=min_value, high=max_value, size=size))
    return chromosome


def ea_mu_comma_lambda(
    dir: str,
    dim: int,
    min_g: int | float,
    max_g: int | float,
    problem: Callable = None,
    cx=tools.cxOnePoint,
    mut=tools.mutUniformInt,
    pop_size: int = 10,
    lambd: int = 100,
    cxpb: float = 0.6,
    mutpb: float = 0.3,
    generations: int = 500,
):
    """Evolutionary Algorithm from DEAP for digneapy

    Args:
        dir (str): Direction of the evolution process. Min (minimisation) or Max (maximisation).
        dim (int): Number of variables of the problem to solve.
        min_g (int | float): Minimum value of the genome of the solutions.
        max_g (int | float): Maximum value of the genome of the solutions.
        problem (Callable): Evaluation function used to calculate the fitness of the individuals.
        pop_size (int, optional): Population size of the evolutionary algorithm. Defaults to 10.
        lambd (int, optional): Number of offspring produced at each generation of the algorithm. Defaults to 100.
        cxpb (float, optional): Crossover probability. Defaults to 0.6.
        mutpb (float, optional): Mutation probability. Defaults to 0.3.
        generations (int, optional): Number of generations to perform. Defaults to 500.

    Returns:
        Population (list): Final population of the algorithm
        Logbook (Logbook): Logbook of the DEAP algorithm
        Hof: Hall Of Fame of DEAP. Best solution found.
    """
    if problem is None:
        msg = "No problem found in args of evolutionary_mu_comma_lambda"
        raise AttributeError(msg)

    if dir not in direction:
        print(f"Direction not valid. It must be {direction}")
        raise AttributeError(msg)

    else:
        mult = 1.0
        if dir == "Min":
            mult = -1.0

        creator.create("Fitness", base.Fitness, weights=(mult,))
        creator.create("Individual", list, fitness=creator.Fitness)
        toolbox = base.Toolbox()
        toolbox.register(
            "individual", gen_dignea_ind, creator.Individual, dim, min_g, max_g
        )

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", cx)
        toolbox.register("mutate", mut, low=min_g, up=max_g, indpb=(1.0 / dim))
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("evaluate", problem)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaMuCommaLambda(
            pop,
            toolbox,
            mu=pop_size,
            lambda_=lambd,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=generations,
            stats=stats,
            halloffame=hof,
        )
        # Convert to Solution class
        cast_pop = [
            Solution(
                chromosome=i,
                objectives=(i.fitness.values[0],),
                fitness=i.fitness.values[0],
            )
            for i in pop
        ]
        best = Solution(
            chromosome=hof[0],
            objectives=(hof[0].fitness.values[0],),
            fitness=hof[0].fitness.values[0],
        )
        return cast_pop, logbook, best
