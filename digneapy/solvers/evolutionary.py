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

import multiprocessing
from operator import attrgetter

import numpy as np
from deap import algorithms, base, creator, tools

from digneapy._constants import Direction
from digneapy.core import Solution, Solver
from digneapy.core.problem import P
from digneapy.core.solver import SupportsSolve


def _gen_dignea_ind(icls, size: int, min_value, max_value):
    """Auxiliar function to generate individual based on
    the Solution class of digneapy
    """
    chromosome = icls(np.random.randint(low=min_value, high=max_value, size=size))
    return chromosome


class EA(Solver, SupportsSolve[P]):
    """Evolutionary Algorithm from DEAP for digneapy"""

    def __init__(
        self,
        direction: Direction,
        dim: int,
        min_g: int | float,
        max_g: int | float,
        cx=tools.cxUniform,
        mut=tools.mutUniformInt,
        pop_size: int = 10,
        cxpb: float = 0.6,
        mutpb: float = 0.3,
        generations: int = 500,
        n_cores: int = 1,
    ):
        """Creates a new EA instance with the given parameters.
        Args:
            dir (str): Direction of the evolution process. Min (minimisation) or Max (maximisation).
            dim (int): Number of variables of the problem to solve.
            min_g (int | float): Minimum value of the genome of the solutions.
            max_g (int | float): Maximum value of the genome of the solutions.
            pop_size (int, optional): Population size of the evolutionary algorithm. Defaults to 10.
            cxpb (float, optional): Crossover probability. Defaults to 0.6.
            mutpb (float, optional): Mutation probability. Defaults to 0.3.
            generations (int, optional): Number of generations to perform. Defaults to 500.

        Raises:
            TypeError: If direction is not in digneapy.solvers.DIRECTIONS
        """
        if not isinstance(direction, Direction):
            raise TypeError(
                f"Direction not allowed. Please use a value of the class Direction({Direction.values()})"
            )

        self.direction = direction
        self._cx = cx
        self._mut = mut
        self._pop_size = pop_size
        self._cxpb = cxpb
        self._mutpb = mutpb
        self._generations = generations
        self._n_cores = n_cores if n_cores > 1 else 1
        self._toolbox = base.Toolbox()

        if direction == Direction.MINIMISE:
            self._toolbox.register(
                "individual", _gen_dignea_ind, creator.IndMin, dim, min_g, max_g
            )

        else:
            self._toolbox.register(
                "individual", _gen_dignea_ind, creator.IndMax, dim, min_g, max_g
            )

        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )
        self._toolbox.register("mate", cx, indpb=0.5)
        self._toolbox.register("mutate", mut, low=min_g, up=max_g, indpb=(1.0 / dim))
        self._toolbox.register("select", tools.selTournament, tournsize=2)

        self._stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)

        self._logbook = None
        self._best_found = Solution()

        self._name = f"EA_PS_{self._pop_size}_CXPB_{self._cxpb}_MUTPB_{self._mutpb}"
        self.__name__ = self._name

        if self._n_cores > 1:
            self._pool = multiprocessing.Pool(processes=self._n_cores)
            self._toolbox.register("map", self._pool.map)

    def __call__(self, problem: P, *args, **kwargs) -> list[Solution]:
        """Call method of the EA solver. It runs the EA to solve the OptProblem given.

        Returns:
            Population (list[Solution]): Final population of the algorithm with the best individual found.
        """
        if problem is None:
            msg = "Problem is None in EA.__call__()"
            raise ValueError(msg)

        self._toolbox.register("evaluate", problem)
        # Reset the algorithm
        self._population = self._toolbox.population(n=self._pop_size)
        self._hof = tools.HallOfFame(1)
        self._logbook = None

        self._population, self._logbook = algorithms.eaSimple(
            self._population,
            self._toolbox,
            cxpb=self._cxpb,
            mutpb=self._mutpb,
            ngen=self._generations,
            halloffame=self._hof,
            stats=self._stats,
            verbose=False,
        )

        # Convert to Solution class
        cast_pop = [
            Solution(
                chromosome=i,
                objectives=(i.fitness.values[0],),
                fitness=i.fitness.values[0],
            )
            for i in self._population
        ]
        self._population = cast_pop
        self._best_found = Solution(
            chromosome=self._hof[0],
            objectives=(self._hof[0].fitness.values[0],),
            fitness=self._hof[0].fitness.values[0],
        )
        return [self._best_found, *cast_pop]
