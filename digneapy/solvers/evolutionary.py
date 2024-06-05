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

from typing import Callable, List
from digneapy.core import Solution, Solver, OptProblem
import numpy as np
from deap import creator, base, tools, algorithms
from digneapy.solvers import DIRECTIONS, MINIMISE, MAXIMISE
import multiprocessing
import par_ea_kp
from digneapy.domains.knapsack import Knapsack

def gen_dignea_ind(icls, size: int, min_value, max_value):
    """Auxiliar function to generate individual based on
    the Solution class of digneapy
    """
    chromosome = icls(np.random.randint(low=min_value, high=max_value, size=size))
    return chromosome


class EA(Solver):
    """Evolutionary Algorithm from DEAP for digneapy"""

    def __init__(
        self,
        dir: str,
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
            AttributeError: If direction is not in digneapy.solvers.DIRECTIONS
        """
        if dir not in DIRECTIONS:
            raise AttributeError("Direction not allowed")

        self.direction = dir
        self._cx = cx
        self._mut = mut
        self._pop_size = pop_size
        self._cxpb = cxpb
        self._mutpb = mutpb
        self._generations = generations
        self._n_cores = n_cores if n_cores >= 1 else 1

        mult = 1.0
        if dir == MINIMISE:
            mult = -1.0

        creator.create("Fitness", base.Fitness, weights=(mult,))
        creator.create("Individual", list, fitness=creator.Fitness)

        self._toolbox = base.Toolbox()
        self._toolbox.register(
            "individual", gen_dignea_ind, creator.Individual, dim, min_g, max_g
        )

        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )
        self._toolbox.register("mate", cx, indpb=0.5)
        self._toolbox.register("mutate", mut, low=min_g, up=max_g, indpb=(1.0 / dim))
        self._toolbox.register("select", tools.selTournament, tournsize=2)

        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)

        self._logbook = None
        self._best_found = None
        self._name = f"EA_PS_{self._pop_size}_CXPB_{self._cxpb}_MUTPB_{self._mutpb}"
        self.__name__ = self._name
        if self._n_cores > 1:
            self._pool = multiprocessing.Pool(processes=self._n_cores)
            self._toolbox.register("map", self._pool.map)

    def __call__(self, problem: OptProblem) -> List[Solution]:
        """Call method of the EA solver. It runs the EA to solve the OptProblem given.

        Returns:
            Population (list): Final population of the algorithm with the best individual found.
        """
        if problem is None:
            msg = "Problem is None in EA.__call__()"
            raise AttributeError(msg)

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
        return [self._best_found]

class ParEAKP(par_ea_kp.ParEAKP):
    def __init__(self, pop_size: int = 32, generations: int = 1000, mutpb: float = 0.2, cxpb: float = 0.7, cores: int = 1):
        super().__init__(pop_size, generations, mutpb, cxpb, cores)
        self._pop_size = pop_size
        self._generations = generations
        self._mutpb = mutpb
        self._cxpb = cxpb
        self._n_cores = cores
        self.__name__ = f"ParEAKP_PS_{self._pop_size}_CXPB_{self._cxpb}_MUTPB_{self._mutpb}"

    def __call__(self, kp: Knapsack = None):
        if kp is None:
            msg = "Knapsack Problem is None in ParEAKP.__call__()"
            raise AttributeError(msg)
        x, fitness = self.run(len(kp), kp.weights, kp.profits, kp.capacity)
        return [Solution(chromosome=x, objectives=(fitness,), fitness=fitness)]