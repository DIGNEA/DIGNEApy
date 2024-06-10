#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   solver.py
@Time    :   2024/06/07 14:10:11
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from digneapy.core.solution import Solution
from digneapy.core.problem import P
from abc import ABC, abstractmethod
from typing import Generic


class Solver(ABC, Generic[P]):
    """Solver is any callable type that receives a OptProblem
    as its argument and returns a tuple with the solution found
    """

    @abstractmethod
    def __call__(self, problem: P, *args, **kwargs) -> list[Solution]:
        """Solves a optimisation problem

        Args:
            problem (OptProblem): Any optimisation problem or callablle that receives a Sequence and returns a Tuple[float]

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            List[Solution]: Returns a sequence of olutions
        """
        msg = "__call__ method not implemented in Solver"
        raise NotImplementedError(msg)
