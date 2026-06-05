#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   instance.py
@Time    :   2024/06/07 14:09:43
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import operator
from typing import Optional, Self, Sequence

import numpy as np
import polars as pl


def _validate_column_names(
    attribute: str,
    names: None | Sequence[str],
    expected_len: int,
    fallback_keyword: str,
) -> Sequence[str]:
    """Validates the names to Instance to_dict

    Args:
        attribute (str): Attribute to extract
        names (None | Sequence[str]): Names of each item to extract.
        expected_len (int): Expected number of items to extract
        fallback_keyword (str): Fallback keyword if names is None

    Raises:
        ValueError: If the given names have a different len that expected
        TypeError: If any given name is not a str

    Returns:
        Sequence[str]: Sequence of names to use in to_dict
    """
    if names is None:
        # If None, fallback to the default keyword with index
        names = tuple(f"{fallback_keyword}{i}" for i in range(expected_len))
    if names is not None and len(names) != expected_len:
        raise ValueError(
            f"Failed to validate names for {attribute}. They must be either None or a sequence of str with {expected_len}. Got: {names} with {len(names)} elements."
        )
    else:
        # names is not None and it has the proper length
        # we still need to check that all items are string
        if not all(type(key) is str for key in names):
            raise TypeError(f"All the names for {attribute} must be str. Got: {names}.")
        else:
            return names


class Instance:
    __slots__ = (
        "_variables",
        "_fitness",
        "_performance_bias",
        "_novelty",
        "_descriptor",
        "_portfolio_scores",
        "_descriptor_dim",
        "_portfolio_dim",
        "_dtype",
    )

    def __init__(
        self,
        variables: Sequence,
        fitness: float | np.float64 = np.float64(0.0),
        performance_bias: float | np.float64 = np.float64(0.0),
        novelty: float | np.float64 = np.float64(0.0),
        descriptor: Optional[Sequence[float | np.float64]] = None,
        portfolio_scores: Optional[Sequence[float | np.float64]] = None,
        dtype=np.uint32,
    ):
        """Creates an instance of a Instance for QD algorithms.


        This class is used to represent a solution in a QD algorithm. It contains the
        variables, fitness, performance bias, novelty, descriptor and portfolio scores
        of the instance.

        The variables, descriptor and portfolio scores are stored as a numpy array.
        The fitness, performance and novelty are stored as np.float64.

        Args:
            variables (Sequence): Variables or genome of the instance.
            fitness (float, optional): Fitness of the instance. Defaults to 0.0.
            performance_bias (float, optional): Performance score. Defaults to 0.0.
            novelty (float, optional): Novelty score. Defaults to 0.0.
            descriptor (Optional[Sequence[float | np.float64], optional): Tuple with the descriptor information of the instance. Defaults to None.
            portfolio_scores (Optional[Sequence[float | np.float64], optional): Scores of the solvers in the portfolio. Defaults to None.

        Raises:
            ValueError: If fitness, performance_bias or novelty are not convertible to float.
        """
        self._dtype = dtype
        try:
            self._fitness = float(fitness)
            self._performance_bias = float(performance_bias)
            self._novelty = float(novelty)
        except Exception:
            raise TypeError(
                "fitness, performance_bias and novelty parameters must be convertible to float."
            )
        # Todo: Consider fixing the dimensions of the descriptor and portfolio
        # if type(descriptor_dim) is not int or descriptor_dim <= 0:
        #     raise ValueError(
        #         f"descriptor_dim must be a positive integer. Got {descriptor_dim}."
        #     )
        # else:
        #     self._descriptor_dim = int(descriptor_dim)

        # if type(portfolio_dim) is not int or portfolio_dim <= 0:
        #     raise ValueError(
        #         f"portfolio_dim must be a positive integer. Got {portfolio_dim}."
        #     )
        # else:
        #     self._portfolio_dim = int(portfolio_dim)
        if variables is None or len(variables) == 0:
            raise ValueError(f"variables has to be a valid sequence. Got: {variables}")
        else:
            self._variables = np.asarray(variables, dtype=self._dtype, copy=True)

        if descriptor is not None and len(descriptor) == 0:
            raise ValueError(
                f"descriptors must be either None or a sequence with at least one value. Got: {descriptor}"
            )
        else:
            if descriptor is None:
                self._descriptor = np.empty(0, dtype=np.float64)
            else:
                self._descriptor = np.asarray(descriptor, dtype=np.float64, copy=True)

        if portfolio_scores is not None and len(portfolio_scores) == 0:
            raise ValueError(
                f"portfolio_scores must be either None or a sequence with at least one value. Got: {portfolio_scores}"
            )
        else:
            if portfolio_scores is None:
                self._portfolio_scores = np.empty((0), dtype=np.float64)
            else:
                self._portfolio_scores = np.asarray(portfolio_scores, dtype=np.float64)

    @property
    def dtype(self):
        return self._dtype

    def clone(self) -> Self:
        """Create a clone of the current instance. More efficient than using copy.deepcopy.

        Returns:
            Self: Instance object
        """
        return type(self)(
            variables=list(self._variables),
            fitness=self._fitness,
            performance_bias=self._performance_bias,
            novelty=self._novelty,
            descriptor=tuple(self._descriptor) if len(self._descriptor) > 0 else None,
            portfolio_scores=tuple(self._portfolio_scores)
            if len(self._portfolio_scores) > 0
            else None,
        )

    def clone_with(self, **overrides):
        """Clones an Instance with overriden attributes

        Returns:
            Instance
        """
        new_object = self.clone()
        for key, value in overrides.items():
            setattr(new_object, key, value)
        return new_object

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, new_variables: np.ndarray):
        if new_variables is None or len(new_variables) == 0:
            raise ValueError(
                f"new_variables cannot be None nor be empty. Got: {new_variables}."
            )

        elif len(new_variables) != len(self._variables):
            raise ValueError(
                "Updating the variables of an Instance object with a different number of values. "
                f"Instance have {len(self._variables)} variables "
                f"and the new_variables sequence have {len(new_variables)}"
            )
        else:
            self._variables = np.asarray(new_variables)

    @property
    def performance_bias(self) -> float | np.float64:
        return self._performance_bias

    @performance_bias.setter
    def performance_bias(self, performance: float | np.float64):
        try:
            self._performance_bias = float(performance)
        except Exception:
            raise ValueError(
                f"performance_bias value is not a float. Got: {performance}."
            )

    @property
    def novelty(self) -> float | np.float64:
        return self._novelty

    @novelty.setter
    def novelty(self, nov_score: float | np.float64):
        try:
            self._novelty = float(nov_score)
        except Exception:
            raise ValueError(f"nov_score value is not a float. Got: {nov_score}.")

    @property
    def fitness(self) -> float | np.float64:
        return self._fitness

    @fitness.setter
    def fitness(self, new_fitness: float | np.float64):
        try:
            self._fitness = float(new_fitness)
        except Exception:
            raise ValueError(f"new_fitness value is not a float. Got: {new_fitness}.")

    @property
    def descriptor(self) -> np.ndarray:
        return self._descriptor

    @descriptor.setter
    def descriptor(self, descriptor: Sequence[float | np.float64] | np.ndarray):
        self._descriptor = np.asarray(descriptor)

    @property
    def portfolio_scores(self):
        return self._portfolio_scores

    @portfolio_scores.setter
    def portfolio_scores(self, scores: Sequence[float | np.float64] | np.ndarray):
        self._portfolio_scores = np.asarray(scores)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        import reprlib

        descriptor = reprlib.repr(self.descriptor)
        performance = reprlib.repr(self.portfolio_scores)
        performance = performance[performance.find("(") : performance.rfind(")") + 1]
        return (
            f"Instance:\n"
            f"   - fitness = {self.fitness}\n"
            f"   - performance_bias = {self.performance_bias}\n"
            f"   - novelty = {self.novelty}\n"
            f"   - descriptor = {descriptor}\n"
            f"   - portfolio scores = {performance}\n"
        )

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __getitem__(self, key: int | slice) -> Self | np.ndarray:
        """Accessor to variables of the Instance

        Args:
            key (index | slice): index or slice to access a subset of variables

        Returns:
            Instance (Self) or np.ndarray: If accessed with a slice, a new Instance is created
            using the subset of variables as the variables of the new Instance. Otherwise, it
            returns variables[i]
        """
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self._variables[key])
        else:
            index = operator.index(key)
            return self._variables[index]

    def __setitem__(self, key: int | slice, value):
        # Todo: Need to update the tests of ths functions
        self._variables[key] = value

    def __eq__(self, other: Self) -> bool:
        """Compares two Instances based on their variables.

        Args:
            other (Self): Another instance to compare.

        Returns:
            bool: Returns True if the two instances have the same number of variables,
            and all of them are equal. Returns False otherwise.
        """
        if not isinstance(other, Instance):
            raise TypeError(
                f"Other of type {other.__class__.__name__} can not be compared with an Instance."
            )

        else:
            try:
                return all(a == b for a, b in zip(self, other, strict=True))
            except ValueError:
                return False

    def __gt__(self, other: Self) -> bool | np.bool:
        """Compares two Instances based on their fitness.

        Args:
            other (Self): Another instance to compare.

        Returns:
            bool: Returns True if the self has a greater fitness
            than the other. Returns False otherwise.
        """
        if not isinstance(other, Instance):
            raise TypeError(
                f"Other of type {other.__class__.__name__} can not be compared with an Instance."
            )
        else:
            return self.fitness > other.fitness

    def __ge__(self, other: Self) -> bool | np.bool:
        """Compares two Instances based on their fitness.

        Args:
            other (Self): Another instance to compare.

        Returns:
            bool: Returns True if the self has a greater or equal fitness
            than the other. Returns False otherwise.
        """
        if not isinstance(other, Instance):
            raise TypeError(
                f"Other of type {other.__class__.__name__} can not be compared with an Instance."
            )
        else:
            return self.fitness >= other.fitness

    def to_dict(
        self,
        variables_names: Optional[Sequence[str]] = None,
        descriptor_names: Optional[Sequence[str]] = None,
        portfolio_names: Optional[Sequence[str]] = None,
    ) -> dict:
        """Convert the instance to a dictionary.

        The keys are the names of the attributes and the values are the values of the attributes.

        Args:
            variables_names   (Optional[Sequence[str]], optional): Names of the variables in the dictionary, otherwise v_i. Defaults to None.
            descriptor_names: (Optional[Sequence[str]], optional): Names of the components of the descriptor, otherwisde di. Default to None.
            portfolio_names   (Optional[Sequence[str]], optional): Name of the solvers, otherwise solver_i. Defaults to None.

        Returns:
            dict: Dictionary with the attributes of the instance as keys and the values of the attributes as values.
        """
        _instance_data = {}

        descriptor_names = _validate_column_names(
            "descriptor", descriptor_names, len(self._descriptor), fallback_keyword="d"
        )
        _instance_data = {
            **{key: value for key, value in zip(descriptor_names, self._descriptor)}
        }

        variables_names = _validate_column_names(
            "variables_names", variables_names, len(self), fallback_keyword="v"
        )
        _instance_data["variables"] = {
            key: value for key, value in zip(variables_names, self._variables)
        }

        portfolio_names = _validate_column_names(
            "portfolio_names",
            portfolio_names,
            len(self.portfolio_scores),
            fallback_keyword="alg",
        )
        _instance_data["portfolio_scores"] = {
            key: value for key, value in zip(portfolio_names, self._portfolio_scores)
        }

        _instance_data = {
            "target": portfolio_names[0],
            "fitness": self._fitness,
            "novelty": self._novelty,
            "performance_bias": self._performance_bias,
            **_instance_data,
        }
        return _instance_data

    def to_json(self) -> str:
        """Convert the instance to a JSON string.

        The keys are the names of the attributes and the values are the values of the attributes.

        Returns:
            str: JSON string with the attributes of the instance as keys and the values of the attributes as values.
        """
        import json

        # Todo: Need to change dtypes because np is not JSON serializable
        return json.dumps(self.to_dict(), sort_keys=False, indent=2)

    def to_df(
        self,
        variables_names: Optional[Sequence[str]] = None,
        descriptor_names: Optional[Sequence[str]] = None,
        portfolio_names: Optional[Sequence[str]] = None,
    ) -> pl.DataFrame:
        """Creates a Polars DataFrame from the instance.

        Args:
            variables_names (Optional[Sequence[str]], optional): Names of the variables in the dictionary, otherwise v_i. Defaults to None.
            descriptor_names: (Optional[Sequence[str]], optional): Names of the components of the descriptor, otherwisde di. Default to None.
            portfolio_names (Optional[Sequence[str]], optional): Name of the solvers, otherwise solver_i. Defaults to None.

        Returns:
            DataFrame: Polars DataFrame with the attributes of the instance as keys and the values of the attributes as values.
        """
        _flatten_data = {}
        for key, value in self.to_dict(
            variables_names=variables_names,
            descriptor_names=descriptor_names,
            portfolio_names=portfolio_names,
        ).items():
            if isinstance(value, dict):  # Flatten nested dicts
                for sub_key, sub_value in value.items():
                    _flatten_data[sub_key] = sub_value
            else:
                _flatten_data[key] = value
        return pl.DataFrame(_flatten_data)
