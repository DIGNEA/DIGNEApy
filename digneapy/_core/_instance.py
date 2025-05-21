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
import numpy.typing as npt
import polars as pl
import pandas as pd


class Instance:
    __slots__ = ("_vars", "_fit", "_p", "_s", "_features", "_desc", "_pscores")
    def __init__(
        self,
        variables: Optional[npt.ArrayLike] = None,
        fitness: float = 0.0,
        p: float = 0.0,
        s: float = 0.0,
        features: Optional[tuple[float]] = None,
        descriptor: Optional[tuple[float]] = None,
        portfolio_scores: Optional[tuple[float]] = None,
    ):
        """Creates an instance of a Instance (unstructured) for QD algorithms
        This class is used to represent a solution in a QD algorithm. It contains the
        variables, fitness, performance, novelty, features, descriptor and portfolio scores
        of the solution.
        The variables are stored as a numpy array, and the fitness, performance and novelty
        are stored as floats. The features, descriptor and portfolio scores are stored as
        numpy arrays.
        
        Args:
            variables (Optional[npt.ArrayLike], optional): Variables or genome of the instance. Defaults to None.
            fitness (float, optional): Fitness of the instance. Defaults to 0.0.
            p (float, optional): Performance score. Defaults to 0.0.
            s (float, optional): Novelty score. Defaults to 0.0.
            features (Optional[tuple[float]], optional): Tuple of features extracted from the domain. Defaults to None.
            descriptor (Optional[tuple[float]], optional): Tuple with the descriptor information of the instance. Defaults to None.
            portfolio_scores (Optional[tuple[float]], optional): Scores of the solvers in the portfolio. Defaults to None.

        Raises:
            ValueError: If fitness, p or s are not convertible to float.
        """
        try:
            fitness = float(fitness)
            p = float(p)
            s = float(s)
        except ValueError:
            raise ValueError(
                "The fitness, p and s parameters must be convertible to float"
            )

        self._vars = np.array(variables) if variables is not None else np.empty(0)
        self._fit = fitness
        self._p = p
        self._s = s
        self._features = np.array(features) if features is not None else np.empty(0)
        self._pscores = (
            np.array(portfolio_scores) if portfolio_scores is not None else np.empty(0)
        )
        self._desc = np.array(descriptor) if descriptor is not None else np.empty(0)

    def clone(self) -> Self:
        """Create a clone of the current instance. More efficient than using copy.deepcopy.

        Returns:
            Self: Instance object
        """
        return Instance(
            variables=list(self._vars),
            fitness=self._fit,
            p=self._p,
            s=self._s,
            features=tuple(self._features),
            portfolio_scores=tuple(self._pscores),
            descriptor=tuple(self._desc),
        )

    @property
    def variables(self):
        return self._vars

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, performance: float):
        try:
            performance = float(performance)
        except ValueError:
            # if performance != 0.0 and not float(performance):
            msg = f"The performance value {performance} is not a float in 'p' setter of class {self.__class__.__name__}"
            raise ValueError(msg)
        self._p = performance

    @property
    def s(self) -> float:
        return self._s

    @s.setter
    def s(self, novelty: float):
        try:
            novelty = float(novelty)
        except ValueError:
            # if novelty != 0.0 and not float(novelty):
            msg = f"The novelty value {novelty} is not a float in 's' setter of class {self.__class__.__name__}"
            raise ValueError(msg)
        self._s = novelty

    @property
    def fitness(self) -> float:
        return self._fit

    @fitness.setter
    def fitness(self, f: float):
        try:
            f = float(f)
        except ValueError:
            # if f != 0.0 and not float(f):
            msg = f"The fitness value {f} is not a float in fitness setter of class {self.__class__.__name__}"
            raise ValueError(msg)

        self._fit = f

    @property
    def features(self) -> np.ndarray:
        return self._features

    @features.setter
    def features(self, features: npt.ArrayLike):
        self._features = np.asarray(features)

    @property
    def descriptor(self) -> np.ndarray:
        return self._desc

    @descriptor.setter
    def descriptor(self, desc: npt.ArrayLike):
        self._desc = np.array(desc)

    @property
    def portfolio_scores(self):
        return self._pscores

    @portfolio_scores.setter
    def portfolio_scores(self, p: npt.ArrayLike):
        self._pscores = np.asarray(p)

    def __repr__(self):
        return f"Instance<f={self.fitness},p={self.p},s={self.s},vars={len(self._vars)},features={len(self.features)},descriptor={len(self.descriptor)},performance={len(self.portfolio_scores)}>"

    def __str__(self):
        import reprlib

        descriptor = reprlib.repr(self.descriptor)
        performance = reprlib.repr(self.portfolio_scores)
        performance = performance[performance.find("(") : performance.rfind(")") + 1]
        return f"Instance(f={self.fitness},p={self.p},s={self.s},features={len(self.features)},descriptor={descriptor},performance={performance})"

    def __iter__(self):
        return iter(self._vars)

    def __len__(self):
        return len(self._vars)

    def __getitem__(self, key):
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self._vars[key])
        index = operator.index(key)
        return self._vars[index]

    def __setitem__(self, key, value):
        self._vars[key] = value

    def __eq__(self, other):
        if not isinstance(other, Instance):
            print(
                f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            )
            return NotImplemented

        else:
            try:
                return all(a == b for a, b in zip(self, other, strict=True))
            except ValueError:
                return False

    def __gt__(self, other):
        if not isinstance(other, Instance):
            print(
                f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            )
            return NotImplemented

        return self.fitness > other.fitness

    def __ge__(self, other):
        if not isinstance(other, Instance):
            print(
                f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            )
            return NotImplemented

        return self.fitness >= other.fitness

    def __hash__(self):
        from functools import reduce

        hashes = (hash(x) for x in self)
        return reduce(operator.or_, hashes, 0)

    def __bool__(self):
        return self._vars.size != 0

    def __format__(self, fmt_spec=""):
        if fmt_spec.endswith("p"):
            # We are showing only the performances
            fmt_spec = fmt_spec[:-1]
            components = self.portfolio_scores
        else:
            fmt_spec = fmt_spec[:-1]
            components = self.descriptor

        components = (format(c, fmt_spec) for c in components)
        decriptor = "descriptor=({})".format(",".join(components))
        msg = f"Instance(f={self.fitness},p={format(self.p, fmt_spec)}, s={format(self.s, fmt_spec)}, {decriptor})"

        return msg

    def asdict(
        self,
        variables_names: Optional[Sequence[str]] = None,
        features_names: Optional[Sequence[str]] = None,
        score_names: Optional[Sequence[str]] = None,
    ) -> dict:
        """Convert the instance to a dictionary. The keys are the names of the attributes
        and the values are the values of the attributes. 

        Args:
            variables_names (Optional[Sequence[str]], optional): Names of the variables in the dictionary, otherwise v_i. Defaults to None.
            features_names (Optional[Sequence[str]], optional): Name of the features in the dictionary, otherwise f_i. Defaults to None.
            score_names (Optional[Sequence[str]], optional): Name of the solvers, otherwise solver_i. Defaults to None.

        Returns:
            dict: Dictionary with the attributes of the instance as keys and the values of the attributes as values.
        """
        sckeys = (
            [f"solver_{i}" for i in range(len(self._pscores))]
            if score_names is None
            else score_names
        )
        _data = {
            "fitness": self._fit,
            "s": self._s,
            "p": self._p,
            "portfolio_scores": {sk: v for sk, v in zip(sckeys, self._pscores)},
        }

        if len(self._desc) not in (
            len(self._vars),
            len(self._features),
            len(self._pscores),
        ):  # Transformed descriptor
            _data["descriptor"] = {f"d{i}": v for i, v in enumerate(self._desc)}
        if len(self.features) != 0:
            f_keys = (
                [f"f{i}" for i in range(len(self._features))]
                if features_names is None or len(features_names) == 0
                else features_names
            )
            _data["features"] = {fk: v for fk, v in zip(f_keys, self._features)}

        if variables_names:
            if len(variables_names) != len(self._vars):
                print(
                    f"Error in asdict(). len(variables_names) = {len(variables_names)} != len(variables) ({len(self._vars)}). Fallback to v#"
                )
                _data["variables"] = {f"v{i}": v for i, v in enumerate(self._vars)}
            else:
                _data["variables"] = {
                    vk: v for vk, v in zip(variables_names, self._vars)
                }

        else:
            _data["variables"] = {f"v{i}": v for i, v in enumerate(self._vars)}

        return _data

    def to_json(self) -> str:
        """Convert the instance to a JSON string. The keys are the names of the attributes
        and the values are the values of the attributes.

        Returns:
            str: JSON string with the attributes of the instance as keys and the values of the attributes as values.
        """
        import json

        return json.dumps(self.asdict(), sort_keys=True, indent=4)

    def to_series(
        self,
        variables_names: Optional[Sequence[str]] = None,
        features_names: Optional[Sequence[str]] = None,
        score_names: Optional[Sequence[str]] = None,
    ) -> pd.Series:
        """Creates a pandas Series from the instance.

        Args:
            variables_names (Optional[Sequence[str]], optional): Names of the variables in the dictionary, otherwise v_i. Defaults to None.
            features_names (Optional[Sequence[str]], optional): Name of the features in the dictionary, otherwise f_i. Defaults to None.
            score_names (Optional[Sequence[str]], optional): Name of the solvers, otherwise solver_i. Defaults to None.

        Returns:
            pd.Series: Pandas Series with the attributes of the instance as keys and the values of the attributes as values.
        """
        _flatten_data = {}
        for key, value in self.asdict(
            variables_names=variables_names,
            features_names=features_names,
            score_names=score_names,
        ).items():
            if isinstance(value, dict):  # Flatten nested dicts
                for sub_key, sub_value in value.items():
                    _flatten_data[f"{sub_key}"] = sub_value
            else:
                _flatten_data[key] = value
        return pd.Series(_flatten_data)
