"""Top-level package for digneapy."""

__author__ = """Alejandro Marrero"""
__email__ = "amarrerd@ull.edu.es"
__version__ = "0.2.0"

from digneapy import core, domains, generators, operators, qd, solvers, utils
from digneapy._constants import Direction

__all__ = [
    "domains",
    "operators",
    "solvers",
    "core",
    "generators",
    "qd",
    "Direction",
    "utils",
]
