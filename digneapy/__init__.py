"""Top-level package for digneapy."""

__author__ = """Alejandro Marrero"""
__email__ = "amarrerd@ull.edu.es"
__version__ = "0.2.0"

from digneapy import domains, operators, solvers, core, transformers, generators, qd

__all__ = [
    "domains",
    "operators",
    "solvers",
    "core",
    "generators",
    "qd",
    "transformers",
]
