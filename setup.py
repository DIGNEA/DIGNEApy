#!/usr/bin/env python

"""The setup script."""

from setuptools import Extension, find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements: list[str] = []

test_requirements = [
    "pytest>=3",
]


class get_numpy_include(object):
    """Helper class to determine the numpy include path lazily."""

    def __str__(self):
        import numpy

        return numpy.get_include()


compile_args = ["-std=c++11"]

cython_extensions = [
    Extension(
        "digneapy.solvers._kp",
        sources=["digneapy/solvers/_kp.pyx"],
        libraries=["m"],
        include_dirs=[str(get_numpy_include())],
    ),
    Extension(
        "digneapy.solvers._tsp_opt",
        sources=["digneapy/solvers/_tsp_opt.pyx"],
        libraries=["m"],
        include_dirs=[str(get_numpy_include())],
    ),
]


def build_extensions():
    from Cython.Build import cythonize

    cythonized_exts = cythonize(
        cython_extensions, compiler_directives={"language_level": "3"}, force=True
    )

    extensions = cythonized_exts
    # SAFETY CHECK:
    if not extensions:
        raise RuntimeError("No extensions were collected! Build is aborting.")

    return extensions


setup(
    packages=find_packages(include=["digneapy", "digneapy.*"]),
    ext_modules=build_extensions(),
)
