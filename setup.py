#!/usr/bin/env python

"""The setup script."""

from glob import glob

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements: list[str] = []

test_requirements = [
    "pytest>=3",
]


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


compile_args = ["-std=c++11", "-fopenmp"]
ext_modules = [
    Extension(
        "pisinger_cpp",
        sorted(glob("digneapy/solvers/_pisinger/src/*.cpp")),
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        extra_compile_args=compile_args,
        language="c++",
    ),
    Extension(
        "parallel_ea",
        sorted(glob("digneapy/solvers/_parallel_ea/src/*.cpp")),
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        extra_compile_args=compile_args,
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
    Extension(
        "digneapy.solvers._kp",
        sources=["digneapy/solvers/_kp.pyx"],
        libraries=["m"],
        compiler_directives={"language_level": "3"},
        include_dirs=[np.get_include()],
    ),
    Extension(
        "digneapy.solvers._tsp_opt",
        sources=["digneapy/solvers/_tsp_opt.pyx"],
        libraries=["m"],
        compiler_directives={"language_level": "3"},
        include_dirs=[np.get_include()],
    ),
]

setup(
    author="Alejandro Marrero",
    author_email="amarrerd@ull.edu.es",
    python_requires=">=3.12",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Typing :: Typed",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    description="Python version of the DIGNEA code for instance generation",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=[
        "dignea",
        "optimization",
        "instance generation",
        "quality-diversity",
        "NS",
    ],
    name="digneapy",
    packages=find_packages(include=["digneapy", "digneapy.*"]),
    platforms=["any"],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/DIGNEA/digneapy",
    version="0.2.5",
    ext_modules=cythonize(ext_modules),
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
