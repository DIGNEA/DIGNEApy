#!/usr/bin/env python

"""The setup script."""

from glob import glob

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
