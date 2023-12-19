#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import digneapy


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Alejandro Marrero",
    author_email="amarrerd@ull.edu.es",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="Python version of the DIGNEA code for instance generation",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=["dignea", "optimization", "instance generation"],
    name="digneapy",
    packages=find_packages(include=["digneapy", "digneapy.*"]),
    platforms=["any"],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/dignea/digneapy",
    version=digneapy.__version__,
    zip_safe=False,
)
