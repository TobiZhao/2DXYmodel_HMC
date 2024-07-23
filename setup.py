"""
XY Model Simulation Package

This package provides tools for simulating and analyzing the 2D XY model 
using Hamiltonian Monte Carlo (HMC) methods.
"""

from setuptools import setup, find_packages

setup(
    name="xymodel",
    version="1.1",
    packages=find_packages(),
    description="Two Dimensional XY Model Simulation using HMC",
    author="T.Zhao",
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'argparse',
    ]
)
