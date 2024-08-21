"""
# HMC Simulation of 2D XY Model

This package provides tools for simulating and analyzing the 2D XY model using Hamiltonian Monte Carlo (HMC) methods.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="xymodel",
    version="2.2.0",
    packages=find_packages(exclude=['tests*']),
    description="Two Dimensional XY Model Simulation using HMC",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="T.Zhao",
    author_email="tl.zhao@outlook.com",
    url="https://github.com/TobiZhao/2DXYmodel_HMC/",
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'argparse',
        'numba',
        'wheel', 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
