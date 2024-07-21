from setuptools import setup, find_packages

setup(
    name="xymodel",
    version="1.0",
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
