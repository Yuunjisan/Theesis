from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="thesis-bayesian-optimization",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    description="Bayesian Optimization with BoTorch and PyTorch",
    author="Ocean",
) 