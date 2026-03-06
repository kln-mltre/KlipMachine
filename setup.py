"""
Setup file for KlipMachine development installation.
"""

from setuptools import setup, find_packages

setup(
    name="klipmachine",
    version="1.0.0",
    packages=find_packages(),
    py_modules=['config'],
    python_requires=">=3.10",
)