from setuptools import setup, find_packages

setup(
    name="gemm",
    version="0.1",
    packages=find_packages(exclude=["benchmark"]),
)