"""Minimal setup.py for SparsePEA."""

from setuptools import setup, find_packages

setup(
    name="sparsepea",
    version="0.1.0",
    author="Simon Mishricky",
    description="Sparse Grid Parameterised Expectations Algorithm for DSGE models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Simon-Mishricky/sparsepea",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "numba>=0.54",
        "matplotlib>=3.3",
        # Tasmanian is the Oak Ridge sparse-grid library. Install it
        # separately — either `brew install tasmanian` (macOS, ships
        # Python bindings) or `pip install Tasmanian` (Linux, builds
        # from source via cmake). It is intentionally NOT listed here
        # because pip cannot resolve a source-only build cleanly inside
        # this metadata.
    ],
    extras_require={
        # Dependencies used only by the replication notebook.
        "notebook": ["pandas>=1.3", "statsmodels>=0.13", "quantecon>=0.5", "jupyter"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
