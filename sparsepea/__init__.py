"""
SparsePEA: Sparse Grid Parameterized Expectations Algorithm

A Python package for solving dynamic stochastic general equilibrium (DSGE) 
models using the Parameterized Expectations Algorithm (PEA) with sparse 
grid interpolation.

Author: Simon Mishricky
Affiliation: Australian National University

Example:
    >>> from sparsepea.models import rbc_jit
    >>> from sparsepea.tools import tools
    >>> 
    >>> # Initialize model
    >>> model = rbc_jit()
    >>> 
    >>> # Set up solver
    >>> solver = tools(model=model, depth=8)
    >>> 
    >>> # Solve
    >>> e_solution, policy, multiplier, status = solver.compute_solution(e_initial)
"""

from .models import rbc_jit, dmp_jit
from .tools import tools

__version__ = "0.1.0"
__author__ = "Simon Mishricky"
__email__ = "simon.mishricky@gmail.com"

__all__ = ['rbc_jit', 'dmp_jit', 'tools']
