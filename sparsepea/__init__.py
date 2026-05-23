"""
SparsePEA: Sparse Grid Parameterized Expectations Algorithm

A Python package for solving dynamic stochastic general equilibrium (DSGE)
models using the Parameterized Expectations Algorithm (PEA) with sparse
grid interpolation.

Author: Simon Mishricky

Example:
    >>> import numpy as np
    >>> from sparsepea.models import rbc_jit
    >>> from sparsepea.tools import tools
    >>>
    >>> model = rbc_jit()
    >>> solver = tools(model=model)
    >>> grid, grid_points = solver.make_states_grid()
    >>>
    >>> # Constant initial guess for the expectation function
    >>> e_initial = np.ones((grid_points.shape[0], 1)) * 0.5
    >>> e_solution, policy, multiplier, status = solver.compute_solution(e_initial)
"""

from .models import rbc_jit, dmp_jit, end_dmp_jit
from .tools import tools

__version__ = "0.1.0"
__author__ = "Simon Mishricky"
__email__ = "simon.mishricky@gmail.com"

__all__ = ['rbc_jit', 'dmp_jit', 'end_dmp_jit', 'tools']
