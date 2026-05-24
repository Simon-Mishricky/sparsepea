"""
Sparse-grid PEA solver and diagnostic tools.

This module provides the :class:`tools` class, which wraps a DSGE model
object and solves for the parameterised expectation function using Tasmanian
sparse grids for interpolation and quadrature.

The solver implements the following workflow:
    1. Construct a sparse grid over the state space (``make_states_grid``)
    2. Construct a quadrature grid for shock integration (``make_shocks_grid``)
    3. Iterate on the PEA fixed-point mapping (``compute_solution``)
    4. Visualise policy functions and Euler residuals (``plot_*`` methods)

Dependencies
------------
- Tasmanian (https://github.com/ORNL/TASMANIAN) for sparse grid construction
- Matplotlib for 3D surface and histogram plots
"""

import numpy as np 
from numba import float64
from numba.experimental import jitclass
import Tasmanian as ts
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import *

class tools:
    """Sparse-grid PEA solver for DSGE models with occasionally binding constraints.
    
    Parameters
    ----------
    model : object
        A JIT-compiled model instance (e.g. ``rbc_jit`` or ``dmp_jit``) 
        implementing the PEA interface: ``x_axis_grid``, ``rhs_euler``,
        ``c_implied``, and ``ar1_conditional_density``.
    states_input : int
        Number of state variables (default 2).
    states_output : int
        Output dimension of the state grid (default 1).
    shocks_input : int
        Number of shock dimensions (default 1).
    shocks_output : int
        Output dimension of the shock grid (default 1).
    depth : int
        Tasmanian sparse grid refinement level (default 8).
    order : int
        Local polynomial order (default 1).
    max_iter : int
        Maximum PEA iterations (default 500).
    tol : float
        Convergence tolerance on the sup-norm (default 1e-5).
    damping : float
        Fixed-point damping factor in (0, 1]. The update is
        ``e := damping·e_new + (1 - damping)·e_old``. Use ``damping=1.0``
        (default) for straight fixed-point iteration. Models with a
        consumption-based stochastic discount factor (e.g. ``end_dmp_jit``)
        typically require ``damping ≈ 0.1`` to avoid blow-up.
    """
    
    def __init__(self,
                 model,                   # Input of a chosen model
                 states_input=2,          # Number of dimensions of the state space
                 states_output=1,         # Dimensions of the state space grid
                 shocks_input=1,          # Number of dimensions of the shocks
                 shocks_output=1,         # Dimensions of the shocks
                 depth=8,                 # Number of sparse grid points
                 order=1,                 # Order of the polynomial for grid
                 max_iter=500,            # Maximum number of iterations to solve e
                 tol=1e-5,                # Tolerance criterion
                 damping=1.0):            # Fixed-point damping (1.0 = undamped)
        
        self.states_input = states_input
        self.states_output = states_output
        self.shocks_input = shocks_input
        self.shocks_output = shocks_output
        self.depth = depth
        self.order = order
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.model = model
        self.state_bounds = np.reshape(model.state_bounds, (2, 2))
        self.shock_bounds = np.atleast_2d(model.shock_bounds)
        self.xy_grid = np.linspace((self.state_bounds[0, 0], self.state_bounds[1, 0]), (self.state_bounds[0, 1], self.state_bounds[1, 1]), 200)
        self.xy_fine = np.linspace((self.state_bounds[0, 0], self.state_bounds[1, 0]), (self.state_bounds[0, 1], self.state_bounds[1, 1]), 1000)

        # Cache grid data that does not change across iterations.
        # The Tasmanian grid object itself is constructed per interpolation
        # call (~0.2ms each) because loadNeededPoints can only be called once
        # per grid lifetime.
        self.state_points = self._build_states_grid().getPoints()
        shocks_grid = self._build_shocks_grid()
        self.shock_points = shocks_grid.getPoints()
        self.quad_weights = np.atleast_2d(shocks_grid.getQuadratureWeights())

    def _build_states_grid(self):
        '''Construct a fresh 2D local-polynomial sparse grid on the state space.'''

        grid = ts.makeLocalPolynomialGrid(self.states_input, self.states_output,
                                          self.depth, self.order, "localp")
        grid.setDomainTransform(self.state_bounds)
        return grid

    def _build_shocks_grid(self):
        '''Construct a fresh 1D local-polynomial sparse grid on the shock space.'''

        grid = ts.makeLocalPolynomialGrid(self.shocks_input, self.shocks_output,
                                          self.depth, self.order, "localp")
        grid.setDomainTransform(self.shock_bounds)
        return grid

    def make_states_grid(self):
        '''Return a fresh state-space sparse grid and its node coordinates.

        Kept for backward compatibility — prefer ``self.state_points`` for
        the coordinates and ``self._build_states_grid()`` for a fresh grid.'''

        grid = self._build_states_grid()
        return grid, grid.getPoints()

    def make_shocks_grid(self):
        '''Return a fresh shock-space sparse grid, its node coordinates, and
        quadrature weights. Kept for backward compatibility.'''

        grid = self._build_shocks_grid()
        return grid, grid.getPoints(), np.atleast_2d(grid.getQuadratureWeights())

    def sparse_interpolate(self, state_p, e):
        '''Interpolate the values ``e`` (one per state-grid node) at the
           query points ``state_p``.'''

        grid = self._build_states_grid()
        grid.loadNeededPoints(e)
        return grid.evaluateBatch(state_p)

    def euler_interpolation(self, x_p_grid, e):
        '''Interpolate the expectation function ``e`` at every combination of
           next-period endogenous state and next-period shock.'''

        shock_points = self.shock_points
        x_state_vals = np.repeat(x_p_grid[:, 0], shock_points[:, 0].size)
        y_state_vals = np.tile(shock_points[:, 0], x_p_grid[:, 0].size)
        states = np.column_stack((x_state_vals, y_state_vals))
        z_interp = self.sparse_interpolate(states, e)
        return z_interp.reshape(x_p_grid.size, shock_points.size, 1)

    def compute_solution(self, e):
        '''Solve for the fixed point of the PEA mapping via damped iteration.'''

        model = self.model
        max_iter, tol, damping = self.max_iter, self.tol, self.damping
        state_points = self.state_points
        shock_points = self.shock_points
        w = self.quad_weights

        st = f"Did not converge in {max_iter} iterations"
        for i in range(max_iter):

            # Solve for the next period x_axis grid
            x_p_grid, function, μ = model.x_axis_grid(e, state_points)

            # Given next period grids interpolate to find e
            ψ_p_grid = self.euler_interpolation(x_p_grid, e)

            # Solve for new e
            e_p = model.rhs_euler(state_points, x_p_grid, shock_points, ψ_p_grid, e, w)

            # Sup-norm distance between successive iterates
            dist = (np.abs(e_p - e)).max()

            # Damped update: e := damping·e_p + (1 - damping)·e
            e = damping * e_p + (1.0 - damping) * e

            if dist < tol:
                st = f"Convergence successful: {i} Iterations"
                break

        return e, function, μ, st
    
    def policy_function(self, c_grid, grid=None):
        '''Interpolate a policy ``c_grid`` (one value per sparse-grid node)
           onto a regular rectangular grid for plotting.'''

        xy_grid = self.xy_grid if grid is None else grid

        x_state_vals = np.repeat(xy_grid[:, 0], xy_grid[:, 1].size)
        y_state_vals = np.tile(xy_grid[:, 1], xy_grid[:, 0].size)
        states = np.column_stack((x_state_vals, y_state_vals))

        z_interp = self.sparse_interpolate(states, c_grid)

        return z_interp.reshape(xy_grid[:, 0].size, xy_grid[:, 1].size)

    def plot_policy_3d(self, c_grid):
        '''This function plots a 3D diagram of a chosen solved function'''

        c_interp = self.policy_function(c_grid)

        xy_grid = self.xy_grid
        state_points = self.state_points

        x_grid = np.atleast_2d(state_points[:, 0])
        y_grid = np.atleast_2d(state_points[:, 1])
        
        fig = plt.figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        x_reg_mesh, y_reg_mesh = np.meshgrid(xy_grid[:, 0], xy_grid[:, 1])

        ax.plot_surface(y_reg_mesh,
                        x_reg_mesh,
                        c_interp.T,
                        rstride=2, cstride=2,
                        cmap='coolwarm',
                        alpha=0.85,
                        linewidth=0.1, 
                        edgecolor='none')

        ax.scatter3D(y_grid,
                     x_grid,
                     c_grid,
                     color='black',
                     s=6,
                     alpha=0.6,
                     zorder=5)

        ax.set_xlabel('Shock', fontsize=10, labelpad=10)
        ax.set_ylabel('State', fontsize=10, labelpad=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.view_init(40, 135)
        # Manually place z-axis label and title using figure coordinates
        # because matplotlib 3D clips set_zlabel at this viewing angle
        fig.text(0.82, 0.5, 'Policy Function', va='center', ha='center',
                 fontsize=10, rotation=90)
        fig.text(0.53, 0.91, 'Policy Function', ha='center', fontsize=13)
        plt.show()
        
    def compute_residuals(self, e):
        '''Solve the model and return the Euler equation residual surface
           on the fine plotting grid.

        Returns
        -------
        ndarray
            Absolute relative residuals shaped like ``xy_fine``.
        '''

        model = self.model
        xy_fine = self.xy_fine

        e_grid, c_grid, μ_grid, _ = self.compute_solution(e)

        c_fine = self.policy_function(c_grid, grid=xy_fine)
        e_fine = self.policy_function(e_grid, grid=xy_fine)
        μ_fine = self.policy_function(μ_grid, grid=xy_fine)

        c_implied = model.c_implied(e_fine, μ_fine, xy_fine[:, 1])

        self.error = np.abs((c_fine - c_implied) / c_implied)
        return self.error

    def plot_errors_3d(self, e=None):
        '''Plot the Euler residuals as a 3D surface alongside their
           distribution in a side-by-side figure.

        Parameters
        ----------
        e : ndarray or None
            If provided, runs ``compute_residuals(e)`` first (which solves
            the model and computes the residual surface). If ``None`` (the
            preferred path), plots the cached ``self.error`` left by a
            prior ``compute_residuals`` call — no re-solve.
        '''

        if e is not None:
            self.compute_residuals(e)
        xy_fine = self.xy_fine
        error = self.error

        # Create side-by-side figure: 3D surface + histogram
        fig = plt.figure(figsize=(14, 5.5), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.25)
        
        # Left panel: 3D surface of Euler residuals
        ax1 = fig.add_subplot(gs[0], projection='3d')
        x_reg_mesh, y_reg_mesh = np.meshgrid(xy_fine[:, 0], xy_fine[:, 1])

        ax1.plot_surface(y_reg_mesh,
                        x_reg_mesh,
                        error.T,
                        cmap='viridis',
                        alpha=0.7,
                        linewidth=0.0,
                        antialiased=True)

        ax1.set_xlabel('Shock', fontsize=9, labelpad=10)
        ax1.set_ylabel('State', fontsize=9, labelpad=10)
        ax1.tick_params(axis='both', labelsize=7)
        ax1.view_init(40, 135)

        # Right panel: histogram of Euler residuals
        ax2 = fig.add_subplot(gs[1])
        error_array = np.concatenate(error, axis=0)
        ax2.hist(error_array, 50, color='#2196F3', alpha=0.75, edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('$|\\varepsilon|$', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(True, alpha=0.2, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        fig.suptitle('Euler Equation Residuals', fontsize=13, y=0.95)
        plt.show()
        
    def plot_errors_dist(self, error=None):
        '''Plot the distribution of the Euler residuals.

        Pass an explicit ``error`` array (from ``compute_residuals``) or rely
        on the cached ``self.error`` left by a prior plot/compute call.
        '''

        if error is None:
            error = self.error

        error_array = np.concatenate(error, axis=0)
        bins = 50
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        ax.set_title('Euler Residual Distribution', fontsize=11)
        ax.set_xlabel('$|\\varepsilon|$', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.hist(error_array, bins, color='#2196F3', alpha=0.75, edgecolor='white', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()  