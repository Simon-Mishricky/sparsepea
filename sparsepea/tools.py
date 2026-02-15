import numpy as np 
from numba import float64
from numba.experimental import jitclass
import Tasmanian as ts
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import *

class tools:
    
    def __init__(self,
                 model,                   # Input of a chosen model
                 states_input=2,          # Number of dimensions of the state space
                 states_output=1,         # Dimensions of the state space grid
                 shocks_input=1,          # Number of dimensions of the shocks
                 shocks_output=1,         # Dimensions of the shocks
                 depth=8,                 # Number of sparse grid points
                 order=1,                 # Order of the polynomial for grid
                 max_iter=500,            # Maximum number of iterations to solve e
                 tol=1e-5):               # Tolerance criterion
        
        self.states_input = states_input
        self.states_output = states_output
        self.shocks_input = shocks_input
        self.shocks_output = shocks_output
        self.depth = depth
        self.order = order
        self.max_iter = max_iter
        self.tol = tol
        self.model = model
        self.state_bounds = np.reshape(model.state_bounds, (2, 2))
        self.shock_bounds = np.atleast_2d(model.shock_bounds)
        self.xy_grid = np.linspace((self.state_bounds[0, 0], self.state_bounds[1, 0]), (self.state_bounds[0, 1], self.state_bounds[1, 1]), 200)
        self.xy_fine = np.linspace((self.state_bounds[0, 0], self.state_bounds[1, 0]), (self.state_bounds[0, 1], self.state_bounds[1, 1]), 1000)
        
    def make_states_grid(self):
        '''This function creates a 2D sparse grid using Tasmanian, 
           used for the state space of the pea algorithm'''
        
        states_input, states_output = self.states_input, self.states_output
        depth, order = self.depth, self.order
        
        # Generate 2D local polynomial grid
        grid_states = ts.makeLocalPolynomialGrid(states_input, states_output, depth, order, "localp")
        grid_states.setDomainTransform(self.state_bounds)
        grid_states_points = grid_states.getPoints()
        
        return grid_states, grid_states_points
    
    def make_shocks_grid(self):
        '''This function creates a 1D sparse grid using Tasmanian,
           used for the shock grid in the pea algorithm'''
        
        shocks_input, shocks_output = self.shocks_input, self.shocks_output
        depth, order = self.depth, self.order
        
        # Generate 1D local polynomial grid
        grid_shocks = ts.makeLocalPolynomialGrid(shocks_input, shocks_output, depth, order, "localp")
        grid_shocks.setDomainTransform(self.shock_bounds)
        grid_shocks_points = grid_shocks.getPoints()
        
        # Generate quadrature weights
        grid_shocks_weights = grid_shocks.getQuadratureWeights()
        w = np.atleast_2d(grid_shocks_weights)
        
        return grid_shocks, grid_shocks_points, w
    
    def sparse_interpolate(self, state_p, e):
        '''This function interpolates over the 2D sparse grid, 
           given a single next period point in order to compute e'''
        
        grid_states, grid_states_points = self.make_states_grid()
        
        grid_states.loadNeededPoints(e)
        
        # Interpolate given the state variables
        interp = grid_states.evaluateBatch(state_p)
    
        return interp
    
    def euler_interpolation(self, x_p_grid, e):
        '''This function interpolates for all next period values
           and solves all values of e'''
        
        y_grid, y_p_grid, w = self.make_shocks_grid()
        
        x_state_vals = np.repeat(x_p_grid[:, 0], y_p_grid[:, 0].size)
        y_state_vals = np.tile(y_p_grid[:, 0], x_p_grid[:, 0].size)
        states = np.column_stack((x_state_vals, y_state_vals))
        z_interp = self.sparse_interpolate(states, e)
        
        return z_interp.reshape(x_p_grid.size, y_p_grid.size, 1)
    
    def compute_solution(self, e):
        '''This function solves for the optimal e using pea algorithm'''
        
        model = self.model
        max_iter, tol = self.max_iter, self.tol
        y_grid, y_p_grid, w = self.make_shocks_grid()
        grid_states, grid_states_points = self.make_states_grid()
        
        for i in range(max_iter):
            
            # Solve for the next period x_axis grid
            x_p_grid, function, μ = model.x_axis_grid(e, grid_states_points)
            
            # Given next period grids interpolate to find e
            ψ_p_grid = self.euler_interpolation(x_p_grid, e)
            
            # Solve for new e
            e_p = model.rhs_euler(grid_states_points, x_p_grid, y_p_grid, ψ_p_grid, e, w)
            # # Use the supremum norm to evaluate the distance between the inital e and the newly generated e
            dist = (np.abs(e_p - e)).max()
            
            # Update initital guess with newly generated e
            e = e_p
        
            if dist < tol:
                st = "Convergence successful: " + str(i) + " Iterations"
                break
  
        return e_p, function, μ, st
    
    def policy_function(self, c_grid, grid=None):
        '''This function generates a policy function defined on a regular
           grid given a sparse grid'''
        
        if grid is None:
            xy_grid = self.xy_grid
        else:
            xy_grid = grid
        
        grid_states, grid_states_points = self.make_states_grid()
        
        x_state_vals = np.repeat(xy_grid[:, 0], xy_grid[:, 1].size)
        y_state_vals = np.tile(xy_grid[:, 1], xy_grid[:, 0].size)
        states = np.column_stack((x_state_vals, y_state_vals))
    
        z_interp = self.sparse_interpolate(states, c_grid)
    
        return z_interp.reshape(xy_grid[:, 0].size, xy_grid[:, 1].size)
    
    def plot_policy_3d(self, c_grid):
        '''This function plots a 3D diagram of a chosen solved function'''
        
        c_interp = self.policy_function(c_grid)
        
        xy_grid = self.xy_grid
        grid_states, grid_states_points = self.make_states_grid()
        
        x_grid = np.atleast_2d(grid_states.getPoints()[:, 0])
        y_grid = np.atleast_2d(grid_states.getPoints()[:, 1])
        
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        x_reg_mesh, y_reg_mesh = np.meshgrid(xy_grid[:, 0], xy_grid[:, 1])

        ax.plot_surface(y_reg_mesh,
                        x_reg_mesh,
                        c_interp.T,
                        rstride=2, cstride=2,
                        color='grey',
                        alpha=0.9,
                        linewidth=0.25, 
                        edgecolor="none")

        ax.scatter3D(y_grid,
                     x_grid,
                     c_grid,
                     color='black')

        ax.set_zlabel('Policy Function', fontsize=14)
        ax.view_init(40, 135)
        plt.show()
        
    def plot_errors_3d(self, e):
        '''This function plots a 3D diagram of the Euler residuals'''
        
        model = self.model
        xy_fine = self.xy_fine
        
        e_grid, c_grid, μ_grid, st = self.compute_solution(e)
        
        c_fine = self.policy_function(c_grid, grid=xy_fine)
        e_fine = self.policy_function(e_grid, grid=xy_fine)
        μ_fine = self.policy_function(μ_grid, grid=xy_fine)
        
        c_implied = model.c_implied(e_fine, μ_fine, xy_fine[:, 1])
        
        self.error = np.abs((c_fine - c_implied) / c_implied)
        
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        x_reg_mesh, y_reg_mesh = np.meshgrid(xy_fine[:, 0], xy_fine[:, 1])

        ax.plot_surface(y_reg_mesh,
                        x_reg_mesh,
                        self.error.T,
                        antialiased=False,
                        alpha=0.3,
                        linewidth=0.0)

        ax.set_zlabel('Euler Residuals', fontsize=14)
        ax.view_init(40, 135)
        plt.show()
        
    def plot_errors_dist(self):
        '''This function plots this distribution of the Euler residuals'''
        
        error = self.error
        
        error_array = np.concatenate(error, axis=0)
        bins = 50
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Error Distribution')
        ax.set_xlabel('$\epsilon$')
        ax.hist(error_array, bins, color='green', alpha=0.5, edgecolor='black')
        ax.grid(True)
        plt.show()  