"""
DSGE model specifications for the SparsePEA solver.

This module contains JIT-compiled model classes built with ``numba.jitclass``.
Each model implements the interface expected by :class:`sparsepea.tools.tools`:

    - ``x_axis_grid(e, grid)``           : expectations → next-period state, policy, KKT multiplier
    - ``rhs_euler(grid, x_p, z_p, ψ_p, e, w)`` : right-hand side of the Euler equation
    - ``c_implied(e_fine, μ_fine, grid)`` : implied policy for Euler residual diagnostics
    - ``ar1_conditional_density(y, z)``   : transition density for the exogenous AR(1) process

Models
------
rbc_jit
    Real Business Cycle model with irreversible investment.
    The planner maximises CRRA utility subject to a Cobb-Douglas production
    function and the constraint K' >= (1-δ)K. The irreversibility constraint
    is enforced via KKT complementarity conditions.

dmp_jit
    Diamond-Mortensen-Pissarides search-and-matching model with the
    Hagedorn and Manovskii (2008) calibration. Features endogenous vacancy
    posting, Nash wage bargaining, and a CES matching function. The
    zero-vacancy constraint is enforced via KKT conditions.

end_dmp_jit
    Endogenous disaster variant of the DMP model with an alternative
    vacancy cost specification (κ = κ₀ + κ₁·q).

Notes
-----
Numba's ``jitclass`` does not support class inheritance, so distribution
helper methods (norm_cdf, log_normal_cdf, etc.) are duplicated across
model classes. This is a known trade-off for JIT compilation performance.
"""

import numpy as np
from numba import float64, njit
from numba.experimental import jitclass
from math import erf, sqrt, pi, log, exp


# =============================================================================
# Shared distribution helpers
# =============================================================================
#
# These are extracted as module-level @njit functions so each jitclass can
# call them as thin wrappers, avoiding duplicated bodies across rbc_jit /
# dmp_jit / end_dmp_jit. jitclass does not support inheritance, so this is
# the standard pattern for sharing code between Numba-compiled classes.


@njit
def _norm_cdf(x, μ, σ):
    """Normal CDF at ``x`` with mean ``μ`` and std ``σ``."""
    return 0.5 * (1.0 + erf((x - μ) / (σ * sqrt(2.0))))


@njit
def _log_normal_cdf(x, μ, σ):
    """Log-normal CDF: P(X ≤ x) where ln(X) ~ N(μ, σ²)."""
    if x <= 0.0:
        return 0.0
    return _norm_cdf(log(x), μ, σ)


@njit
def _log_normal_pdf(x, μ, σ):
    """Log-normal PDF where ln(X) ~ N(μ, σ²)."""
    if x <= 0.0:
        return 0.0
    den = x * σ * sqrt(2.0 * pi)
    return exp(-0.5 * ((log(x) - μ) / σ)**2) / den


@njit
def _log_truncnorm_pdf(x, μ, σ, a, b):
    """Log-normal PDF truncated to the interval [a, b]."""
    if x <= a or b <= x:
        return 0.0
    cdf_a = _log_normal_cdf(a, μ, σ)
    cdf_b = _log_normal_cdf(b, μ, σ)
    return _log_normal_pdf(x, μ, σ) / (cdf_b - cdf_a)


@njit
def _ar1_conditional_density(y, z, ρ, σ, a, b):
    """Truncated log-normal density of y given z under ln(y') = ρ·ln(z) + σ·ε."""
    return _log_truncnorm_pdf(y, ρ * log(z), σ, a, b)


# =============================================================================
# Real Business Cycle Model with Irreversible Investment
# =============================================================================

# Numba jitclass requires explicit type declarations for all instance attributes
rbc_data = [('α', float64),                 # Production function exponent (capital share)
            ('δ', float64),                 # Depreciation rate
            ('η', float64),                 # Coefficient of relative risk aversion
            ('ρ', float64),                 # Persistence of log-TFP AR(1) process
            ('σ', float64),                 # Conditional std dev of TFP innovation
            ('yk',float64),                 # Steady-state output-capital ratio
            ('β', float64),                 # Discount factor (derived from α, yk, δ)
            ('e_min', float64),             # Innovation lower bound (±3σ)
            ('e_max', float64),             # Innovation upper bound (±3σ)
            ('z_min', float64),             # Unconditional TFP lower bound
            ('z_max', float64),             # Unconditional TFP upper bound
            ('k_ss', float64),              # Deterministic steady-state capital
            ('k_min', float64),             # Capital grid lower bound (0.6 × k_ss)
            ('k_max', float64),             # Capital grid upper bound (1.6 × k_ss)
            ('state_bounds', float64[:]),    # [k_min, k_max, z_min, z_max]
            ('shock_bounds', float64[:])]   # [z_min, z_max]

@jitclass(rbc_data)
class rbc_jit:
    """Real Business Cycle model with an irreversible investment constraint.
    
    The social planner maximises:
        v(K,Z) = max_{K'} u(C) + β E[v(K',Z') | Z]
    subject to:
        C = Z·K^α + (1-δ)·K - K'
        K' >= (1-δ)·K               (irreversibility constraint)
        ln(Z') = ρ·ln(Z) + σ·ε'    where ε' ~ N(0,1)
    
    The Euler equation with KKT complementarity is:
        C^(-η) - μ = β E[C'^(-η)·(1-δ + α·Z'·K'^(α-1)) - μ'·(1-δ)]
    where μ >= 0 is the multiplier on the irreversibility constraint.
    """
    
    def __init__(self,
                 α=0.33,
                 δ=0.025,
                 η=3.0,
                 ρ=0.8,
                 σ=0.05,
                 yk=0.11210762331838565):
        
        self.α, self.δ, self.η = α, δ, η
        self.ρ, self.σ = ρ, σ
        self.yk = yk
        self.β = 1.0 / (self.α * self.yk + 1.0 - self.δ)
        
        self.e_min = np.exp(-3 * self.σ)
        self.e_max = np.exp(3 * self.σ)
        
        z_extreme = lambda e: ((np.log(e)) / (1 - self.ρ))
        
        self.z_min = np.exp(z_extreme(self.e_min))
        self.z_max = np.exp(z_extreme(self.e_max))
    
        self.k_ss = ((1 - (self.β * (1 - self.δ))) / (self.α * self.β))**(1 / (self.α - 1))
        
        self.k_min = self.k_ss * 0.6
        self.k_max = self.k_ss * 1.6
        
        self.state_bounds = np.array([self.k_min, self.k_max, self.z_min, self.z_max])
        self.shock_bounds = np.array([self.z_min, self.z_max])

    def ar1_conditional_density(self, y, z):
        '''Conditional density of TFP next period given TFP this period,
           truncated to the unconditional [z_min, z_max] bounds.'''

        return _ar1_conditional_density(y, z, self.ρ, self.σ,
                                        self.z_min, self.z_max)

    def x_axis_grid(self, e, grid_all):
        '''This function solves for K_+ grid, given a guess of the 
           RHS of the Euler equation, e(K, Z) and a grid of the state space
           (K, Z)'''
        
        α, δ, η = self.α, self.δ, self.η
        
        # Storage space
        c = np.zeros(e.shape)
        μ = np.zeros(e.shape)
        k_next = np.zeros(e.shape)
        
        # Generate iteration length which is the same length as the previous K grid
        grid_size = grid_all[:, 1].size
        
        for i in range(grid_size):
            
            # Since this function is JIT compiled we cannot use enumerate
            # We need to extract values from the grid
            state = grid_all[i, :]
            
            # This grid is 2D, we separate each dimension
            k, z = state[0], state[1]
            
            # Extract the value for e(K, Z) for this iteration (K, Z)
            ψ = e[i, :]
            
            # Solve of K_+
            k_p = z * k**α + (1 - δ) * k - ψ[0]**(-1/η)
            
            # Check the KKT condition and store values
            if k_p > (1 - δ) * k:
                c[i, :] = ψ[0]**(-1/η)
                μ[i, :] = 0.0
            else:
                k_p = (1 - δ) * k
                c[i, :] = z * k**α
                μ[i, :] = c[i, :]**(-η) - ψ[0]
        
            # Ensure that the value of K_+ fits within the K grid bounds
            k_check = np.minimum(k_p, self.k_max)
            k_next[i, :] = np.maximum(k_check, self.k_min)
        
        return k_next, c, μ
    
    def rhs_euler(self, grid_all, k_p_grid, z_p_grid, ψ_p_grid, e, w):
        '''This function solves for the RHS of the Euler equation,
           given K_+, Z_+ and e(K_+, Z_+) grids'''
        
        α, δ, η, β = self.α, self.δ, self.η, self.β

        # Storage space
        e_p = k_p_grid.copy()

        # Generate grid lengths for iteration
        grid_size = grid_all[:, 1].size
        z_grid_size = z_p_grid[:, 0].size
    
        for i in range(grid_size):

            # Since this function is JIT compiled we cannot use enumerate
            # We need to extract values from the grid
            state = grid_all[i, :]
            
            # This grid is 2D, we separate each dimension here
            k, z = state[0], state[1]
            
            # Storage space for the integrand of the Euler equation
            integrand = np.zeros(z_p_grid.shape)

            for j in range(z_grid_size):
                
                # Extract single values of K_+, Z_+ and e(K_+, Z_+)
                z_p = z_p_grid[j, :][0]
                k_p = k_p_grid[i, :][0]
                ψ_p = ψ_p_grid[i, j][0]
                
                # Solve for K_++
                k_pp = z_p * k_p**α + (1 - δ) * k_p - ψ_p**(-1/η)
            
                # Check KKT conditions
                if k_pp > (1 - δ) * k_p:
                    c_p = ψ_p**(-1/η)
                    μ_p = 0.0
                else:
                    k_pp = (1 - δ) * k_p
                    c_p = z_p * k_p**α
                    μ_p = c_p**(-η) - ψ_p
                
                # Solve for the probabilities associated with being in state Z_+ given Z
                Qz = self.ar1_conditional_density(z_p, z)
                
                # Solve and store the integrand of the RHS of the Euler equation
                integrand[j] = (β * (c_p**(-η) * (1 - δ + (α * z_p * k_p**(α - 1)))) - (μ_p * (1 - δ))) * Qz
          
            # Integrate the integrand and store RHS of the Euler equation
            e_p[i, :] = np.sum(integrand * w.T)
        
        return e_p
    
    def c_implied(self, e_fine, μ_fine, grid):
        '''This function computes an implied solution of the LHS of the Euler
           equation, used for calculating Euler residuals'''
        
        η = self.η
        
        c_implied = (e_fine + μ_fine)**(-1/η)
        
        return c_implied
    
# =============================================================================
# Diamond-Mortensen-Pissarides Search and Matching Model
# =============================================================================

# This class uses numba, therefore we need to specify the data types at the begining
dmp_data = [('β', float64),                 # Discount factor
            ('ρ', float64),                 # Persistence of log-productivity AR(1)
            ('σ', float64),                 # Conditional std dev of productivity innovation
            ('η', float64),                 # Workers' Nash bargaining weight
            ('b', float64),                 # Flow value of non-market activity
            ('s', float64),                 # Exogenous job separation rate
            ('ι', float64),                 # CES matching function elasticity
            ('κ_k', float64),               # Vacancy cost — capital component
            ('κ_w', float64),               # Vacancy cost — labour component
            ('ξ', float64),                 # Vacancy cost exponent on labour component
            ('e_min', float64),
            ('e_max', float64),
            ('x_min', float64),             # Productivity shock lower bound
            ('x_max', float64),             # Productivity shock upper bound
            ('n_min', float64),             # Employment rate lower bound    
            ('n_max', float64),             # Employment rate upper bound
            ('state_bounds', float64[:]),    # [n_min, n_max, x_min, x_max]
            ('shock_bounds', float64[:])]   # [x_min, x_max]
        
@jitclass(dmp_data)
class dmp_jit:
    """Diamond-Mortensen-Pissarides search model with Hagedorn-Manovskii calibration.
    
    The free-entry condition with KKT complementarity is:
        κ/q(θ) - λ = β E[X' - W' + (1-s)(κ'/q(θ') - λ')]
    where:
        θ = V/(1-N)             labour market tightness
        q(θ) = (1+θ^ι)^(-1/ι)  vacancy-filling rate (CES matching)
        W = η(X + κθ) + (1-η)b Nash-bargained wage
        κ = κ_K·X + κ_W·X^ξ    state-dependent vacancy cost
        ln(X') = ρ·ln(X) + σε' productivity process
    
    The Hagedorn-Manovskii calibration sets b close to average productivity,
    generating realistic labour-market volatility (the Shimer puzzle).
    """
    
    def __init__(self,
                 β=0.99**(1/2),
                 ρ=0.9895,
                 σ=0.0034,
                 η=0.052,
                 b=0.955,
                 s=0.0081,
                 ι=0.407,
                 κ_k=0.474,
                 κ_w=0.11,
                 ξ=0.449,
                 n_min=0.02,
                 n_max=0.98):
        
        self.β, self.ρ, self.σ = β, ρ, σ
        self.η, self.b, self.s, self.ι = η, b, s, ι
        self.κ_k, self.κ_w, self.ξ = κ_k, κ_w, ξ
        
        self.e_min = np.exp(-3.4645 * self.σ)
        self.e_max = np.exp(3.4645 * self.σ)
        
        self.x_min = np.exp((-3.4645 * self.σ) / np.sqrt(1.0 - self.ρ**2.0))
        self.x_max = np.exp((3.4645 * self.σ) / np.sqrt(1.0 - self.ρ**2.0))
        
        self.n_min = n_min
        self.n_max = n_max
        
        self.state_bounds = np.array([self.n_min, self.n_max, self.x_min, self.x_max])
        self.shock_bounds = np.array([self.x_min, self.x_max])

    def ar1_conditional_density(self, y, z):
        '''Conditional density of productivity next period given productivity
           this period, truncated to [x_min, x_max].'''

        return _ar1_conditional_density(y, z, self.ρ, self.σ,
                                        self.x_min, self.x_max)

    def x_axis_grid(self, e, grid_all):
        '''This function solves for N_+ grid, given a guess of the 
           RHS of the Euler equation, e(N, X) and a grid of the state space
           (N, X)'''
        
        κ_k, κ_w, ξ = self.κ_k, self.κ_w, self.ξ
        s, ι = self.s, self.ι
        
        # Storage space
        n_next = np.zeros(e.shape)
        θ = np.zeros(e.shape)
        λ = np.zeros(e.shape)
    
        # Generate iteration length which is the same length as the previous N grid
        grid_size = grid_all[:, 1].size
    
        for i in range(grid_size):
        
            # Since this function is JIT compiled we cannot use enumerate
            # We need to extract values from the grid
            state = grid_all[i, :]
        
            # This grid is 2D, we separate each dimension
            n, x = state[0], state[1]
            
            κ = κ_k * x + κ_w * x**ξ            # Cost incurred in posting vacancies
            
            # Extract the value for e(N, X) for this iteration (N, X)
            ψ = e[i, :]
            
            q = κ / ψ[0]                         # Vacancy filling probability

            # Check the KKT condition and store values
            if q >= 1.0 or q <= 0.0:
                v = 0.0
                q = 1.0
                θ[i, :] = 0.0
                n_p = (1 - s) * n
                λ[i, :] = κ - ψ[0]
            elif q < 1.0:
                θ[i, :] = ((q)**(-ι) - 1)**(1 / ι)
                v = θ[i, :][0] * (1 - n)
                n_p = (1 - s) * n + q * v
                λ[i, :] = 0.0
            
            # Ensure that the value of N_+ fits within the N grid bounds
            n_check = np.minimum(n_p, self.n_max)
            n_next[i, :] = np.maximum(n_check, self.n_min)
   
        return n_next, θ, λ

    def rhs_euler(self, grid_all, n_p_grid, x_p_grid, ψ_p_grid, e, w):
        '''This function solves for the RHS of the Euler equation,
           given N_+, X_+ and e(N_+, X_+) grids'''
        
        κ_k, κ_w, ξ = self.κ_k, self.κ_w, self.ξ
        β, s, ι, η = self.β, self.s, self.ι, self.η
        b = self.b
        
        # Storage space
        e_p = n_p_grid.copy()

        # Generate grid lengths for iteration
        grid_size = grid_all[:, 1].size
        x_grid_size = x_p_grid[:, 0].size
    
        for i in range(grid_size):
        
            # Since this function is JIT compiled we cannot use enumerate
            # We need to extract values from the grid
            state = grid_all[i, :]
            
            # This grid is 2D, we separate each dimension here
            n, x = state[0], state[1]
        
            # Storage space for the integrand of the Euler equation
            integrand = np.zeros(x_p_grid.shape)
        
            for j in range(x_grid_size):
            
                # Extract single values of N_+, X_+ and e(N_+, X_+)
                x_p = x_p_grid[j, :][0]
                n_p = n_p_grid[i, :][0]
                ψ_p = ψ_p_grid[i, j][0]
            
                κ_p = κ_k * x_p + κ_w * x_p**ξ  # Cost incurred in posting vacancies
                q_p = κ_p / ψ_p                 # Vacancy filling probability
                
                # Check KKT conditions
                if q_p >= 1.0 or q_p <= 0.0:
                    q_p = 1.0
                    θ_p = 0.0
                    w_p = η * (x_p + (κ_p * θ_p)) + (1 - η) * b
                    λ_p = κ_p - ψ_p
                elif q_p < 1.0:
                    q_p = κ_p / ψ_p
                    θ_p = ((q_p)**(-ι) - 1)**(1 / ι)
                    w_p = η * (x_p + (κ_p * θ_p)) + (1 - η) * b
                    λ_p = 0.0

                # Solve for the probabilities associated with being in state X_+ given X
                Qz = self.ar1_conditional_density(x_p, x)

                # Solve and store the integrand of the RHS of the Euler equation
                integrand[j] = β * (x_p - w_p + ((1 - s) * ((κ_p / q_p) - λ_p))) * Qz
            # Integrate the integrand and store RHS of the Euler equation
            e_p[i, :] = np.sum(integrand * w.T)
            
        return e_p
    
    def c_implied(self, e_fine, μ_fine, grid):
        '''This function computes an implied solution of the LHS of the Euler
           equation, used for calculating Euler residuals'''
        
        x = grid
        
        κ_k, κ_w, ξ = self.κ_k, self.κ_w, self.ξ
        ι = self.ι 
        
        κ = κ_k * x + κ_w * x**ξ
        
        c_implied = (((e_fine + μ_fine) / κ)**(ι) - 1)**(1 / ι)
        
        return c_implied

# =============================================================================
# Endogenous Disaster DMP Model (alternative vacancy cost specification)
# =============================================================================

# This class uses numba, therefore we need to specify the data types at the begining
end_dmp_data = [('β', float64),                 # Discount factor
                ('ρ', float64),                 # Persistence factor of AR1 process
                ('σ', float64),                 # Conditional volatility of AR1 process
                ('η', float64),                 # Workers' bargaining weight
                ('b', float64),                 # Flow value of unemployment activities
                ('s', float64),                 # Job separation rate
                ('ι', float64),                 # Elasticity of the matching function
                ('κ_0', float64),               # Fixed vacancy cost component
                ('κ_1', float64),               # Variable vacancy cost component
                ('e_min', float64),
                ('e_max', float64),
                ('x_min', float64),             # Shock lower bound
                ('x_max', float64),             # Shock upper bound
                ('n_min', float64),             # Employment lower bound    
                ('n_max', float64),             # Employment upper bound
                ('state_bounds', float64[:]),
                ('shock_bounds', float64[:])]
        
@jitclass(end_dmp_data)
class end_dmp_jit:
    """Endogenous disaster variant of the DMP model.
    
    Uses an alternative vacancy cost specification κ = κ₀ + κ₁ 
    rather than the state-dependent κ = κ_K·X + κ_W·X^ξ in dmp_jit.
    """
    
    def __init__(self,
                 β=0.9954,
                 ρ=0.95**(1/3),
                 σ=0.01,
                 η=0.04,
                 b=0.85,
                 s=0.04,
                 ι=1.25,
                 κ_0=0.5,
                 κ_1=0.5,
                 n_min=0.4,
                 n_max=0.99):
        
        self.β, self.ρ, self.σ = β, ρ, σ
        self.η, self.b, self.s, self.ι = η, b, s, ι
        self.κ_0, self.κ_1 = κ_0, κ_1
        
        self.e_min = np.exp(-3.4645 * self.σ)
        self.e_max = np.exp(3.4645 * self.σ)
        
        self.x_min = np.exp((-4 * self.σ) / np.sqrt(1.0 - self.ρ**2.0))
        self.x_max = np.exp((4 * self.σ) / np.sqrt(1.0 - self.ρ**2.0))
        
        self.n_min = n_min
        self.n_max = n_max
        
        self.state_bounds = np.array([self.n_min, self.n_max, self.x_min, self.x_max])
        self.shock_bounds = np.array([self.x_min, self.x_max])

    def ar1_conditional_density(self, y, z):
        '''Conditional density of productivity next period given productivity
           this period, truncated to [x_min, x_max].'''

        return _ar1_conditional_density(y, z, self.ρ, self.σ,
                                        self.x_min, self.x_max)

    def x_axis_grid(self, e, grid_all):
        '''This function solves for N_+ grid, given a guess of the 
           RHS of the Euler equation, e(N, X) and a grid of the state space
           (N, X)'''
        
        κ_0, κ_1 = self.κ_0, self.κ_1
        s, ι = self.s, self.ι
        
        # Storage space
        n_next = np.zeros(e.shape)
        θ = np.zeros(e.shape)
        λ = np.zeros(e.shape)
    
        # Generate iteration length which is the same length as the previous N grid
        grid_size = grid_all[:, 1].size
    
        for i in range(grid_size):
        
            # Since this function is JIT compiled we cannot use enumerate
            # We need to extract values from the grid
            state = grid_all[i, :]
        
            # This grid is 2D, we separate each dimension
            n, x = state[0], state[1]
            
            # Extract the value for e(N, X) for this iteration (N, X)
            ψ = e[i, :]
            
            q = κ_0 / (ψ[0] - κ_1)                        # Vacancy filling probability
            
            # Check the KKT condition and store values
            if q >= 1.0 or q <= 0.0:
                q = 1.0
                θ[i, :] = 0.0
                v = 0.0
                n_p = (1 - s) * n
                λ[i, :] = (κ_0 + κ_1) - ψ[0]
            elif q < 1.0:
                q = κ_0 / (ψ[0] - κ_1)
                θ[i, :] = ((q)**(-ι) - 1)**(1 / ι)
                v = θ[i, :][0] * (1 - n)
                n_p = (1 - s) * n + q * v
                λ[i, :] = 0.0
            
            # Ensure that the value of N_+ fits within the N grid bounds
            n_check = np.minimum(n_p, self.n_max)
            n_next[i, :] = np.maximum(n_check, self.n_min)
   
        return n_next, θ, λ

    def rhs_euler(self, grid_all, n_p_grid, x_p_grid, ψ_p_grid, e, w):
        '''This function solves for the RHS of the Euler equation,
           given N_+, X_+ and e(N_+, X_+) grids'''
        
        κ_0, κ_1 = self.κ_0, self.κ_1
        β, s, ι, η = self.β, self.s, self.ι, self.η
        b = self.b
        
        # Storage space
        e_p = n_p_grid.copy()

        # Generate grid lengths for iteration
        grid_size = grid_all[:, 1].size
        x_grid_size = x_p_grid[:, 0].size
    
        for i in range(grid_size):
        
            # Since this function is JIT compiled we cannot use enumerate
            # We need to extract values from the grid
            state = grid_all[i, :]
            
            # This grid is 2D, we separate each dimension here
            n, x = state[0], state[1]
            
            ψ = e[i, :]
            
            q = κ_0 / (ψ[0] - κ_1)                        # Vacancy filling probability

            # Check the KKT condition
            if q >= 1.0 or q <= 0.0:
                q = 1.0
                κ = κ_0 + κ_1
                v = 0.0
            elif q < 1.0:
                q = κ_0 / (ψ[0] - κ_1)
                κ = κ_0 + (κ_1 * q)
                θ = ((q)**(-ι) - 1)**(1 / ι)
                v = θ * (1 - n)

            c = (x * n) - κ * v
            
            # Storage space for the integrand of the Euler equation
            integrand = np.zeros(x_p_grid.shape)
        
            for j in range(x_grid_size):
            
                # Extract single values of N_+, X_+ and e(N_+, X_+)
                x_p = x_p_grid[j, :][0]
                n_p = n_p_grid[i, :][0]
                ψ_p = ψ_p_grid[i, j][0]
                
                q_p = κ_0 / (ψ_p - κ_1)               # Vacancy filling probability
                
                # Check KKT conditions
                if q_p >= 1.0 or q_p <= 0.0:
                    q_p = 1.0
                    κ_p = κ_0 + κ_1
                    θ_p = 0.0
                    w_p = η * (x_p + (κ_p * θ_p)) + (1 - η) * b
                    v_p = 0.0
                    λ_p = κ_0 + κ_1 - ψ_p
                elif q_p < 1.0:
                    q_p = κ_0 / (ψ_p - κ_1)
                    κ_p = κ_0 + (κ_1 * q_p)
                    θ_p = ((q_p)**(-ι) - 1)**(1 / ι)
                    w_p = η * (x_p + (κ_p * θ_p)) + (1 - η) * b
                    v_p = θ_p * (1 - n_p) 
                    λ_p = 0.0

                c_p = (x_p * n_p) - (κ_p * v_p)
                
                # Solve for the probabilities associated with being in state X_+ given X
                
                Qz = self.ar1_conditional_density(x_p, x)
                
                # Stochastic discount factor under log utility: M_{t+1} = β·C_t/C_{t+1}
                # (Petrosky-Nadeau, Zhang & Kuehn 2018, AER, eq. 8).
                M = β * (c / c_p)

                # Solve and store the integrand of the RHS of the Euler equation
                integrand[j] = M * (x_p - w_p + ((1 - s) * ((κ_0 / q_p) + κ_1 - λ_p))) * Qz

            # Integrate the integrand and store RHS of the Euler equation
            e_p[i, :] = np.sum(integrand * w.T)

        return e_p

    def c_implied(self, e_fine, μ_fine, grid):
        '''This function computes an implied solution of the LHS of the Euler
           equation, used for calculating Euler residuals'''

        κ_0, κ_1 = self.κ_0, self.κ_1
        ι = self.ι

        c_implied = (((e_fine + μ_fine - κ_1) / κ_0)**(ι) - 1)**(1 / ι)

        return c_implied