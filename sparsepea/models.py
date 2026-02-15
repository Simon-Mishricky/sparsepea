import numpy as np
from numba import float64
from numba.experimental import jitclass
from math import *
import quantecon.markov as qe


# This class uses numba, therefore we need to specify the data types at the begining
rbc_data = [('α', float64),                 # Production function exponent
            ('δ', float64),                 # Depreciation rate
            ('η', float64),                 # Utility function exponent
            ('ρ', float64),                 # Persistence factor of AR1 process
            ('σ', float64),                 # Conditional volatility of AR1 process
            ('yk',float64),                 # Output - capital ratio
            ('β', float64),                 # Discount factor
            ('e_min', float64),
            ('e_max', float64),
            ('z_min', float64),             # Shock lower bound
            ('z_max', float64),             # Shock upper bound
            ('k_ss', float64),              # Capital steady state
            ('k_min', float64),             # Capital lower bound
            ('k_max', float64),             # Capital upper bound
            ('state_bounds', float64[:]),
            ('shock_bounds', float64[:])]

@jitclass(rbc_data)
class rbc_jit:
    
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
        
    def norm_cdf(self, x, μ):
        '''This function solves for the cumulative density function 
           of the normal distribution given μ and σ'''
        
        σ = self.σ 
    
        cdf = 0.5 * (1 + erf((x - μ) / (σ * np.sqrt(2))))
    
        return cdf
    
    def log_normal_cdf(self, x, μ):
        '''This function solves for the cumulative density function
           of the log-normal distribution given μ and σ'''
        
        σ = self.σ 
        
        if x <= 0.0:
            cdf = 0.0
        else:
            logx = np.log(x)
            cdf = self.norm_cdf(logx, μ)
    
        return cdf
    
    def log_normal_pdf(self, x, μ):
        '''This function solves for the probability density function
           of the log-normal distribution given μ and σ'''
        
        σ = self.σ
    
        if x <= 0.0:
            pdf = 0.0
        else:
            den = x * σ * np.sqrt(2.0 * np.pi)
            pdf = np.exp(-0.5 * ((np.log(x) - μ) / σ)**2) / den
        
        return pdf
    
    def log_truncnorm_pdf(self, x, μ):
        '''This function solves for the probability density function
           of a truncated normal distribution given μ, σ and truncations
           bounds [a, b]'''
        
        σ, a, b = self.σ, self.z_min, self.z_max

        # Check whether x fits within the truncation bounds
        if x <= a:
            pdf = 0.0
        elif b <= x:
            pdf = 0.0
        else:
            cdf_a = self.log_normal_cdf(a, μ)
            cdf_b = self.log_normal_cdf(b, μ)
            pdf_x = self.log_normal_pdf(x, μ)
        
            pdf = pdf_x / (cdf_b - cdf_a)
        
        return pdf

    def ar1_conditional_density(self, y, z):
        '''This function solves for the conditional density generated 
           by an AR1 process given this period state y and the 
           previous period state z'''
        
        ρ = self.ρ
        
        μ = ρ * np.log(z)
    
        return self.log_truncnorm_pdf(y, μ)
        
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
    
# This class uses numba, therefore we need to specify the data types at the begining
dmp_data = [('β', float64),                 # Discount factor
            ('ρ', float64),                 # Persistence factor of AR1 process
            ('σ', float64),                 # Conditional volatility of AR1 process
            ('η', float64),                 # Workers' bargaining weight
            ('b', float64),                 # Flow value of unemployment activities
            ('s', float64),                 # Job separation rate
            ('ι', float64),                 # Elasticity of the matching function
            ('κ_k', float64),               # Capital cost parameter
            ('κ_w', float64),               # Labour cost parameter
            ('ξ', float64),                 # Exponential parameter for labour cost
            ('e_min', float64),
            ('e_max', float64),
            ('x_min', float64),             # Shock lower bound
            ('x_max', float64),             # Shock upper bound
            ('n_min', float64),             # Employment lower bound    
            ('n_max', float64),             # Employment upper bound
            ('state_bounds', float64[:]),
            ('shock_bounds', float64[:])]
        
@jitclass(dmp_data)
class dmp_jit:
    '''This model uses the parameters from Hagedorn and Manovskii'''
    
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
        
    def norm_cdf(self, x, μ):
        '''This function solves for the cumulative density function 
           of the normal distribution given μ and σ'''        
        
        σ = self.σ 
    
        cdf = 0.5 * (1 + erf((x - μ) / (σ * np.sqrt(2))))
    
        return cdf
    
    def log_normal_cdf(self, x, μ):
        '''This function solves for the cumulative density function
           of the log-normal distribution given μ and σ'''
    
        σ = self.σ 
        
        if x <= 0.0:
            cdf = 0.0
        else:
            logx = np.log(x)
            cdf = self.norm_cdf(logx, μ)
    
        return cdf
    
    def log_normal_pdf(self, x, μ):
        '''This function solves for the probability density function
           of the log-normal distribution given μ and σ'''
        
        σ = self.σ
    
        if x <= 0.0:
            pdf = 0.0
        else:
            den = x * σ * np.sqrt(2.0 * np.pi)
            pdf = np.exp(-0.5 * ((np.log(x) - μ) / σ)**2) / den
        
        return pdf
    
    def log_truncnorm_pdf(self, x, μ):
        '''This function solves for the probability density function
           of a truncated normal distribution given μ, σ and truncations
           bounds [a, b]'''
        
        σ, a, b = self.σ, self.x_min, self.x_max

        if x <= a:
            pdf = 0.0
        elif b <= x:
            pdf = 0.0
        else:
            cdf_a = self.log_normal_cdf(a, μ)
            cdf_b = self.log_normal_cdf(b, μ)
            pdf_x = self.log_normal_pdf(x, μ)
        
            pdf = pdf_x / (cdf_b - cdf_a)
        
        return pdf

    def ar1_conditional_density(self, y, z):
        '''This function solves for the conditional density generated 
           by an AR1 process given this period state y and the 
           previous period state z'''
        
        ρ = self.ρ
        
        μ = ρ * np.log(z)
    
        return self.log_truncnorm_pdf(y, μ)
    
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

# This class uses numba, therefore we need to specify the data types at the begining
end_dmp_data = [('β', float64),                 # Discount factor
                ('ρ', float64),                 # Persistence factor of AR1 process
                ('σ', float64),                 # Conditional volatility of AR1 process
                ('η', float64),                 # Workers' bargaining weight
                ('b', float64),                 # Flow value of unemployment activities
                ('s', float64),                 # Job separation rate
                ('ι', float64),                 # Elasticity of the matching function
                ('κ_0', float64),               # Capital cost parameter
                ('κ_1', float64),               # Labour cost parameter
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
    '''This model uses the parameters from Hagedorn and Manovskii'''
    
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
        
    def norm_cdf(self, x, μ):
        '''This function solves for the cumulative density function 
           of the normal distribution given μ and σ'''        
        
        σ = self.σ 
    
        cdf = 0.5 * (1 + erf((x - μ) / (σ * np.sqrt(2))))
    
        return cdf
    
    def log_normal_cdf(self, x, μ):
        '''This function solves for the cumulative density function
           of the log-normal distribution given μ and σ'''
    
        σ = self.σ 
        
        if x <= 0.0:
            cdf = 0.0
        else:
            logx = np.log(x)
            cdf = self.norm_cdf(logx, μ)
    
        return cdf
    
    def log_normal_pdf(self, x, μ):
        '''This function solves for the probability density function
           of the log-normal distribution given μ and σ'''
        
        σ = self.σ
    
        if x <= 0.0:
            pdf = 0.0
        else:
            den = x * σ * np.sqrt(2.0 * np.pi)
            pdf = np.exp(-0.5 * ((np.log(x) - μ) / σ)**2) / den
        
        return pdf
    
    def log_truncnorm_pdf(self, x, μ):
        '''This function solves for the probability density function
           of a truncated normal distribution given μ, σ and truncations
           bounds [a, b]'''
        
        σ, a, b = self.σ, self.x_min, self.x_max

        if x <= a:
            pdf = 0.0
        elif b <= x:
            pdf = 0.0
        else:
            cdf_a = self.log_normal_cdf(a, μ)
            cdf_b = self.log_normal_cdf(b, μ)
            pdf_x = self.log_normal_pdf(x, μ)
        
            pdf = pdf_x / (cdf_b - cdf_a)
        
        return pdf

    def ar1_conditional_density(self, y, z):
        '''This function solves for the conditional density generated 
           by an AR1 process given this period state y and the 
           previous period state z'''
        
        ρ = self.ρ
        
        μ = ρ * np.log(z)
    
        return self.log_truncnorm_pdf(y, μ)
    
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
                
                # Solve and store the integrand of the RHS of the Euler equation
                integrand[j] = β  * (x_p - w_p + ((1 - s) * ((κ_0 / q_p) + κ_1 - λ_p))) * Qz
                
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