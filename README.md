# SparsePEA: Sparse Grid Parameterized Expectations Algorithm

A Python package for solving dynamic stochastic general equilibrium (DSGE) models using the Parameterized Expectations Algorithm (PEA) with sparse grid interpolation.

**Author:** Simon Mishricky  
**Affiliation:** Australian National University

## Overview

SparsePEA implements a computationally efficient method for solving high-dimensional macroeconomic models by combining:
- **Parameterized Expectations Algorithm (PEA):** A projection method that parameterizes conditional expectations in Euler equations
- **Sparse Grid Interpolation:** Using Tasmanian's local polynomial grids to dramatically reduce the curse of dimensionality
- **JIT Compilation:** Leveraging Numba for high-performance numerical computation

This approach is particularly well-suited for models with:
- Occasionally binding constraints (e.g., irreversible investment, collateral constraints)
- Multiple state variables
- Non-linear dynamics
- Stochastic shocks

## Features

- **Multiple Pre-built Models:**
  - Real Business Cycle (RBC) model with irreversible investment
  - Diamond-Mortensen-Pissarides (DMP) search and matching model

- **Efficient Numerical Methods:**
  - Sparse grid state space representation (drastically reduces grid points)
  - Adaptive sparse grid integration for expectation operators
  - Fast JIT-compiled model evaluation using Numba

- **Solution Quality Analysis:**
  - Euler equation residual computation
  - 3D visualization of policy functions and errors
  - Error distribution analysis

## Installation

### Prerequisites

```bash
# Install Tasmanian (sparse grid library)
# See: https://tasmanian.ornl.gov/

# For macOS with Homebrew:
brew install tasmanian

# For Ubuntu/Debian:
# Build from source or use available packages
```

### Python Dependencies

```bash
pip install numpy numba matplotlib quantecon
pip install scikit-tasmanian  # Python bindings for Tasmanian
```

## Quick Start

### Example: Real Business Cycle Model

```python
import numpy as np
from sparsepea.models import rbc_jit
from sparsepea.tools import tools

# Initialize the RBC model
rbc_model = rbc_jit()

# Set up the sparse grid solver
solver = tools(model=rbc_model, 
               states_input=2,   # Capital and TFP shock
               states_output=1,
               shocks_input=1,
               shocks_output=1,
               depth=8,          # Sparse grid depth
               order=1)          # Polynomial order

# Create initial guess for Euler equation
grid_states, grid_states_points = solver.make_states_grid()
e_initial = np.ones((grid_states_points.shape[0], 1)) * 0.5

# Solve the model
e_solution, policy_function, multiplier, status = solver.compute_solution(e_initial)
print(status)

# Visualize the solution
solver.plot_policy_3d(policy_function)

# Analyze solution accuracy
solver.plot_errors_3d(e_initial)
solver.plot_errors_dist()
```

### Example: DMP Search Model

```python
from sparsepea.models import dmp_jit

# Initialize Diamond-Mortensen-Pissarides model
dmp_model = dmp_jit()

# Set up solver (same interface as RBC)
solver = tools(model=dmp_model, 
               states_input=2,   # Employment and productivity
               depth=8)

# Solve and analyze...
```

## Repository Structure

```
sparsepea/
├── models.py              # DSGE model specifications (JIT-compiled)
├── tools.py               # Solver and interpolation tools
├── notebooks/
│   ├── Sparsepea_Module.ipynb              # Package usage tutorial
│   └── Petrovsky-Nadeau_Zhang_Simulations.ipynb  # Replication example
├── README.md
├── requirements.txt
└── LICENSE
```

## Models Included

### 1. Real Business Cycle with Irreversible Investment (`rbc_jit`)

A standard RBC model with an irreversibility constraint on investment. Features:
- Occasionally binding constraint (investment ≥ 0)
- TFP follows AR(1) process
- Complementarity conditions handled via KKT approach

**Key parameters:** Capital share (α), depreciation (δ), risk aversion (η), TFP persistence (ρ)

### 2. Diamond-Mortensen-Pissarides Model (`dmp_jit`)

Search and matching model of unemployment with Hagedorn-Manovskii calibration:
- Endogenous job creation (vacancy posting)
- Nash wage bargaining
- Job destruction shocks
- Matching function with congestion effects

**Key parameters:** Bargaining power (η), separation rate (s), matching efficiency (κ)

## Methodology

### Parameterized Expectations Algorithm (PEA)

The PEA solves dynamic models by parameterizing the conditional expectation term in Euler equations. For a generic Euler equation:

```
E_t[m(x_t, x_{t+1}, θ)] = 0
```

We approximate `ψ(x_t) = E_t[g(x_{t+1})]` using a flexible functional form and iterate until convergence.

**Advantages:**
- Naturally handles occasionally binding constraints
- No need for explicit policy function iteration
- Efficient for models with persistent shocks

### Sparse Grids

Traditional tensor product grids suffer from the curse of dimensionality (grid points grow exponentially with dimensions). Sparse grids reduce this to near-linear growth by adaptively selecting only the most important grid points.

**Computational savings:**
- 2D problem: ~100 points vs 10,000+ for tensor grid
- 3D problem: ~500 points vs 1,000,000+ for tensor grid

## Example Applications

### Replicating Petrosky-Nadeau and Zhang (2017)

The `Petrovsky-Nadeau_Zhang_Simulations.ipynb` notebook demonstrates:
- Model calibration to match U.S. labor market moments
- Business cycle statistics computation
- Impulse response functions to productivity shocks
- Comparison with published results

## Performance

Typical solve times on a modern laptop:
- RBC model (depth=8): ~10-30 seconds
- DMP model (depth=8): ~30-60 seconds

Euler equation errors:
- Median error: < 0.01%
- 99th percentile: < 0.1%

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mishricky2025sparsepea,
  author = {Mishricky, Simon},
  title = {SparsePEA: Sparse Grid Parameterized Expectations Algorithm},
  year = {2025},
  url = {https://github.com/simon-mishricky/sparsepea}
}
```

## Related Research

This implementation builds on methodological work from:
- **Maliar and Maliar (2015):** "Merging simulation and projection approaches to solve high-dimensional problems with an application to a new Keynesian model"
- **Petrosky-Nadeau and Zhang (2017):** "Solving the Diamond-Mortensen-Pissarides model accurately"
- **Judd, Maliar and Maliar (2011):** "Numerically stable and accurate stochastic simulation approaches for solving dynamic economic models"

## License

MIT License - see LICENSE file for details

## Contact

**Simon Mishricky**  
Email: simon.mishricky@gmail.com  
Website: [github.com/Simon-Mishricky](https://github.com/Simon-Mishricky)
