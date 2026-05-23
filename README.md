# SparsePEA

[![test](https://github.com/Simon-Mishricky/sparsepea/actions/workflows/test.yml/badge.svg)](https://github.com/Simon-Mishricky/sparsepea/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python solver for DSGE models with occasionally binding constraints, using the Parameterised Expectations Algorithm on [Tasmanian](https://tasmanian.ornl.gov/) sparse grids. Solves a calibrated DMP search model in under 60 seconds on a laptop, with Euler equation errors below 0.01%.

```python
import numpy as np
from sparsepea.models import rbc_jit
from sparsepea.tools import tools

model = rbc_jit()
solver = tools(model=model, depth=8)
grid, grid_points = solver.make_states_grid()

e_init = np.ones((grid_points.shape[0], 1)) * 0.5
e_solution, policy, multiplier, status = solver.compute_solution(e_init)
print(status)  # "Convergence successful: 142 Iterations"

solver.plot_policy_3d(policy)
```

## Why this exists

Two problems make DSGE models hard to solve numerically:

1. **The curse of dimensionality.** Tensor-product grids grow exponentially in the number of state variables. A 2D problem with 100 points per dimension needs 10,000 grid points; a 3D problem needs 1,000,000. Sparse grids (Smolyak, 1963) reduce this to polynomial growth — roughly 100 points in 2D and 500 in 3D — by keeping only the grid points that matter most for interpolation accuracy.

2. **Occasionally binding constraints.** Perturbation methods linearise around a steady state, which fails at the kink where a constraint starts binding. The PEA is a global projection method that checks complementarity conditions pointwise, so it handles the binding and non-binding regions simultaneously without smoothing or penalty functions.

SparsePEA combines both, using [Tasmanian](https://tasmanian.ornl.gov/) for sparse grid construction and quadrature, and [Numba](https://numba.pydata.org/) `@jitclass` for JIT-compiled inner loops.

## Install

SparsePEA depends on the [Tasmanian](https://tasmanian.ornl.gov/) sparse-grid library from Oak Ridge National Laboratory. Install it separately first:

```bash
# macOS — Homebrew ships the Python bindings
brew install tasmanian

# Linux — build from source via pip (needs cmake + a C++ compiler)
sudo apt-get install -y cmake build-essential
pip install Tasmanian
```

Then install the package:

```bash
git clone https://github.com/Simon-Mishricky/sparsepea.git
cd sparsepea
pip install -e .
```

## Models included

**`rbc_jit`** — Real Business Cycle with irreversible investment. A social planner maximises lifetime CRRA utility subject to a Cobb–Douglas production function and the constraint that gross investment cannot be negative. TFP follows a log-AR(1). The irreversibility constraint is enforced via KKT complementarity conditions. Parameters: capital share α, depreciation δ, risk aversion η, TFP persistence ρ.

**`dmp_jit`** — Diamond–Mortensen–Pissarides search and matching with the Hagedorn–Manovskii (2008) calibration. Firms post vacancies, workers search, and a CES matching function governs meetings. Wages are Nash-bargained. The flow value of unemployment is set close to productivity, generating the large labour-market fluctuations that standard calibrations miss (the Shimer puzzle). Vacancy costs are state-dependent: κ = κ_K·X + κ_W·X^ξ. The zero-vacancy constraint binds in bad states and is enforced via KKT conditions. Parameters: bargaining power η, separation rate s, matching elasticity ι, vacancy cost components κ_K, κ_W, and vacancy cost exponent ξ.

**`end_dmp_jit`** — Endogenous disaster variant of the DMP model with an alternative vacancy cost specification κ = κ₀ + κ₁·q, where q is the vacancy-filling probability. Parameters: bargaining power η, separation rate s, matching elasticity ι, fixed vacancy cost κ₀, variable vacancy cost κ₁.

## How it works

The PEA parameterises the conditional expectation in the Euler equation:

$$\psi(x_t) \approx \mathbb{E}_t\bigl[h(x_t, x_{t+1})\bigr]$$

and iterates on a fixed-point mapping: given a guess $\psi$, solve for the endogenous variables (capital or employment), interpolate $\psi$ at the implied next-period states, evaluate the right-hand side of the Euler equation by numerical integration, and update $\psi$. At each grid point, the algorithm checks whether the occasionally binding constraint is active and applies the appropriate complementarity condition.

Sparse grids enter in two places: the **state-space grid** on which $\psi$ is defined and interpolated, and the **quadrature rule** used to compute the expectation integral over future shocks. Both use Tasmanian's local polynomial basis, which provides exact interpolation at the grid nodes and smooth approximation elsewhere.

The solver iterates until the sup-norm distance between successive $\psi$ iterates falls below a tolerance (default $10^{-5}$). Convergence is typically achieved in 100–400 iterations depending on the model.

## Solution diagnostics

SparsePEA reports solution quality via **Euler equation residuals** — the percentage error in the Euler equation at a fine regular grid over the state space. If the solution were exact, the residual would be zero everywhere. Residuals below $10^{-3}$ (0.1%) are considered acceptable in the computational economics literature; SparsePEA typically achieves $10^{-4}$ at the median.

```python
solver.plot_errors_3d(e_init)       # 3D surface of residuals + histogram
solver.plot_errors_dist()           # standalone histogram (backward compat)
```

## Notebooks

| Notebook | Description |
|:---------|:------------|
| [`tutorial.ipynb`](tutorial.ipynb) | Solves both models from scratch with full mathematical exposition |
| [`petrosky_nadeau_zhang_replication.ipynb`](petrosky_nadeau_zhang_replication.ipynb) | Replicates Panel D of Table 1 from Petrosky-Nadeau & Zhang (2017, *QE*) |

## Performance

| Model | Grid depth | Solve time | Median Euler error |
|:------|:---:|:---:|:---:|
| RBC | 8 | 10–30 s | < 0.01% |
| DMP | 8 | 30–60 s | < 0.01% |

Benchmarked on an Apple M-series laptop.

## Project layout

```
sparsepea/
├── sparsepea/
│   ├── __init__.py          # Public API
│   ├── models.py            # JIT-compiled model specifications (rbc_jit, dmp_jit, end_dmp_jit)
│   └── tools.py             # Solver, interpolation, diagnostics
├── tests/
│   └── test_models.py
├── tutorial.ipynb           # Package walkthrough
├── petrosky_nadeau_zhang_replication.ipynb
├── setup.py
├── requirements.txt
└── LICENSE                  # MIT
```

## Adding a new model

Write a `@jitclass` that implements four methods:

| Method | Purpose |
|:-------|:--------|
| `x_axis_grid(e, grid_all)` | Map expectations → next-period endogenous state, policy, and KKT multiplier |
| `rhs_euler(grid_all, x_p_grid, z_p_grid, ψ_p_grid, e, w)` | Evaluate the RHS of the Euler equation via quadrature |
| `c_implied(e_fine, μ_fine, grid)` | Implied policy for Euler residual diagnostics |
| `ar1_conditional_density(y, z)` | Transition density for the exogenous shock process |

The solver handles grid construction, interpolation, iteration, and plotting. See `models.py` for working examples.

## References

- Judd, K., Maliar, L. and Maliar, S. (2011). Numerically stable and accurate stochastic simulation approaches for solving dynamic economic models. *Quantitative Economics*, 2(2), 173–210.
- Maliar, L. and Maliar, S. (2015). Merging simulation and projection approaches to solve high-dimensional problems with an application to a new Keynesian model. *Quantitative Economics*, 6(1), 1–47.
- Petrosky-Nadeau, N. and Zhang, L. (2017). Solving the Diamond–Mortensen–Pissarides model accurately. *Quantitative Economics*, 8(2), 611–650.
- Hagedorn, M. and Manovskii, I. (2008). The cyclical behavior of equilibrium unemployment and vacancies revisited. *American Economic Review*, 98(4), 1692–1706.

## Citation

```bibtex
@software{mishricky2025sparsepea,
  author  = {Mishricky, Simon},
  title   = {{SparsePEA}: Sparse Grid Parameterised Expectations Algorithm},
  year    = {2025},
  url     = {https://github.com/Simon-Mishricky/sparsepea}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
