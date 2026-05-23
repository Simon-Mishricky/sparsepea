"""Basic smoke tests for SparsePEA model construction and solver."""

import numpy as np
import pytest


def _rbc_initial_guess(grid_states):
    """Log-linear certainty-equivalent guess for the RBC expectation function."""
    α = 0.33
    δ = 0.025
    y_k = 0.11210762331838565
    β = 1.0 / (α * y_k + 1.0 - δ)
    γ = np.array([-np.log(1 - α * β), -α, -1.0])

    n = grid_states.shape[0]
    e0 = np.zeros((n, 1))
    for i in range(n):
        k, z = grid_states[i, 0], grid_states[i, 1]
        e0[i, 0] = np.exp(γ[0] + γ[1] * np.log(k) + γ[2] * np.log(z))
    return e0


def test_rbc_model_initialises():
    """rbc_jit should construct with default parameters and compute derived values."""
    from sparsepea.models import rbc_jit

    model = rbc_jit()

    assert 0.0 < model.β < 1.0, "Discount factor should be in (0, 1)"
    assert model.k_min < model.k_ss < model.k_max
    assert model.z_min < 1.0 < model.z_max
    assert model.state_bounds.shape == (4,)
    assert model.shock_bounds.shape == (2,)


def test_dmp_model_initialises():
    """dmp_jit should construct with default parameters."""
    from sparsepea.models import dmp_jit

    model = dmp_jit()

    assert 0.0 < model.β < 1.0
    assert model.x_min < 1.0 < model.x_max
    assert model.n_min < model.n_max
    assert model.state_bounds.shape == (4,)


def test_rbc_conditional_density_integrates():
    """AR(1) conditional density should approximately integrate to 1."""
    from sparsepea.models import rbc_jit

    model = rbc_jit()
    z_grid = np.linspace(model.z_min + 1e-6, model.z_max - 1e-6, 500)
    z_current = 1.0

    densities = np.array([model.ar1_conditional_density(y, z_current) for y in z_grid])
    integral = np.trapz(densities, z_grid)

    assert abs(integral - 1.0) < 0.05, f"Density integral = {integral:.4f}, expected ≈ 1"


def test_rbc_solver_converges_with_low_residuals():
    """End-to-end RBC solve on a coarse grid: convergence + Euler residual quality.

    Uses depth=5 to keep CI runtime under ~1s while still triggering the full
    PEA iteration path (x_axis_grid, euler_interpolation, rhs_euler, residuals).
    """
    from sparsepea.models import rbc_jit
    from sparsepea.tools import tools

    model = rbc_jit()
    solver = tools(model=model, depth=5, max_iter=400, tol=1e-5)
    _, state_points = solver.make_states_grid()
    e0 = _rbc_initial_guess(state_points)

    e, policy, μ, status = solver.compute_solution(e0)

    assert status.startswith("Convergence successful"), f"Solver did not converge: {status}"
    assert np.all(np.isfinite(e)), "Converged expectation contains non-finite values"
    assert np.all(policy > 0), "RBC consumption policy must be strictly positive"
    assert np.all(μ >= 0), "KKT multiplier must be non-negative"

    error = solver.compute_residuals(e0)
    median_error = float(np.median(error[np.isfinite(error)]))
    assert median_error < 1e-2, f"Median Euler residual {median_error:.2e} exceeds 1e-2"


def test_compute_solution_handles_non_convergence_gracefully():
    """If max_iter is exhausted, the solver should return a status string
       rather than raising UnboundLocalError."""
    from sparsepea.models import rbc_jit
    from sparsepea.tools import tools

    model = rbc_jit()
    solver = tools(model=model, depth=5, max_iter=1, tol=1e-20)
    _, state_points = solver.make_states_grid()
    e0 = _rbc_initial_guess(state_points)

    e, policy, μ, status = solver.compute_solution(e0)
    assert "Did not converge" in status
