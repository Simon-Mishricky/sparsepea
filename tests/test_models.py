"""Basic smoke tests for SparsePEA model construction."""

import numpy as np
import pytest


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
