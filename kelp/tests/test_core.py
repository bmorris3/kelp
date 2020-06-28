import numpy as np
from ..core import Model
from ..registries import Planet, Filter
import pytest


@pytest.mark.parametrize("alpha,omega_drag",
                         ((0.2, 1),
                          (0.5, 1),
                          (0.6, 1),
                          (0.6, 5),
                          (0.6, 20),
                          (0.6, 100)))
def test_cython_temperature_map(alpha, omega_drag):
    filt = Filter.from_name('IRAC 1')
    lmax = 1
    C_ml = [[0],
            [0, 1, 0]]
    model = Model(0, alpha, omega_drag, 0,
                  C_ml, lmax, 15, 1e-3, 5770, filt=filt)

    n_phi = n_theta = 25

    T0 = model.temperature_map(n_theta, n_phi, cython=False)[0]
    T1 = model.temperature_map(n_theta, n_phi, cython=True)[0]

    np.testing.assert_allclose(T0, T1, atol=1e-3)


@pytest.mark.parametrize("n_theta,n_phi,rtol",
                         ((10, 150, 1.5e-2),
                          (20, 500, 1.5e-3),
                          (50, 2000, 5e-4)))
def test_cython_phase_curve(n_theta, n_phi, rtol):
    filt = Filter.from_name('IRAC 1')
    p = Planet.from_name('HD 189733')

    lmax = 1

    C_ml = [[0],
            [0, 0.1, 0]]
    model = Model(0, 0.5, 20, 0, C_ml, lmax, planet=p, filt=filt)

    xi = np.linspace(-np.pi, np.pi, 50)
    pc0 = model.phase_curve(xi, n_theta=n_theta, n_phi=n_phi, quad=True)
    pc1 = model.phase_curve(xi, n_theta=n_theta, n_phi=n_phi, quad=False)

    np.testing.assert_allclose(pc0.flux, pc1.flux, rtol=rtol)
