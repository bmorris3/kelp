import numpy as np
from ..core import Model
from ..registries import Planet, Filter
import pytest
from ..fast import bl_test
from ..fast import trapz
from ..fast import argmin_test
from astropy.modeling.models import BlackBody
import astropy.units as u

@pytest.mark.parametrize("y, x",
                         ((np.array([0.1e-3,0.2e-3,0.3e-3,0.4e-3,0.5e-3]), np.array([0.0,1.0,2.0,3.0,4.0])),
                          (np.array([0.1e-1,0.2e-1,0.3e-1,0.4e-1,0.5e-1]), np.array([0.0,1.0,2.0,3.0,4.0])),
                          (np.array([1.0,2.0,3.0,4.0,5.0]), np.array([0.0,1.0,2.0,3.0,4.0]))))
def test_trapz(y , x):
    kelp_test = trapz(y, x)
    check = np.trapz(y, x)

    np.testing.assert_allclose(check, kelp_test, atol=1e-5)

@pytest.mark.parametrize("wavelength, temp",
                         ((0.5e-6, 1000),
                          (0.5e-6, 500),
                          (0.5e-5, 100),
                          (0.5e-6, 10000),
                          (0.5e-5, 1000)))
def test_bl(wavelength, temp):
    kelp_test = bl_test(wavelength, temp)
    check = BlackBody(temp * u.K, scale=1 * (u.W / (u.m ** 2 * u.nm * u.sr)))(wavelength * u.m)

    np.testing.assert_allclose(check.si.value, kelp_test, rtol=1e-4)


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


@pytest.mark.parametrize("n_theta,n_phi,atol",
                         ((10, 150, 1e-3),
                          (20, 500, 3e-4),
                          (50, 2000, 5e-5)))
def test_cython_phase_curve(n_theta, n_phi, atol):
    filt = Filter.from_name('IRAC 1')
    p = Planet.from_name('HD 189733')

    lmax = 1

    C_ml = [[0],
            [0, 0.1, 0]]
    model = Model(0, 0.5, 20, 0, C_ml, lmax, planet=p, filt=filt)

    xi = np.linspace(-np.pi, np.pi, 50)
    pc0 = model.phase_curve(xi, n_theta=n_theta, n_phi=n_phi, quad=True)
    pc1 = model.phase_curve(xi, n_theta=n_theta, n_phi=n_phi, quad=False)

    np.testing.assert_allclose(pc0.flux, pc1.flux, atol=atol)

@pytest.mark.parametrize("y, x",
                         ((np.linspace(0,10,10), 0.23),
                          (np.linspace(0,10,100), 0.23),
                          (np.linspace(0,10,1000), 0.23)))
def test_argmin(y, x):
    kelp_test = argmin_test(y, x)
    check = (np.abs(y - x)).argmin()

    np.testing.assert_array_equal(check, kelp_test)