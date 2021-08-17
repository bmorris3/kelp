import pytest

import numpy as np
from astropy.modeling.models import BlackBody
import astropy.units as u

from ..core import Model
from ..registries import Planet, Filter
from ..fast import bl_test, argmin_test


def trapz2d(z, x, y):
    """
    Integrates a regularly spaced 2D grid using the composite trapezium rule.

    Source: https://github.com/tiagopereira/python_tips/blob/master/code/trapz2d.py

    Parameters
    ----------
    z : `~numpy.ndarray`
        2D array
    x : `~numpy.ndarray`
        grid values for x (1D array)
    y : `~numpy.ndarray`
        grid values for y (1D array)

    Returns
    -------
    t : `~numpy.ndarray`
        Trapezoidal approximation to the integral under z
    """
    m = z.shape[0] - 1
    n = z.shape[1] - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    s1 = z[0, 0, :] + z[m, 0, :] + z[0, n, :] + z[m, n, :]
    s2 = (np.sum(z[1:m, 0, :], axis=0) + np.sum(z[1:m, n, :], axis=0) +
          np.sum(z[0, 1:n, :], axis=0) + np.sum(z[m, 1:n, :], axis=0))
    s3 = np.sum(np.sum(z[1:m, 1:n, :], axis=0), axis=0)
    return dx * dy * (s1 + 2 * s2 + 4 * s3) / 4


@pytest.mark.parametrize("wavelength, temp",
                         ((0.5e-6, 1000),
                          (0.5e-6, 500),
                          (0.5e-5, 100),
                          (0.5e-6, 10000),
                          (0.5e-5, 1000)))
def test_bl(wavelength, temp):
    kelp_test = bl_test(wavelength, temp)
    check = BlackBody(
        temp * u.K, scale=1 * (u.W / (u.m ** 2 * u.nm * u.sr))
    )(wavelength * u.m)

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
                         ((10, 150, 1000),
                          (20, 500, 300),
                          (50, 2000, 50)))
def test_cython_phase_curve(n_theta, n_phi, atol):
    filt = Filter.from_name('IRAC 1')
    p = Planet.from_name('HD 189733')

    lmax = 1

    C_ml = [[0],
            [0, 0.1, 0]]
    model = Model(0, 0.5, 20, 0, C_ml, lmax, planet=p, filt=filt)

    xi = np.linspace(-np.pi, np.pi, 50)
    pc0 = model.thermal_phase_curve(
        xi, n_theta=n_theta, n_phi=n_phi, quad=True, cython=False
    )
    pc1 = model.thermal_phase_curve(
        xi, n_theta=n_theta, n_phi=n_phi, quad=False, cython=False
    )

    np.testing.assert_allclose(pc0.flux, pc1.flux, atol=atol)


@pytest.mark.parametrize("y, x",
                         ((np.linspace(0, 10, 10), 0.23),
                          (np.linspace(0, 10, 100), 0.23),
                          (np.linspace(0, 10, 1000), 0.23)))
def test_argmin(y, x):
    kelp_test = argmin_test(y, x)
    check = (np.abs(y - x)).argmin()

    np.testing.assert_array_equal(check, kelp_test)


def test_integrated_temperatures():

    # These parameters have been chi-by-eye "fit" to the Spitzer/3.6 um PC
    f = 0.67

    C_ml = [[0],
            [0, 0.18, 0]]
    m = Model(
        -0.8, 0.575, 4.5, 0, C_ml, 1,
        planet=Planet.from_name('HD 189733'),
        filt=Filter.from_name('IRAC 1')
    )

    dayside, nightside = m.integrated_temperatures(f=f)

    # Check within 1 sigma of Knutson et al. 2012 dayside
    assert abs(dayside - 1328) / 11 < 1

    # Check within 1 sigma of Keating et al. 2019 nightside
    assert abs(979 - nightside) / 58 < 1


def test_albedo():
    f = 2**-0.5
    p = Planet.from_name('HD 189733')
    C_ml = [[0],
            [0, 0.0, 0]]
    m = Model(
        -0.8, 0.575, 4.5, 0, C_ml, 1,
        planet=p,
        filt=Filter.from_name('IRAC 1')
    )
    n_theta, n_phi = 15, 100
    temp, theta, phi = m.temperature_map(n_theta, n_phi, f=f)

    theta2d, phi2d = np.meshgrid(theta, phi)

    A_B = 1 - p.a**2 * (
        trapz2d(
            temp.T[..., None]**4 * np.sin(theta2d[..., None]) *
            (phi2d[..., None] > 0), phi, theta
        ) / ( np.pi * p.T_s**4)
    )[0]

    assert abs(A_B - 0) < 1e-2