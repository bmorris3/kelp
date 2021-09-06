import pytest

import numpy as np
from astropy.modeling.models import BlackBody
import astropy.units as u

from ..core import Model, tilda_mu, H, StellarSpectrum
from ..registries import Planet, Filter
from ..fast import bl_test, argmin_test, H_cython_test


def H_testing(l, theta, alpha):
    r"""
    Hermite Polynomials in :math:`\tilde{\mu}(\theta)`.

    Parameters
    ----------
    l : int
        Implemented through :math:`\ell \leq 7`.
    theta : float
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`

    Returns
    -------
    result : `~numpy.ndarray`
        Hermite Polynomial evaluated at angles :math:`\theta`.
    """
    if l == 0:
        return np.ones_like(theta)
    elif l == 1:
        return 2 * tilda_mu(theta, alpha)
    elif l == 2:
        return 4 * tilda_mu(theta, alpha) ** 2 - 2
    elif l == 3:
        return 8 * tilda_mu(theta, alpha) ** 3 - 12 * tilda_mu(theta, alpha)
    elif l == 4:
        return (16 * tilda_mu(theta, alpha) ** 4 - 48 *
                tilda_mu(theta, alpha) ** 2 + 12)
    elif l == 5:
        return (32 * tilda_mu(theta, alpha) ** 5 - 160 *
                tilda_mu(theta, alpha) ** 3 + 120 *
                tilda_mu(theta, alpha))
    elif l == 6:
        return (64 * tilda_mu(theta, alpha) ** 6 - 480 *
                tilda_mu(theta, alpha) ** 4 + 720 *
                tilda_mu(theta, alpha) ** 2 - 120)
    elif l == 7:
        return (128 * tilda_mu(theta, alpha) ** 7 -
                1344 * tilda_mu(theta, alpha) ** 5 +
                3360 * tilda_mu(theta, alpha) ** 3 -
                1680 * tilda_mu(theta, alpha))


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

    A_B, eps = m.albedo_redist(temp, theta, phi)

    assert abs(A_B - 0) < 1e-2
    assert abs(eps - 1) < 1e-2


@pytest.mark.parametrize("lmax", np.arange(1, 8, dtype=int))
def test_hermite_polynomials(lmax):
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(-2*np.pi, 2*np.pi, 150)
    theta2d, phi2d = np.meshgrid(theta, phi)

    # Compare manual implementation against scipy implementation
    np.testing.assert_allclose(
        H_testing(lmax, theta2d, 0.6),
        H(lmax, theta2d, 0.6)
    )

    # Compare cython implementation against scipy implementation
    np.testing.assert_allclose(
        [H_cython_test(lmax, th, 0.6) for th in theta],
        H(lmax, theta, 0.6), rtol=1e-5
    )


def test_stellar_spectra():
    xi = np.linspace(-np.pi, np.pi, 100)

    p = Planet.from_name("HD 189733")

    ss_bb = StellarSpectrum.from_blackbody(p.T_s)
    ss = StellarSpectrum.from_phoenix(p.T_s)

    # Compute phase curve given a blackbody model
    m = Model(
        0, 0.6, 4.5, 0, [[0], [0, 0.1, 0]], 1,
        planet=p,
        filt=Filter.from_name("IRAC 1"),
    )

    # Compute phase curve from PHOENIX model stellar spectrum
    m_phoenix = Model(
        0, 0.6, 4.5, 0, [[0], [0, 0.1, 0]], 1,
        planet=p,
        filt=Filter.from_name("IRAC 1"),
        stellar_spectrum=ss
    )

    # compute phase curve with astropy blackbody function:
    m_bb = Model(
        0, 0.6, 4.5, 0, [[0], [0, 0.1, 0]], 1,
        planet=p,
        filt=Filter.from_name("IRAC 1"),
        stellar_spectrum=ss_bb
    )

    pc = m.thermal_phase_curve(xi)
    pc_phoenix = m_phoenix.thermal_phase_curve(xi)
    pc_bb = m_bb.thermal_phase_curve(xi)

    # Check that all three are roughly equivalent:
    np.testing.assert_allclose(pc.flux, pc_bb.flux, rtol=1e-5)
    np.testing.assert_allclose(pc_phoenix.flux, pc_bb.flux, rtol=0.05)
