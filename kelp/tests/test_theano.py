import pytest

import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import astropy.units as u

from .test_core import trapz2d
from ..core import Model
from ..registries import Planet, Filter


@pytest.mark.parametrize(
    "f, c11",
    ((0.2, 0.1),
     (0.5, 0.1),
     (0.6, 0.1),
     (0.6, 0.5),
     (0.7, 0.2),
     (0.7, 0.1))
)
def test_cython_vs_theano(f, c11):
    from ..theano.theano import thermal_phase_curve
    planet = Planet.from_name('HD 189733')
    filt = Filter.from_name('IRAC 1')
    C_ml = [[0],
            [0, c11, 0]]
    m = Model(
        -0.8, 0.575, 4.5, 0, C_ml, 1,
        planet=planet,
        filt=filt
    )
    xi = np.linspace(-np.pi, np.pi, 100)

    # Set resolution of grid points on sphere:
    n_phi = 100
    n_theta = 10
    phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)
    theta2d, phi2d = np.meshgrid(theta, phi)

    cython_phase_curve = m.thermal_phase_curve(xi, f=f).flux
    cython_temp_map, _, _ = m.temperature_map(n_theta, n_phi, f=f)

    with pm.Model():
        thermal_pc, T = thermal_phase_curve(
            xi, -0.8, 4.5, 0.575, c11, planet.T_s, planet.a, planet.rp_a, 0,
            theta2d, phi2d, filt.wavelength.to(u.m).value, filt.transmittance, f
        )

        theano_phase_curve = 1e6 * pmx.eval_in_model(thermal_pc)
        theano_map = pmx.eval_in_model(T)[..., 0, 0].T

    np.testing.assert_allclose(
        cython_phase_curve, theano_phase_curve, atol=5
    )

    np.testing.assert_allclose(
        cython_temp_map, theano_map, atol=10
    )


def test_albedo():
    from ..theano.theano import thermal_phase_curve

    f = 2**-0.5
    p = Planet.from_name('HD 189733')
    filt = Filter.from_name('IRAC 1')

    # Set resolution of grid points on sphere:
    n_phi = 100
    n_theta = 10
    phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)
    theta2d, phi2d = np.meshgrid(theta, phi)
    xi = np.linspace(-np.pi, np.pi, 100)

    with pm.Model():
        thermal_pc, T = thermal_phase_curve(
            xi, -0.8, 4.5, 0.575, 0, p.T_s, p.a, p.rp_a, 0,
            theta2d, phi2d, filt.wavelength.to(u.m).value, filt.transmittance, f
        )

        theano_map = pmx.eval_in_model(T)[..., 0, 0]

    A_B = 1 - p.a**2 * (
        trapz2d(
            theano_map[..., None]**4 * np.sin(theta2d[..., None]) *
            (phi2d[..., None] > 0), phi, theta
        ) / ( np.pi * p.T_s**4)
    )[0]

    assert abs(A_B - 0) < 5e-2
