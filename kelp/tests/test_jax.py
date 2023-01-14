import numpy as np
import pytest
import astropy.units as u

from ..core import Model
from ..registries import Planet, Filter
from ..jax import (
    reflected_phase_curve as refl_jax,
    thermal_phase_curve as therm_jax
)


@pytest.mark.parametrize("omega, g",
                         ((0.3, 0),
                          (0.1, -0.1),
                          (0.05, -0.2)))
def test_jax_vs_cython_reflected(omega, g):
    phases = np.linspace(0.1, 0.9, 100)
    xi = 2 * np.pi * (phases - 0.5)
    args = (phases, omega, g, 100)
    kwargs = dict(hotspot_offset=0, rp_a=args[-1]**-1, lmax=0, C_ml=[0])
    jax_pc = refl_jax(*args)[0]
    m = Model(**kwargs)
    cython_pc = m.reflected_phase_curve(xi, omega, g)[0].flux

    np.testing.assert_allclose(jax_pc, cython_pc)


@pytest.mark.parametrize("C_11, hotspot_offset",
                         ((0.3, 0.1),
                          (0.1, 0),
                          (0.05, -0.2)))
def test_jax_vs_cython_thermal(C_11, hotspot_offset):
    # These parameters have been chi-by-eye "fit" to the Spitzer/3.6 um PC
    f = 0.68
    planet = Planet.from_name('HD 189733')
    filt = Filter.from_name('IRAC 1')
    filt.bin_down(5)

    xi = np.linspace(-np.pi, np.pi, 100)
    # Set resolution of grid points on sphere:
    n_phi = 100
    n_theta = 10
    phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)
    theta2d, phi2d = np.meshgrid(theta, phi)

    thermal_pc_jax, T = therm_jax(
        xi, -0.8, 4.5, 0.575, C_11, planet.T_s, planet.a, planet.rp_a, 0,
        theta2d, phi2d, filt.wavelength.to(u.m).value, filt.transmittance, f
    )

    m = Model(-0.8, 0.575, 4.5, 0, [[0], [0, C_11, 0]], 1,
              planet.a, planet.rp_a, planet.T_s, filt=filt)
    thermal_pc_cython = 1e-6 * m.thermal_phase_curve(xi, f=f).flux

    np.testing.assert_allclose(thermal_pc_jax, thermal_pc_cython, rtol=0.01)
