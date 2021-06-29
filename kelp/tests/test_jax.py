import numpy as np
import pytest
import pymc3 as pm
import pymc3_ext as pmx
import astropy.units as u

from ..registries import Planet, Filter
from ..jax import (
    reflected_phase_curve as refl_jax,
    thermal_phase_curve as therm_jax
)
from ..theano import (
    reflected_phase_curve as refl_theano,
    thermal_phase_curve as therm_theano
)


@pytest.mark.parametrize("omega, delta_phi",
                         ((0.3, 0),
                          (0.1, -0.1),
                          (0.05, -0.2)))
def test_jax_vs_theano_reflected(omega, delta_phi):
    args = (np.linspace(0, 1, 100), omega, delta_phi, 100)
    jax_pc = refl_jax(*args)[0]

    with pm.Model():
        theano_pc = pmx.eval_in_model(refl_theano(*args)[0])

    np.testing.assert_allclose(jax_pc, theano_pc)


def test_jax_vs_theano_thermal():
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
        xi, -0.8, 4.5, 0.575, 0.18, planet.T_s, planet.a, planet.rp_a, 0,
        theta2d, phi2d, filt.wavelength.to(u.m).value, filt.transmittance, f
    )

    with pm.Model():
        therm, T = therm_theano(
            xi, -0.8, 4.5, 0.575, 0.18, planet.T_s, planet.a, planet.rp_a, 0,
            theta2d, phi2d, filt.wavelength.to(u.m).value, filt.transmittance, f
        )

        thermal_pc_theano = pmx.eval_in_model(therm)

    np.testing.assert_allclose(thermal_pc_jax, thermal_pc_theano, atol=1e-6)
