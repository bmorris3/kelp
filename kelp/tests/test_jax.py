import numpy as np
import pytest
import pymc3 as pm
import pymc3_ext as pmx

from ..jax import reflected_phase_curve as refl_jax
from ..theano import reflected_phase_curve as refl_theano

@pytest.mark.parametrize("omega, delta_phi",
                         ((0.3, 0),
                          (0.1, -0.1),
                          (0.05, -0.2)))
def compare_jax_vs_theano_reflected(omega, delta_phi):
    args = (np.linspace(0, 1, 100), omega, delta_phi, 100)
    jax_pc = refl_jax(*args)[0]

    with pm.Model() as model:
        theano_pc = pmx.eval_in_model(refl_theano(*args)[0])

    np.testing.assert_allclose(jax_pc, theano_pc)
