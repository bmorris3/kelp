import numpy as np
from ..core import Model
from ..registries import Filter


def test_cython():
    filt = Filter.from_name('IRAC 1')
    lmax = 1
    C_ml = [[0],
            [0, 1, 0]]
    model = Model(0, 0.5, 20, 0, C_ml, lmax, 15, 1e-3, 5770, filt=filt)

    n_phi = n_theta = 25

    T0 = model.temperature_map(n_theta, n_phi, cython=False)[0]
    T1 = model.temperature_map(n_theta, n_phi, cython=True)[0]

    np.testing.assert_allclose(T0, T1, atol=1e-3)
