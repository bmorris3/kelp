"""
To compile: python setup.py build_ext --inplace
"""
from kelp import Model, Filter, h_ml_sum_cy
import numpy as np

filt = Filter.from_name('IRAC 1')
lmax = 1
C_ml = [[0],
        [0, 1, 0]]
model = Model(0, 0.5, 20, 0, C_ml, lmax, 15, 1e-3, 5770, filt=filt)

m = 1
l = 1
n_phi = n_theta = 20
f = 2**-0.5

theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(-2*np.pi, 2*np.pi, n_phi)

theta2d, phi2d = np.meshgrid(theta, phi)
phase_offset = np.pi / 2

import time
start0 = time.time()
for i in range(100):
    T0 = model.temperature_map(n_theta, n_phi)[0]
# hml_py = model.h_ml(m, l, theta2d, phi2d)
end0 = time.time()

start1 = time.time()
# hml_cy = h_ml_cython(model.omega_drag, model.alpha,
#                      m, l, theta2d, phi2d, C_ml[l][m])
for i in range(100):
    h_ml_sum = h_ml_sum_cy(model.hotspot_offset, model.omega_drag, model.alpha,
                           theta2d, phi2d, C_ml, lmax)
T_eq = f * model.T_s * np.sqrt(1 / model.a_rs)

T1 = T_eq * (1 - model.A_B) ** 0.25 * (1 + h_ml_sum)
end1 = time.time()


print((end0-start0) / (end1-start1))


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].pcolormesh(phi, theta, T0)
ax[1].pcolormesh(phi, theta, T1)
plt.show()

np.testing.assert_allclose(T0, T1, atol=0.01)

# np.testing.assert_allclose(hml_py, hml_cy, atol=1e-6)

