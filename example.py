from kelp import Model, Filter, h_ml_cython
import numpy as np

filt = Filter.from_name('IRAC 1')
lmax = 1
C_ml = [[0],
        [0, 1, 0]]
model = Model(0, 0.5, 20, 0, C_ml, lmax, 15, 1e-3, 5770, filt=filt)

m = 1
l = 1

theta = np.linspace(0, np.pi, 500)
phi = np.linspace(-2*np.pi, 2*np.pi, 500)

theta2d, phi2d = np.meshgrid(theta, phi)
phase_offset = np.pi / 2
import time
start0 = time.time()
# T = model.temperature_map(500, 500)
hml_py = model.h_ml(m, l, theta2d, phi2d)
end0 = time.time()

start1 = time.time()
hml_cy = h_ml_cython(model.omega_drag, model.alpha,
                     m, l, theta2d, phi2d, C_ml[l][m])
end1 = time.time()

print((end1-start1) / (end0-start0))


# np.testing.assert_allclose(hml_py, hml_cy, atol=1e-6)

