"""
# cython: linetrace=True
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

from libc.math cimport sin, cos, exp, pi

__all__ = ["h_ml_sum_cy", "blackbody", "integrate_planck",
           "integrated_blackbody"]

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float mu(float theta) nogil:
    r"""
    Angle :math:`\mu = \cos(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    """
    return cos(theta)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float tilda_mu(float theta, float alpha) nogil:
    r"""
    The normalized quantity
    :math:`\tilde{\mu} = \alpha \mu(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`
    """
    return alpha * mu(theta)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float H(int l, float theta, float alpha) nogil:
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
        return 1
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
                tilda_mu(theta, alpha) ** 3 + 120)
    elif l == 6:
        return (64 * tilda_mu(theta, alpha) ** 6 - 480 *
                tilda_mu(theta, alpha) ** 4 + 720 *
                tilda_mu(theta, alpha) ** 2 - 120)
    elif l == 7:
        return (128 * tilda_mu(theta, alpha) ** 7 -
                1344 * tilda_mu(theta, alpha) ** 5 +
                3360 * tilda_mu(theta, alpha) ** 3 -
                1680 * tilda_mu(theta, alpha))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float h_ml_cython(float omega_drag, float alpha, int m, int l, float theta,
                       float phi, float C) nogil:
    r"""
    The :math:`h_{m\ell}` basis function.

    Parameters
    ----------
    omega_drag : float
        Dimensionless drag
    alpha : float
        Dimensionless fluid number
    m : int
        Spherical harmonic ``m`` index
    l : int
        Spherical harmonic ``l`` index
    theta : `~numpy.ndarray`
        Latitudinal coordinate
    phi : `~numpy.ndarray`
        Longitudinal coordinate
    C_ml : list
        Spherical harmonic coefficients

    Returns
    -------
    hml : `~numpy.ndarray`
        :math:`h_{m\ell}` basis function.
    """
    cdef float prefactor, result = 0

    if m == 0:
        return result

    prefactor = (C /
                 (omega_drag ** 2 * alpha ** 4 + m ** 2) *
                 exp(-tilda_mu(theta, alpha) ** 2 / 2))

    result = prefactor * (mu(theta) * m * H(l, theta, alpha) * cos(m * phi) +
                          alpha * omega_drag * (tilda_mu(theta, alpha) *
                                                H(l, theta, alpha) -
                                                H(l + 1, theta, alpha)) *
                          sin(m * phi))
    return result

@cython.boundscheck(False)
def h_ml_sum_cy(float hotspot_offset, float omega_drag, float alpha,
                double [:, :] theta2d, double [:, :] phi2d, list C, int lmax):
    """
    Cythonized implementation of the quadruple loop over: theta's, phi's,
    l's and m's to compute the h_ml_sum term at C speeds
    """
    cdef Py_ssize_t theta_max = theta2d.shape[1]
    cdef Py_ssize_t phi_max = phi2d.shape[0]
    cdef Py_ssize_t l, m, i, j
    cdef float Cml, phase_offset = pi / 2
    cdef DTYPE_t tmp
    hml_sum = np.zeros((theta_max, phi_max), dtype=DTYPE)
    cdef double [:, ::1] h_ml_sum_view = hml_sum

    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Cml = C[l][m]
            if Cml != 0:
                for i in prange(phi_max, nogil=True):
                    for j in range(theta_max):
                        tmp = h_ml_cython(omega_drag, alpha,
                                          m, l, theta2d[i, j],
                                          phi2d[i, j] +
                                          phase_offset +
                                          hotspot_offset,
                                          Cml)
                        h_ml_sum_view[j, i] = h_ml_sum_view[j, i] + tmp
    return hml_sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float blackbody_lambda(float lam, float temperature) nogil:
    """
    Compute the blackbody flux as a function of wavelength `lam` in mks units
    """
    cdef float h = 6.62607015e-34  # J s
    cdef float c = 299792458.0  # m/s
    cdef float k_B = 1.380649e-23  # J/K

    return (2 * h * c**2 / lam**5 /
            (exp(h * c / (lam * k_B * temperature)) - 1))

def bl_test(float lam, float temperature):
    return blackbody_lambda(lam, temperature)

@cython.boundscheck(False)
@cython.wraparound(False)
def blackbody(double [:] wavelengths, float temperature):
    """
    Planck function evaluated for a vector of wavelengths in units of meters
    and temperature in units of Kelvin

    Parameters
    ----------
    wavelengths : `~numpy.ndarray`
        Wavelength array in units of meters
    temperature : float
        Temperature in units of Kelvin

    Returns
    -------
    pl : `~numpy.ndarray`
        Planck function evaluated at each wavelength
    """
    cdef Py_ssize_t i, n=len(wavelengths)
    planck = np.zeros(n, dtype=DTYPE)
    cdef double [::1] planck_view = planck

    for i in prange(n, nogil=True):
        planck_view[i] = blackbody_lambda(wavelengths[i], temperature)

    return planck

@cython.boundscheck(False)
@cython.wraparound(False)
cdef blackbody2d(double [:] wavelengths, double [:, :] temperature):
    """
    Planck function evaluated for a vector of wavelengths in units of meters
    and temperature in units of Kelvin

    Parameters
    ----------
    wavelengths : `~numpy.ndarray`
        Wavelength array in units of meters
    temperature : `~numpy.ndarray`
        Temperature in units of Kelvin

    Returns
    -------
    pl : `~numpy.ndarray`
        Planck function evaluated at each wavelength
    """
    cdef int i, j, k, l=temperature.shape[0], m=temperature.shape[1], n=len(wavelengths)
    cdef np.ndarray[DTYPE_t, ndim=3] planck = np.zeros((n, l, m), dtype=DTYPE)
    cdef double [:, :, :] planck_view = planck

    for i in prange(n, nogil=True):
        for j in range(l):
            for k in range(m):
                planck_view[i, j, k] = blackbody_lambda(wavelengths[i],
                                                        temperature[j, k])

    return planck

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def trapz(double [:] y, double [:] x):
    """
    Pure cython version of trapezoid rule
    """
    cdef Py_ssize_t i, n = len(x)
    cdef float s = 0

    for i in range(1, n):
        s += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef trapz3d(double [:, :, :] y_3d, double [:] x):
    """
    Pure cython version of trapezoid rule in ~more dimensions~
    """
    cdef int i, j, k, l = len(x), m = y_3d.shape[1], n = y_3d.shape[2]

    s = np.zeros((m, n), dtype=DTYPE)
    cdef double [:, ::1] s_view = s

    for i in prange(1, l, nogil=True):
        for k in range(m):
            for j in range(n):
                s_view[k, j] += ((x[i] - x[i-1]) *
                                 (y_3d[i, k, j] + y_3d[i-1, k, j]) / 2)
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int argmin_lowest(double [:] arr, float value):
    """
    Return the index of `arr` which is closest to *and less than* `value`
    """
    cdef int i, min_ind = 0, n = len(arr)
    cdef float dist, min_dist = 1e10

    for i in range(n):
        dist = abs(arr[i] - value)
        if dist < min_dist and value > arr[i]:
            min_dist = dist
            min_ind = i

    return min_ind

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int argmin(double [:] arr, float value):
    """
    Return the index of `arr` which is closest to `value`
    """
    cdef int i, min_ind = 0, n = len(arr)
    cdef float dist, min_dist = 1e10

    for i in range(n):
        dist = abs(arr[i] - value)
        if dist < min_dist:
            min_dist = dist
            min_ind = i

    return min_ind

def argmin_test(arr, value):
    return argmin(arr, value)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float bilinear_interpolate(double [:, :] im,
                                double [:] x_grid, double [:] y_grid,
                                float x, float y):
    """
    Bilinear interpolation over an image `im` which is computed on grid
    `x_grid` vs `y_grid`, evaluated at position (x, y).

    Source: https://stackoverflow.com/a/12729229/1340208
    """
    cdef int minind0, minind1
    cdef float x0, x1, y0, y1
    cdef float Ia, Ib, Ic, Id, wa, wb, wc, wd
    minind0 = min([argmin_lowest(x_grid, x), len(x_grid) - 1])
    minind1 = min([argmin_lowest(y_grid, y), len(y_grid) - 1])

    x0 = x_grid[minind0]
    x1 = x_grid[minind0 + 1]
    y0 = y_grid[minind1]
    y1 = y_grid[minind1 + 1]

    Ia = im[minind0, minind1]
    Ib = im[minind0, minind1 + 1]
    Ic = im[minind0 + 1, minind1]
    Id = im[minind0 + 1, minind1 + 1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (wa*Ia + wb*Ib + wc*Ic + wd*Id) / ((x1 - x0) * (y1 - y0))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def integrate_planck(double [:] filt_wavelength, double [:] filt_trans,
                     double [:, :] temperature, double [:, :] T_s,
                     double [:] theta_grid, double [:] phi_grid, float rp_rs,
                     int n_phi, bint return_interp=True):
    """
    Integrate the Planck function over wavelength for the temperature map of the
    planet `temperature` and the temperature of the host star `T_s`. If
    `return_interp`, returns the interpolation function for the integral over
    the ratio of the blackbodies over wavelength; else returns only the map
    (which can be used for trapezoidal approximation integration)
    """

    cdef int i, j, k
    cdef Py_ssize_t l = len(filt_wavelength), m = len(theta_grid), n = len(phi_grid)
    cdef np.ndarray[DTYPE_t, ndim=3] bb = np.zeros((l, m, n), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=3] bb_num = blackbody2d(filt_wavelength, temperature)
    cdef np.ndarray[DTYPE_t, ndim=3] bb_den = blackbody2d(filt_wavelength, T_s)
    cdef np.ndarray[DTYPE_t, ndim=3] broadcast_trans = filt_trans[:, None, None] * np.ones((l, m, n), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] int_bb_num = trapz3d(bb_num * broadcast_trans, filt_wavelength)
    cdef np.ndarray[DTYPE_t, ndim=2] int_bb_den = trapz3d(bb_den * broadcast_trans, filt_wavelength)
    cdef np.ndarray[DTYPE_t, ndim=2] int_bb = int_bb_num / int_bb_den

    if return_interp:
        def interp(theta, phi, theta_grid=theta_grid, phi_grid=phi_grid,
                   int_bb=int_bb):
            return bilinear_interpolate(int_bb, theta_grid, phi_grid,
                                        theta, phi)
        return int_bb, interp
    else:
        return int_bb

@cython.boundscheck(False)
@cython.wraparound(False)
def integrated_blackbody(float hotspot_offset, float omega_drag,
                         float alpha, list C_ml,
                         int lmax, float T_s, float a_rs, float rp_a, float A_B,
                         int n_theta, int n_phi, double [:] filt_wavelength,
                         double [:] filt_transmittance, float f=2**-0.5):
    """
    Compute the temperature field using `h_ml_sum_cy`, then integrate the
    temperature map over wavelength and take the ratio of blackbodies with
    `integrate_planck`
    """
    cdef float T_eq, rp_rs
    cdef np.ndarray[DTYPE_t, ndim=1] phi = np.linspace(-2 * pi, 2 * pi, n_phi, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] theta = np.linspace(0, pi, n_theta, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] theta2d = np.zeros((n_theta, n_phi), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] phi2d = np.zeros((n_theta, n_phi), dtype=DTYPE)

    theta2d, phi2d = np.meshgrid(theta, phi)

    # Cython alternative to the pure python implementation:
    cdef np.ndarray[DTYPE_t, ndim=2] h_ml_sum = h_ml_sum_cy(hotspot_offset,
                                                            omega_drag,
                                                            alpha, theta2d,
                                                            phi2d, C_ml, lmax)
    T_eq = f * T_s * a_rs**-0.5

    cdef np.ndarray[DTYPE_t, ndim=2] T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs

    int_bb, func = integrate_planck(filt_wavelength,
                                    filt_transmittance, T,
                                    T_s * np.ones_like(T),
                                    theta, phi, rp_rs, n_phi)

    return int_bb, func

@cython.boundscheck(False)
@cython.wraparound(False)
def phase_curve(double [:] xi, float hotspot_offset, float omega_drag,
                float alpha, list C_ml,
                int lmax, float T_s, float a_rs, float rp_a, float A_B,
                int n_theta, int n_phi, double [:] filt_wavelength,
                double [:] filt_transmittance, float f=2**-0.5):
    """
    Compute the phase curve evaluated at phases `xi`.
    """
    cdef float T_eq, rp_rs
    cdef DTYPE_t integral
    cdef int k, phi_min, phi_max, n_xi = len(xi)
    cdef np.ndarray[DTYPE_t, ndim=1] phi = np.linspace(-2 * pi, 2 * pi, n_phi, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] theta = np.linspace(0, pi, n_theta, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] fluxes = np.zeros(n_xi, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] theta2d = np.zeros((n_theta, n_phi), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] phi2d = np.zeros((n_theta, n_phi), dtype=DTYPE)

    theta2d, phi2d = np.meshgrid(theta, phi)

    cdef double [::1] fluxes_view = fluxes

    # Cython alternative to the pure python implementation:
    h_ml_sum = h_ml_sum_cy(hotspot_offset, omega_drag,
                           alpha, theta2d, phi2d, C_ml,
                           lmax)
    T_eq = f * T_s * a_rs**-0.5

    T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs
    cdef np.ndarray[DTYPE_t, ndim=2] ones = np.ones((n_theta, n_phi), dtype=DTYPE)
    int_bb = integrate_planck(filt_wavelength,
                              filt_transmittance, T,
                              T_s * ones,
                              theta, phi, rp_rs, n_phi,
                              return_interp=False).T
    cdef double [:, :] int_bb_view = int_bb
    cdef double [:] xi_view = xi

    for k in range(n_xi):
        phi_min = argmin(phi, -xi_view[k] - pi/2)
        phi_max = argmin(phi, -xi_view[k] + pi/2)
        integral = trapz2d((int_bb_view[phi_min:phi_max] *
                           sinsq_2d(theta2d[phi_min:phi_max]) *
                           cos_2d(phi2d[phi_min:phi_max] + xi_view[k])),
                           phi[phi_min:phi_max], theta)
        # integral = trapz2d(int_bb_view[phi_min:phi_max],
        #                    phi[phi_min:phi_max], theta)

        fluxes_view[k] = integral * rp_rs**2 / pi
    return fluxes

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum2d(double [:, :] z):
    """
    Sum a 2d array over its axes
    """
    cdef int m = z.shape[0], n = z.shape[1]
    cdef float s = 0

    for i in range(m):
        for j in range(n):
            s += z[i, j]
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum1d(double [:] z):
    """
    Sum a 1d array over its first axis
    """
    cdef int m = z.shape[0]
    cdef float s = 0

    for i in range(m):
        s += z[i]
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef sinsq_2d(double [:, :] z):
    """
    The square of the sine of a 2d array
    """
    cdef int m = z.shape[0], n = z.shape[1]
    cdef np.ndarray s = np.zeros((m, n), dtype=DTYPE)
    cdef double [:, :] s_view = s

    for i in range(m):
        for j in range(n):
            s_view[i, j] = sin(z[i, j])**2
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cos_2d(double [:, :] z):
    """
    The cosine of a 2d array
    """
    cdef int m = z.shape[0], n = z.shape[1]
    cdef np.ndarray s = np.zeros((m, n), dtype=DTYPE)
    cdef double [:, :] s_view = s

    for i in range(m):
        for j in range(n):
            s_view[i, j] = cos(z[i, j])
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float trapz2d(double [:, :] z, double [:] x, double [:] y):
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
    cdef float dx, dy, s1, s2, s3
    cdef int m = z.shape[0] - 1, n = z.shape[1] - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    s1 = z[0, 0] + z[m, 0] + z[0, n] + z[m, n]
    s2 = sum1d(z[1:m, 0]) + sum1d(z[1:m, n]) + sum1d(z[0, 1:n]) + sum1d(z[m, 1:n])
    s3 = sum2d(z[1:m, 1:n])

    return 0.25 * dx * dy * (s1 + 2 * s2 + 4 * s3)

def trapz2d_test(double [:,:] z , double [:] x, double [:] y):
    return trapz2d(z, x, y)