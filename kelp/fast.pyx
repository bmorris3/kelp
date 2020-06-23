import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

from libc.math cimport sin, cos, exp, pi

__all__ = ["h_ml_sum_cy", "blackbody", "trapz3d", "blackbody2d",
           "bilinear_interpolate", "integrate_planck", "integrated_blackbody"]

DTYPE = np.float64

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
    cdef float result = 0

    if m == 0:
        return result

    cdef float prefactor

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
                double [:, :] theta2d, double [:, :] phi2d, list C,
                int lmax):
    """
    Cythonized implementation of the quadruple loop over: theta's, phi's,
    l's and m's to compute the h_ml_sum term at C speeds
    """
    cdef Py_ssize_t theta_max = theta2d.shape[1]
    cdef Py_ssize_t phi_max = phi2d.shape[0]
    cdef Py_ssize_t l, m, i, j
    cdef float Cml, tmp, phase_offset = pi / 2
    hml_sum = np.zeros((theta_max, phi_max), dtype=DTYPE)
    cdef double[:, ::1] h_ml_sum_view = hml_sum

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

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef bisect_right(list a, float x):
#     """Return the index where to insert item x in list a, assuming a is sorted.
#     The return value i is such that all e in a[:i] have e <= x, and all e in
#     a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
#     insert just after the rightmost x already there.
#     Optional args lo (default 0) and hi (default len(a)) bound the
#     slice of a to be searched.
#
#     Source: https://github.com/python/cpython/blob/3.8/Lib/bisect.py
#     """
#     cdef int mid, lo = 0
#     cdef int hi = len(a)
#
#     while lo < hi:
#         mid = (lo + hi) // 2
#         if x < a[mid]:
#             hi = mid
#         else:
#             lo = mid + 1
#     return lo
#
#
# cdef class Interpolate:
#     """
#     Simple pure-Python implementation of linear interpolation
#
#     Source: https://stackoverflow.com/a/56233642/1340208
#     """
#     cdef object x_list, y_list, slopes
#
#     def __cinit__(self, list x_list, list y_list):
#         # if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
#         #     raise ValueError("x_list must be in strictly ascending order!")
#         self.x_list = x_list
#         self.y_list = y_list
#         intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
#         self.slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]
#
#     cpdef double interp(self, x) except *:
#         if x == self.x_list[-1]:
#             return self.y_list[-1]
#         i = bisect_right(self.x_list, x) - 1
#         return self.y_list[i] + self.slopes[i] * (x - self.x_list[i])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float blackbody_lambda(float lam, float temperature) nogil:
    cdef float h = 6.62607015e-34  # J s
    cdef float c = 299792458.0  # m/s
    cdef float k_B = 1.380649e-23  # J/K

    return (2 * h * c**2 / lam**5 /
            (exp(h * c / (lam * k_B * temperature)) - 1))

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
    cdef double[::1] planck_view = planck

    for i in prange(n, nogil=True):
        planck_view[i] = blackbody_lambda(wavelengths[i], temperature)

    return planck

@cython.boundscheck(False)
@cython.wraparound(False)
def blackbody2d(double [:] wavelengths, double [:, :] temperature):
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
    cdef Py_ssize_t i, j, k, l=temperature.shape[0], m=temperature.shape[1]
    cdef Py_ssize_t n=len(wavelengths)
    planck = np.zeros((n, l, m), dtype=DTYPE)
    cdef double[:, :, ::1] planck_view = planck

    for i in range(n):
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
def trapz3d(double [:, :, :] y_3d, double [:] x):
    """
    Pure cython version of trapezoid rule
    """
    cdef Py_ssize_t i, j, k, l = len(x), m = y_3d.shape[1], n = y_3d.shape[2]
    # cdef float s = 0

    s = np.zeros((m, n), dtype=DTYPE)
    cdef double[:, :] s_view = s

    for k in range(m):
        for j in range(n):
            for i in range(1, l):
                s_view[k, j] = (s_view[k, j] + (x[i] - x[i-1]) *
                                (y_3d[i, k, j] + y_3d[i-1, k, j]) / 2)

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int argmin(double [:] arr, float value):

    cdef int i, min_ind = 0, n = len(arr)
    cdef float dist, min_dist = 1e10

    for i in range(n):
        dist = abs(arr[i] - value)
        if dist < min_dist:
            min_dist = dist
            min_ind = i

    return min_ind

@cython.boundscheck(False)
@cython.wraparound(False)
def bilinear_interpolate(double [:, :] im,
                         double [:] x_grid, double [:] y_grid,
                         float y, float x):
    """
    Source: https://stackoverflow.com/a/12729229/1340208
    """
    cdef int minind0, minind1
    cdef float x0, x1, y0, y1
    cdef float Ia, Ib, Ic, Id, wa, wb, wc, wd
    minind0 = min([argmin(x_grid, x), len(x_grid) - 1])
    minind1 = min([argmin(y_grid, y), len(y_grid) - 1])

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

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef float GetBilinearPixel(double [:, :] imArr, float posX, float posY):
#
#     cdef int modXi, modYi, modXiPlusOneLim, modYiPlusOneLim
#     cdef float modXf, modYf
#
#     #Get integer and fractional parts of numbers
#     modXi = int(posX)
#     modYi = int(posY)
#     modXf = posX - modXi
#     modYf = posY - modYi
#     modXiPlusOneLim = min(modXi+1, imArr.shape[1]-1)
#     modYiPlusOneLim = min(modYi+1, imArr.shape[0]-1)
#
#     #Get pixels in four corners
#     bl = imArr[modYi, modXi]
#     br = imArr[modYi, modXiPlusOneLim]
#     tl = imArr[modYiPlusOneLim, modXi]
#     tr = imArr[modYiPlusOneLim, modXiPlusOneLim]
#
#     #Calculate interpolation
#     b = modXf * br + (1. - modXf) * bl
#     t = modXf * tr + (1. - modXf) * tl
#     pxf = modYf * t + (1. - modYf) * b
#
#     return pxf+0.5
#
# def bilinear_interpolate(double [:, :] im,
#                          double [:] x_grid, double [:] y_grid,
#                          float y, float x):
#
#     return GetBilinearPixel(im, x, y)

@cython.boundscheck(False)
@cython.wraparound(False)
def integrate_planck(double [:] filt_wavelength, double [:] filt_trans,
                     double [:, :] temperature, double [:, :] T_s,
                     double [:] theta_grid, double [:] phi_grid, float rp_rs):
    # cdef Py_ssize_t i, j, l = len(theta_grid), m = len(phi_grid)

    bb_num = blackbody2d(filt_wavelength, temperature)
    bb_den = blackbody2d(filt_wavelength, T_s)

    # bb_ratio = np.zeros((l, m), dtype=DTYPE)
    # cdef double [:, :] bb_ratio_view = bb_ratio
    #
    # for i in range(l):
    #     for j in range(m):
    #         bb_ratio_view[i, j] = bb_num[i, j] / bb_den[i, j]

    bb_ratio = bb_num / bb_den

    int_bb = trapz3d(bb_ratio * filt_trans[:, None, None],
                     filt_wavelength).T

    def interp(theta, phi, theta_grid=theta_grid, phi_grid=phi_grid,
               rp_rs=rp_rs):
        return bilinear_interpolate(int_bb, phi_grid, theta_grid,
                                    theta, phi) * rp_rs ** 2
    return int_bb, interp

@cython.boundscheck(False)
@cython.wraparound(False)
def integrated_blackbody(float hotspot_offset, float omega_drag,
                         float alpha, list C_ml,
                         int lmax, float T_s, float a_rs, float rp_a, float A_B,
                         int n_theta, int n_phi, double [:] filt_wavelength,
                         double [:] filt_transmittance, float f=2**-0.5):
    cdef float T_eq, rp_rs
    phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi, dtype=DTYPE)
    theta = np.linspace(0, np.pi, n_theta, dtype=DTYPE)

    theta2d, phi2d = np.meshgrid(theta, phi)

    # Cython alternative to the pure python implementation:
    h_ml_sum = h_ml_sum_cy(hotspot_offset, omega_drag,
                           alpha, theta2d, phi2d, C_ml,
                           lmax)
    T_eq = f * T_s * a_rs**-0.5

    T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs

    int_bb, func = integrate_planck(filt_wavelength,
                                    filt_transmittance, T,
                                    T_s * np.ones_like(T),
                                    theta, phi, rp_rs)

    return int_bb, func
