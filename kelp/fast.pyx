import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

from libc.math cimport sin, cos, exp, pi

__all__ = ["h_ml_sum_cy"]

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
