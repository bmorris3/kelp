"""
# cython: linetrace=True
"""

import numpy as np
from numpy import pi
import theano.tensor as tt


def linspace(start, stop, n):
    dx = (stop-start) / (n-1)
    return tt.arange(start, stop + dx, dx)


def mu(theta):
    r"""
    Angle :math:`\mu = \cos(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    """
    return tt.cos(theta)


def tilda_mu(theta, alpha):
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


def H(l, theta, alpha):
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
        return tt.ones_like(theta)
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
    else: 
        return None


def h_ml(omega_drag, alpha, m, l, theta, phi, C):
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
    # result = 0

    if m == 0:
        return tt.zeros_like(theta)

    prefactor = (C /
                 (omega_drag ** 2 * alpha ** 4 + m ** 2) *
                 tt.exp(-tilda_mu(theta, alpha) ** 2 / 2))

    result = prefactor * (mu(theta) * m * H(l, theta, alpha) * tt.cos(m * phi) +
                          alpha * omega_drag * (tilda_mu(theta, alpha) *
                                                H(l, theta, alpha) -
                                                H(l + 1, theta, alpha)) *
                          tt.sin(m * phi))

    return result

def h_ml_sum_cy(hotspot_offset, omega_drag, alpha,
                theta2d, phi2d, C, lmax):
    """
    Cythonized implementation of the quadruple loop over: theta's, phi's,
    l's and m's to compute the h_ml_sum term at C speeds
    """
    # theta_max = theta2d.shape[1]
    # phi_max = phi2d.shape[0]
    phase_offset = pi / 2

    if lmax != 1:
        raise NotImplementedError()

    hml_sum = h_ml(omega_drag, alpha,
                      1, 1, theta2d,
                      phi2d +
                      phase_offset +
                      hotspot_offset,
                      C[1][1])
    # for l in range(1, lmax + 1):
    #     for m in range(-l, l + 1):
    #         Cml = C[l][m]
    #         if Cml != 0:
    #             hml_sum += h_ml(omega_drag, alpha,
    #                               m, l, theta2d,
    #                               phi2d +
    #                               phase_offset +
    #                               hotspot_offset,
    #                               Cml)
    return hml_sum


def blackbody_lambda(lam, temperature):
    """
    Compute the blackbody flux as a function of wavelength `lam` in mks units
    """
    h = 6.62607015e-34  # J s
    c = 299792458.0  # m/s
    k_B = 1.380649e-23  # J/K

    return (2 * h * c**2 / lam**5 /
            tt.expm1(h * c / (lam * k_B * temperature)))


def blackbody(wavelengths, temperature):
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
    n=len(wavelengths)
    planck = tt.zeros(n)
    planck_view = planck

    for i in range(n):
        planck_view[i] = blackbody_lambda(wavelengths[i], temperature)

    return planck


def blackbody2d(wavelengths, temperature):
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
    # l = temperature.shape[0]
    # m = temperature.shape[1]
    # n = len(wavelengths)

    return blackbody_lambda(wavelengths, temperature)


def trapz(y, x):
    """
    Pure cython version of trapezoid rule
    """
    n = len(x)
    s = 0

    for i in range(1, n):
        s += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    return s


def trapz3d(y_3d, x):
    """
    Pure cython version of trapezoid rule in ~more dimensions~
    """
    # l = x.shape[0]
    # m = y_3d.shape[1]
    # n = y_3d.shape[2]

    # s = tt.zeros((m, n))

    # for i in range(1, l):
    #     for k in range(m):
    #         for j in range(n):
    #             s[k, j] += ((x[i] - x[i-1]) *
    #                         (y_3d[i, k, j] + y_3d[i-1, k, j]) / 2)
    s = ((x[..., 1:] - x[..., :-1]) * (y_3d[..., 1:] + y_3d[..., :-1]) / 2)
    return tt.sum(s, axis=-1)

# def argmin_lowest(arr, value):
#     """
#     Return the index of `arr` which is closest to *and less than* `value`
#     """
#     min_ind = 0
#     n = len(arr)
#     min_dist = 1e10
# 
#     for i in range(n):
#         dist = abs(arr[i] - value)
#         if dist < min_dist and value > arr[i]:
#             min_dist = dist
#             min_ind = i
# 
#     return min_ind
# 
# def argmin(arr, value):
#     """
#     Return the index of `arr` which is closest to `value`
#     """
#     min_ind = 0
#     n = len(arr)
#     min_dist = 1e10
# 
#     for i in range(n):
#         dist = abs(arr[i] - value)
#         if dist < min_dist:
#             min_dist = dist
#             min_ind = i
# 
#     return min_ind


def bilinear_interpolate(im, x_grid, y_grid, x, y):
    """
    Bilinear interpolation over an image `im` which is computed on grid
    `x_grid` vs `y_grid`, evaluated at position (x, y).

    Source: https://stackoverflow.com/a/12729229/1340208
    """
    minind0 = tt.min([tt.argmin(tt.abs_(x_grid - x)), len(x_grid) - 1])
    minind1 = tt.min([tt.argmin(tt.abs_(y_grid - y)), len(y_grid) - 1])

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


def integrate_planck(filt_wavelength, filt_trans,
                     temperature, T_s,
                     theta_grid, phi_grid, rp_rs,
                     return_interp=True):
    """
    Integrate the Planck function over wavelength for the temperature map of the
    planet `temperature` and the temperature of the host star `T_s`. If
    `return_interp`, returns the interpolation function for the integral over
    the ratio of the blackbodies over wavelength; else returns only the map
    (which can be used for trapezoidal approximation integration)
    """

    # l = filt_wavelength.shape[3]
    # m, n = theta_grid.shape[0], phi_grid.shape[1]

    bb_num = blackbody2d(filt_wavelength, temperature)
    bb_den = blackbody2d(filt_wavelength, T_s)
    int_bb_num = trapz3d(bb_num, filt_wavelength)
    int_bb_den = trapz3d(bb_den, filt_wavelength)
    int_bb = int_bb_num / int_bb_den

    return int_bb

def integrated_blackbody(hotspot_offset, omega_drag,
                         alpha, C_ml,
                         lmax, T_s, a_rs, rp_a, A_B,
                         theta2d, phi2d, filt_wavelength,
                         filt_transmittance, f=2**-0.5):
    """
    Compute the temperature field using `h_ml_sum_cy`, then integrate the
    temperature map over wavelength and take the ratio of blackbodies with
    `integrate_planck`
    """
    # Cython alternative to the pure python implementation:
    h_ml_sum = h_ml_sum_cy(hotspot_offset, omega_drag, alpha, theta2d,
                           phi2d, C_ml, lmax)
    T_eq = f * T_s * a_rs**-0.5

    T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs

    int_bb, func = integrate_planck(filt_wavelength,
                                    filt_transmittance, T,
                                    T_s * tt.ones_like(T),
                                    theta2d, phi2d, rp_rs)

    return int_bb, func


def phase_curve(xi, hotspot_offset, omega_drag,
                alpha, C_ml,
                lmax, T_s, a_rs, rp_a, A_B,
                n_theta, n_phi, filt_wavelength,
                filt_transmittance, f=2**-0.5):
    """
    Compute the phase curve evaluated at phases `xi`.
    """
    n_xi = len(xi)
    phi = linspace(-2 * pi, 2 * pi, n_phi)
    theta = linspace(0, pi, n_theta)
    fluxes = tt.zeros(n_xi)

    theta2d, phi2d = np.meshgrid(theta, phi)

    # Cython alternative to the pure python implementation:
    h_ml_sum = h_ml_sum_cy(hotspot_offset, omega_drag,
                           alpha, theta2d, phi2d, C_ml,
                           lmax)
    T_eq = f * T_s * a_rs**-0.5

    T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs
    ones = tt.ones((n_theta, n_phi))
    int_bb = integrate_planck(filt_wavelength,
                              filt_transmittance, T,
                              T_s * ones,
                              theta, phi, rp_rs, n_phi,
                              return_interp=False).T

    for k in range(n_xi):
        phi_min = tt.argmin(tt.abs_(phi - (-xi[k] - pi/2)))
        phi_max = tt.argmin(tt.abs_(phi - (-xi[k] + pi/2)))
        integral = trapz2d((int_bb[phi_min:phi_max] *
                           sinsq_2d(theta2d[phi_min:phi_max]) *
                           cos_2d(phi2d[phi_min:phi_max] + xi[k])),
                           phi[phi_min:phi_max], theta)

        fluxes[k] = integral * rp_rs**2 / pi
    return fluxes


def phase_curve_tt(xi, hotspot_offset, omega_drag,
                    alpha, C_ml,
                    lmax, T_s, a_rs, rp_a, A_B,
                    theta2d, phi2d, filt_wavelength,
                    filt_transmittance, f=2**-0.5):
    """
    Compute the phase curve evaluated at phases `xi`.
    """
    # Cython alternative to the pure python implementation:
    h_ml_sum = h_ml_sum_cy(hotspot_offset, omega_drag,
                           alpha, theta2d, phi2d, C_ml,
                           lmax)
    T_eq = f * T_s * a_rs**-0.5

    T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs
    int_bb = integrate_planck(filt_wavelength,
                              filt_transmittance, T,
                              T_s * tt.ones_like(T),
                              theta2d, phi2d, rp_rs)
    phi = phi2d[..., 0]
    visible = ((phi > - xi[..., 0] - pi/2) &
               (phi < - xi[..., 0] + pi/2))

    integrand = (int_bb *
                 sinsq_2d(theta2d[..., 0]) *
                 cos_2d(phi2d[..., 0] + xi[..., 0]))

    integral = trapz2d(tt.where(visible, integrand, 0),
                       phi2d[:, 0, 0, 0],
                       theta2d[0, :, 0, 0])

    fluxes = integral * rp_rs**2
    # return int_bb, visible, sinsq_2d(theta2d[..., 0]), cos_2d(phi2d[..., 0] + xi[..., 0])
    return fluxes

def sum2d(z):
    """
    Sum a 2d array over its axes
    """
    return tt.sum(z)


def sum1d(z):
    """
    Sum a 1d array over its first axis
    """
    return tt.sum(z)


def sinsq_2d(z):
    """
    The square of the sine of a 2d array
    """
    return tt.sin(z)**2



def cos_2d(z):
    """
    The cosine of a 2d array
    """
    return tt.cos(z)



def trapz2d(z, x, y):
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
    m = z.shape[0] - 1
    n = z.shape[1] - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    s1 = z[0, 0, :] + z[m, 0, :] + z[0, n, :] + z[m, n, :]
    s2 = (tt.sum(z[1:m, 0, :], axis=0) + tt.sum(z[1:m, n, :], axis=0) +
          tt.sum(z[0, 1:n, :], axis=0) + tt.sum(z[m, 1:n, :], axis=0))
    s3 = tt.sum(tt.sum(z[1:m, 1:n, :], axis=0), axis=0)

    return 0.25 * dx * dy * (s1 + 2 * s2 + 4 * s3)
