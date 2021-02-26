import numpy as np
from numpy import pi as pi64
import theano.tensor as tt
from theano import config

__all__ = ['phase_curve']

floatX = config.floatX
pi = np.cast[floatX](pi64)

h = np.cast[floatX](6.62607015e-34)  # J s
c = np.cast[floatX](299792458.0)  # m/s
k_B = np.cast[floatX](1.380649e-23)  # J/K
hc2 = np.cast[floatX](6.62607015e-34 * 299792458.0 ** 2)

zero = np.cast[floatX](0)
one = np.cast[floatX](1)
two = np.cast[floatX](2)
half = np.cast[floatX](0.5)


def linspace(start, stop, n):
    dx = (stop - start) / (n - 1)
    return tt.arange(start, stop + dx, dx, dtype=floatX)


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
        return 1
    elif l == 1:
        return two * tilda_mu(theta, alpha)
    elif l == 2:
        return (two + two) * tilda_mu(theta, alpha) ** 2 - two
    else:
        raise NotImplementedError()


def h_ml(omega_drag, alpha, theta, phi, C_11, m=one, l=one):
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
    C_11 : float
        Spherical harmonic coefficient

    Returns
    -------
    hml : `~numpy.ndarray`
        :math:`h_{m\ell}` basis function.
    """
    prefactor = (C_11 /
                 (tt.pow(omega_drag, two) *
                  tt.pow(alpha, two * two) +
                  tt.pow(m, two)) *
                 tt.exp(-tt.pow(tilda_mu(theta, alpha), two) * half))

    result = prefactor * (mu(theta) * m * H(l, theta, alpha) * tt.cos(m * phi) +
                          alpha * omega_drag * (tilda_mu(theta, alpha) *
                                                H(l, theta, alpha) -
                                                H(l + one, theta, alpha)) *
                          tt.sin(m * phi))
    return result


def h_ml_sum(hotspot_offset, omega_drag, alpha,
                theta2d, phi2d, C_11):
    """
    Cythonized implementation of the quadruple loop over: theta's, phi's,
    l's and m's to compute the h_ml_sum term at C speeds
    """
    phase_offset = half * pi

    hml_sum = h_ml(omega_drag, alpha,
                   theta2d,
                   phi2d +
                   phase_offset +
                   hotspot_offset,
                   C_11)

    return hml_sum


def blackbody_lambda(lam, temperature):
    """
    Compute the blackbody flux as a function of wavelength `lam` in mks units
    """
    return (two * hc2 / tt.pow(lam, 5) /
            tt.expm1(h * c / (lam * k_B * temperature)))


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

    return blackbody_lambda(wavelengths, temperature)


def trapz3d(y_3d, x):
    """
    Trapezoid rule in ~more dimensions~
    """
    s = half * ((x[..., 1:] - x[..., :-1]) * (y_3d[..., 1:] + y_3d[..., :-1]))
    return tt.sum(s, axis=-1)


def integrate_planck(filt_wavelength, filt_trans,
                     temperature):
    """
    Integrate the Planck function over wavelength for the temperature map of the
    planet `temperature` and the temperature of the host star `T_s`. If
    `return_interp`, returns the interpolation function for the integral over
    the ratio of the blackbodies over wavelength; else returns only the map
    (which can be used for trapezoidal approximation integration)
    """

    bb_num = blackbody2d(filt_wavelength, temperature)
    int_bb_num = trapz3d(bb_num * filt_trans, filt_wavelength)

    return int_bb_num


broadcaster = tt.TensorType(floatX, 4 * [True, ])


def phase_curve(xi, hotspot_offset, omega_drag,
                alpha, C_11, T_s, a_rs, rp_a, A_B,
                theta2d, phi2d, filt_wavelength,
                filt_transmittance, f):
    r"""
    Compute the phase curve evaluated at phases ``xi``.

    .. warning::

        Assumes ``xi`` is sorted, and that ``theta2d`` and ``phi2d`` are
        linearly spaced and increasing.

    Parameters
    ----------
    xi : array-like
        Orbital phase angle, must be sorted
    hotspot_offset : float
        Angle of hotspot offset [radians]
    omega_drag : float
        Dimensionless drag frequency
    alpha : float
        Dimensionless fluid number
    C_11 : float
        Spherical harmonic power in the :math:`m=1\,\ell=1` mode
    T_s : float [K]
        Stellar effective temperature
    a_rs : float
        Semimajor axis in units of stellar radii
    rp_a : float
        Planet radius normalized by the semimajor axis
    A_B : float
        Bond albedo
    theta2d : array-like
        Grid of latitude values evaluated over the surface of the sphere
    phi2d : array-like
        Grid of longitude values evaluated over the surface of the sphere
    filt_wavelength : array-like
        Filter transmittance curve wavelengths [m]
    filt_transmittance : array-like
        Filter transmittance
    f : float
        Greenhouse parameter (typically ~1/sqrt(2)).

    Returns
    -------
    fluxes : `~numpy.ndarray`
        System fluxes as a function of phase angle :math:`\xi`.

    Examples
    --------
    Users will typically create the ``theta2d`` and ``phi2d`` grids like so:

    >>> # Set resolution of grid points on sphere:
    >>> n_phi = 100
    >>> n_theta = 10
    >>> phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi, dtype=floatX)
    >>> theta = np.linspace(0, np.pi, n_theta, dtype=floatX)
    >>> theta2d, phi2d = np.meshgrid(theta, phi)
    """
    # Handle broadcasting for 4D tensors
    xi_tt = broadcaster()
    xi_tt = xi[None, None, :, None]
    theta2d_tt = broadcaster()
    theta2d_tt = theta2d[..., None, None]
    phi2d_tt = broadcaster()
    phi2d_tt = phi2d[..., None, None]
    filt_wavelength_tt = broadcaster()
    filt_wavelength_tt = filt_wavelength[None, None, None, :]
    filt_transmittance_tt = broadcaster()
    filt_transmittance_tt = filt_transmittance[None, None, None, :]

    # Cython alternative to the pure python implementation:
    h_ml_sum = h_ml_sum(hotspot_offset, omega_drag,
                           alpha, theta2d_tt, phi2d_tt, C_11)
    T_eq = f * T_s * tt.pow(a_rs, -half)

    T = T_eq * tt.pow(one - A_B, half * half) * (one + h_ml_sum)

    rp_rs = rp_a * a_rs
    int_bb = integrate_planck(filt_wavelength_tt,
                              filt_transmittance_tt, T)
    phi = phi2d_tt[..., 0]
    visible = ((phi > - xi_tt[..., 0] - pi * half) &
               (phi < - xi_tt[..., 0] + pi * half))

    integrand = (int_bb *
                 sinsq_2d(theta2d_tt[..., 0]) *
                 cos_2d(phi + xi_tt[..., 0]))

    planck_star = trapz3d(filt_transmittance *
                          blackbody_lambda(filt_wavelength, T_s),
                          filt_wavelength)

    integral = trapz2d(integrand * visible,
                       phi2d_tt[:, 0, 0, 0],
                       theta2d_tt[0, :, 0, 0])

    fluxes = integral * tt.pow(rp_rs, 2) / pi / planck_star
    return fluxes, T


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
    return tt.pow(tt.sin(z), 2)


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

    return dx * dy * (s1 + two * s2 + (two + two) * s3) / (two + two)
