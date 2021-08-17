import numpy as np
from numpy import pi as pi64
import theano.tensor as tt
from theano import config

__all__ = [
    'thermal_phase_curve',
    'reflected_phase_curve',
    'reflected_phase_curve_inhomogeneous'
]

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


def h_ml_sum_theano(hotspot_offset, omega_drag, alpha,
                    theta2d, phi2d, C_11):
    """
    Cythonized implementation of the quadruple loop over: theta's, phi's,
    l's and m's to compute the h_ml_sum term
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


def interpolate(x, x0, y0):
    idx = np.searchsorted(x0, x)
    dl = x - x0[idx - 1]
    dr = x0[idx] - x
    d = dl + dr
    wl = dr / d

    return wl * y0[idx - 1] + (1 - wl) * y0[idx]


broadcaster = tt.TensorType(floatX, 4 * [True, ])


def thermal_phase_curve(xi, hotspot_offset, omega_drag,
                        alpha, C_11, T_s, a_rs, rp_a, A_B,
                        theta2d, phi2d, filt_wavelength,
                        filt_transmittance, f,
                        stellar_spectrum_wavelength=None,
                        stellar_spectrum_spectral_flux_density=None):
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
    fluxes : tensor-like
        System fluxes as a function of phase angle :math:`\xi`.
    T : tensor-like
        Temperature map

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

    h_ml_sum = h_ml_sum_theano(hotspot_offset, omega_drag,
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

#    if tt.any(tt.neq(stellar_spectrum_spectral_flux_density, 0)):
    if stellar_spectrum_spectral_flux_density is not None:
        planck_star = trapz3d(filt_transmittance *
                              interpolate(filt_wavelength,
                                  stellar_spectrum_wavelength,
                                  stellar_spectrum_spectral_flux_density
                              ),
                              filt_wavelength)
    else:
        planck_star = trapz3d(filt_transmittance *
                              blackbody_lambda(filt_wavelength, T_s),
                              filt_wavelength) * pi

    integral = trapz2d(integrand * visible,
                       phi2d_tt[:, 0, 0, 0],
                       theta2d_tt[0, :, 0, 0])

    fluxes = integral * tt.pow(rp_rs, 2) / planck_star
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


def reflected_phase_curve(phases, omega, g, a_rp, return_q=True):
    """
    Reflected light phase curve for a homogeneous sphere by
    Heng, Morris & Kitzmann (2021).

    Parameters
    ----------
    phases : `~np.ndarray`
        Orbital phases of each observation defined on (0, 1)
    omega : tensor-like
        Single-scattering albedo as defined in
    g : tensor-like
        Scattering asymmetry factor, ranges from (-1, 1).
    a_rp : float, tensor-like
        Semimajor axis scaled by the planetary radius

    Returns
    -------
    flux_ratio_ppm : tensor-like
        Flux ratio between the reflected planetary flux and the stellar flux in
        units of ppm.
    A_g : tensor-like
        Geometric albedo derived for the planet given {omega, g}.
    q : tensor-like
        Integral phase function
    """
    # Convert orbital phase on (0, 1) to "alpha" on (0, np.pi)
    alpha = (2 * np.pi * phases - np.pi).astype(floatX)
    abs_alpha = np.abs(alpha).astype(floatX)
    alpha_sort_order = np.argsort(alpha)
    sin_abs_sort_alpha = np.sin(abs_alpha[alpha_sort_order]).astype(floatX)
    sort_alpha = alpha[alpha_sort_order].astype(floatX)

    gamma = tt.sqrt(1 - omega)
    eps = (1 - gamma) / (1 + gamma)

    # Equation 34 for Henyey-Greestein
    P_star = (1 - g ** 2) / (1 + g ** 2 +
                             2 * g * tt.cos(alpha)) ** 1.5
    # Equation 36
    P_0 = (1 - g) / (1 + g) ** 2

    # Equation 10:
    Rho_S = P_star - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_S_0 = P_0 - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_L = 0.5 * eps * (2 - eps) * (1 + eps) ** 2
    Rho_C = eps ** 2 * (1 + eps) ** 2

    alpha_plus = tt.sin(abs_alpha / 2) + tt.cos(abs_alpha / 2)
    alpha_minus = tt.sin(abs_alpha / 2) - tt.cos(abs_alpha / 2)

    # Equation 11:
    Psi_0 = tt.log((1 + alpha_minus) * (alpha_plus - 1) /
                   (1 + alpha_plus) / (1 - alpha_minus))
    Psi_S = 1 - 0.5 * (tt.cos(abs_alpha / 2) -
                       1.0 / tt.cos(abs_alpha / 2)) * Psi_0
    Psi_L = (tt.sin(abs_alpha) + (np.pi - abs_alpha) *
             tt.cos(abs_alpha)) / np.pi
    Psi_C = (-1 + 5 / 3 * tt.cos(abs_alpha / 2) ** 2 - 0.5 *
             tt.tan(abs_alpha / 2) * tt.sin(abs_alpha / 2) ** 3 * Psi_0)

    # Equation 8:
    A_g = omega / 8 * (P_0 - 1) + eps / 2 + eps ** 2 / 6 + eps ** 3 / 24

    # Equation 9:
    Psi = ((12 * Rho_S * Psi_S + 16 * Rho_L *
            Psi_L + 9 * Rho_C * Psi_C) /
           (12 * Rho_S_0 + 16 * Rho_L + 6 * Rho_C))

    flux_ratio_ppm = 1e6 * (a_rp ** -2 * A_g * Psi)

    if return_q:
        q = _integral_phase_function(
            Psi, sin_abs_sort_alpha, sort_alpha, alpha_sort_order
        )

        return flux_ratio_ppm, A_g, q
    else:
        return flux_ratio_ppm, A_g


def rho(omega, P_0, P_star):
    """
    Equation 10
    """
    gamma = tt.sqrt(1 - omega)
    eps = (1 - gamma) / (1 + gamma)

    Rho_S = P_star - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_S_0 = P_0 - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_L = 0.5 * eps * (2 - eps) * (1 + eps) ** 2
    Rho_C = eps ** 2 * (1 + eps) ** 2

    return Rho_S, Rho_S_0, Rho_L, Rho_C


def I(alpha, Phi):
    """
    Equation 39
    """
    cos_alpha = tt.cos(alpha)
    cos_alpha_2 = tt.cos(alpha / 2)

    z = tt.sin(alpha / 2 - Phi / 2) / tt.cos(Phi / 2)
    z = tt.switch(tt.abs_(z) < 1, z, tt.sgn(z) * 0.99)

    # The following expression has the same behavior
    # as I_0 = 0.5 * (tt.log1p(z) - tt.log1p(-z)), but it may blow up at alpha=0
    I_0 = tt.arctanh(z)

    I_S = (-1 / (2 * cos_alpha_2) *
           (tt.sin(alpha / 2 - Phi) +
            (cos_alpha - 1) * I_0))
    I_L = 1 / np.pi * (Phi * cos_alpha -
                       0.5 * tt.sin(alpha - 2 * Phi))
    I_C = -1 / (24 * cos_alpha_2) * (
        -3 * tt.sin(alpha / 2 - Phi) +
        tt.sin(3 * alpha / 2 - 3 * Phi) +
        6 * tt.sin(3 * alpha / 2 - Phi) -
        6 * tt.sin(alpha / 2 + Phi) +
        24 * tt.sin(alpha / 2) ** 4 * I_0
    )

    return I_S, I_L, I_C


def trapz1d(y_1d, x):
    """
    Trapezoid rule in one dimension. This only works if x is increasing.
    """
    s = 0.5 * ((x[1:] - x[:-1]) * (y_1d[1:] + y_1d[:-1]))
    return tt.sum(s, axis=-1)


def _integral_phase_function(Psi, sin_abs_sort_alpha, sort_alpha, sort):
    """
    Integral phase function q for a generic, possibly asymmetric reflectivity
    map
    """
    return trapz1d(Psi[sort] * sin_abs_sort_alpha, sort_alpha)


def _g_from_ag(A_g, omega_0, omega_prime, x1, x2):
    """
    Compute the scattering asymmetry factor g for a given geometric albedo,
    and possibly asymmetric single scattering albedos.

    Parameters
    ----------
    A_g : tensor-like
        Geometric albedo on (0, 1)
    omega_0 : tensor-like
        Single-scattering albedo of the less reflective region.
        Defined on (0, 1).
    omega_prime : tensor-like
        Additional single-scattering albedo of the more reflective region,
        such that the single-scattering albedo of the reflective region is
        ``omega_0 + omega_prime``. Defined on (0, ``1-omega_0``).
    x1 : tensor-like
        Start longitude of the darker region [radians] on (-pi/2, pi/2)
    x2 : tensor-like
        Stop longitude of the darker region [radians] on (-pi/2, pi/2)

    Returns
    -------
    g : tensor-like
        Scattering asymmetry factor
    """
    gamma = tt.sqrt(1 - omega_0)
    eps = (1 - gamma) / (1 + gamma)

    gamma_prime = tt.sqrt(1 - omega_prime)
    eps_prime = (1 - gamma_prime) / (1 + gamma_prime)

    Rho_L = eps / 2 * (1 + eps) ** 2 * (2 - eps)
    Rho_L_prime = eps_prime / 2 * (1 + eps_prime) ** 2 * (2 - eps_prime)
    Rho_C = eps ** 2 * (1 + eps) ** 2
    Rho_C_prime = eps_prime ** 2 * (1 + eps_prime) ** 2
    C = -1 + 0.25 * (1 + eps) ** 2 * (2 - eps) ** 2
    C_prime = -1 + 0.25 * (1 + eps_prime) ** 2 * (2 - eps_prime) ** 2

    C_2 = 2 + tt.sin(x1) - tt.sin(x2)
    C_1 = (omega_0 * Rho_L * np.pi / 12 + omega_prime * Rho_L_prime / 12 *
           (x1 - x2 + np.pi + 0.5 * (tt.sin(2 * x1) - tt.sin(
               2 * x2))) +
           np.pi * omega_0 * Rho_C / 32 + 3 * np.pi * omega_prime *
           Rho_C_prime / 64 *
           (2 / 3 + 3 / 8 * (tt.sin(x1) - tt.sin(x2)) +
            1 / 24 * (tt.sin(3 * x1) - tt.sin(3 * x2))))
    C_3 = (16 * np.pi * A_g - 32 * C_1 - 2 * np.pi * omega_0 * C -
           np.pi * omega_prime * C_2 * C_prime
           ) / (2 * np.pi * omega_0 + np.pi * omega_prime * C_2)

    return - ((2 * C_3 + 1) - tt.sqrt(1 + 8 * C_3)) / (2 * C_3)


def reflected_phase_curve_inhomogeneous(phases, omega_0, omega_prime, x1, x2,
                                        A_g, a_rp, return_q=True):
    """
    Reflected light phase curve for an inhomogeneous sphere by
    Heng, Morris & Kitzmann (2021), with inspiration from Hu et al. (2015).

    Parameters
    ----------
    phases : `~np.ndarray`
        Orbital phases of each observation defined on (0, 1)
    omega_0 : tensor-like
        Single-scattering albedo of the less reflective region.
        Defined on (0, 1).
    omega_prime : tensor-like
        Additional single-scattering albedo of the more reflective region,
        such that the single-scattering albedo of the reflective region is
        ``omega_0 + omega_prime``. Defined on (0, ``1-omega_0``).
    x1 : tensor-like
        Start longitude of the darker region [radians] on (-pi/2, pi/2)
    x2 : tensor-like
        Stop longitude of the darker region [radians] on (-pi/2, pi/2)
    a_rp : float, tensor-like
        Semimajor axis scaled by the planetary radius

    Returns
    -------
    flux_ratio_ppm : tensor-like
        Flux ratio between the reflected planetary flux and the stellar flux
        in units of ppm.
    g : tensor-like
        Scattering asymmetry factor on (-1, 1)
    q : tensor-like
        Integral phase function
    """

    g = _g_from_ag(A_g, omega_0, omega_prime, x1, x2)

    # Redefine alpha to be on (-pi, pi)
    alpha = (2 * np.pi * phases - np.pi).astype(floatX)
    abs_alpha = np.abs(alpha).astype(floatX)

    # Equation 34 for Henyey-Greestein
    P_star = (1 - g ** 2) / (1 + g ** 2 +
                             2 * g * tt.cos(abs_alpha)) ** 1.5
    # Equation 36
    P_0 = (1 - g) / (1 + g) ** 2

    Rho_S, Rho_S_0, Rho_L, Rho_C = rho(omega_0, P_0, P_star)

    Rho_S_prime, Rho_S_0_prime, Rho_L_prime, Rho_C_prime = rho(
        omega_prime, P_0, P_star
    )

    alpha_plus = tt.sin(abs_alpha / 2) + tt.cos(abs_alpha / 2)
    alpha_minus = tt.sin(abs_alpha / 2) - tt.cos(abs_alpha / 2)

    # Equation 11:
    Psi_0 = tt.log((1 + alpha_minus) * (alpha_plus - 1) /
                   (1 + alpha_plus) / (1 - alpha_minus))

    Psi_S = 1 - 0.5 * (tt.cos(abs_alpha / 2) -
                       1.0 / tt.cos(abs_alpha / 2)) * Psi_0
    Psi_L = (tt.sin(abs_alpha) + (np.pi - abs_alpha) *
             tt.cos(abs_alpha)) / np.pi
    Psi_C = (-1 + 5 / 3 * tt.cos(abs_alpha / 2) ** 2 -
             0.5 * tt.tan(abs_alpha / 2) *
             tt.sin(abs_alpha / 2) ** 3 * Psi_0)

    # Table 1:
    condition_a = (-np.pi / 2 <= alpha - np.pi / 2)
    condition_0 = ((alpha - np.pi / 2 <= np.pi / 2) &
                   (np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= alpha + x2))
    condition_1 = ((alpha - np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= np.pi / 2) &
                   (np.pi / 2 <= alpha + x2))
    condition_2 = ((alpha - np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= alpha + x2) &
                   (alpha + x2 <= np.pi / 2))

    condition_b = (alpha + np.pi / 2 <= np.pi / 2)
    condition_3 = ((alpha + x1 <= alpha + x2) &
                   (alpha + x2 <= -np.pi / 2) &
                   (-np.pi / 2 <= alpha + np.pi / 2))
    condition_4 = ((alpha + x1 <= -np.pi / 2) &
                   (-np.pi / 2 <= alpha + x2) &
                   (alpha + x2 <= alpha + np.pi / 2))
    condition_5 = ((-np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= alpha + x2) &
                   (alpha + x2 <= alpha + np.pi / 2))

    integration_angles = [
        [alpha - np.pi / 2, np.pi / 2],
        [alpha - np.pi / 2, alpha + x1],
        [alpha - np.pi / 2, alpha + x1, alpha + x2, np.pi / 2],
        [-np.pi / 2, alpha + np.pi / 2],
        [alpha + x2, alpha + np.pi / 2],
        [-np.pi / 2, alpha + x1, alpha + x2, alpha + np.pi / 2]
    ]

    conditions = [
        condition_a & condition_0,
        condition_a & condition_1,
        condition_a & condition_2,
        condition_b & condition_3,
        condition_b & condition_4,
        condition_b & condition_5,
    ]

    Psi_S_prime = 0
    Psi_L_prime = 0
    Psi_C_prime = 0

    for condition_i, angle_i in zip(conditions, integration_angles):
        for i, phi_i in enumerate(angle_i):
            sign = (-1) ** (i + 1)
            I_phi_S, I_phi_L, I_phi_C = I(alpha, phi_i)
            Psi_S_prime += tt.switch(condition_i, sign * I_phi_S, 0)
            Psi_L_prime += tt.switch(condition_i, sign * I_phi_L, 0)
            Psi_C_prime += tt.switch(condition_i, sign * I_phi_C, 0)

    # Compute everything for alpha=0
    angles_alpha0 = [-np.pi / 2, x1, x2, np.pi / 2]
    Psi_S_prime_alpha0 = 0
    Psi_L_prime_alpha0 = 0
    Psi_C_prime_alpha0 = 0
    for i, phi_i in enumerate(angles_alpha0):
        sign = (-1) ** (i + 1)
        I_phi_S_alpha0, I_phi_L_alpha0, I_phi_C_alpha0 = I(0, phi_i)

        Psi_S_prime_alpha0 += sign * I_phi_S_alpha0
        Psi_L_prime_alpha0 += sign * I_phi_L_alpha0
        Psi_C_prime_alpha0 += sign * I_phi_C_alpha0

    # P_star_alpha0 = (1 - g ** 2) / (1 + g ** 2 + 2 * g * 1) ** 1.5

    # Rho_S_alpha0, Rho_S_0_alpha0, Rho_L_alpha0, Rho_C_alpha0 = rho(omega_0, P_0,
    #                                                                P_star_alpha0)
    #
    # Rho_S_prime_alpha0, Rho_S_0_prime_alpha0, Rho_L_prime_alpha0, Rho_C_prime_alpha0 = rho(
    #     omega_prime, P_0, P_star_alpha0
    # )

    # # Equation 11:
    # Psi_S_alpha0 = 1
    # Psi_L_alpha0 = 1
    # Psi_C_alpha0 = (-1 + 5 / 3)

    # # Equation 37
    # F_S_alpha0 = np.pi / 16 * (omega_0 * Rho_S_alpha0 * Psi_S_alpha0 +
    #                            omega_prime * Rho_S_prime_alpha0 *
    #                            Psi_S_prime_alpha0)
    # F_L_alpha0 = np.pi / 12 * (omega_0 * Rho_L_alpha0 * Psi_L_alpha0 +
    #                            omega_prime * Rho_L_prime_alpha0 *
    #                            Psi_L_prime_alpha0)
    # F_C_alpha0 = 3 * np.pi / 64 * (omega_0 * Rho_C_alpha0 * Psi_C_alpha0 +
    #                                omega_prime * Rho_C_prime_alpha0 *
    #                                Psi_C_prime_alpha0)

    # Equation 37
    F_S = np.pi / 16 * (omega_0 * Rho_S * Psi_S +
                        omega_prime * Rho_S_prime * Psi_S_prime)
    F_L = np.pi / 12 * (omega_0 * Rho_L * Psi_L +
                        omega_prime * Rho_L_prime * Psi_L_prime)
    F_C = 3 * np.pi / 64 * (omega_0 * Rho_C * Psi_C +
                            omega_prime * Rho_C_prime * Psi_C_prime)

    sobolev_fluxes = F_S + F_L + F_C
    F_max = tt.max(sobolev_fluxes)

    Psi = sobolev_fluxes / F_max

    flux_ratio_ppm = 1e6 * a_rp**-2 * Psi * A_g

    if return_q:
        alpha_sort_order = np.argsort(alpha)
        sin_abs_sort_alpha = np.sin(abs_alpha[alpha_sort_order]).astype(floatX)
        sort_alpha = alpha[alpha_sort_order].astype(floatX)

        q = _integral_phase_function(Psi, sin_abs_sort_alpha, sort_alpha,
                                     alpha_sort_order)

        # F_0 = F_S_alpha0 + F_L_alpha0 + F_C_alpha0

        return flux_ratio_ppm, g, q
    else:
        return flux_ratio_ppm, g
