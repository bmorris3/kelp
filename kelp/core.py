from math import sin, cos

import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline
from scipy.special import hermite
import astropy.units as u
from astropy.modeling.models import BlackBody

from .registries import PhaseCurve

__all__ = ['Model', 'StellarSpectrum']


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
    s2 = (np.sum(z[1:m, 0, :], axis=0) + np.sum(z[1:m, n, :], axis=0) +
          np.sum(z[0, 1:n, :], axis=0) + np.sum(z[m, 1:n, :], axis=0))
    s3 = np.sum(np.sum(z[1:m, 1:n, :], axis=0), axis=0)
    return dx * dy * (s1 + 2 * s2 + 4 * s3) / 4


def mu(theta):
    r"""
    Angle :math:`\mu = \cos(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    """
    return np.cos(theta)


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


def H(lmax, theta, alpha):
    r"""
    Hermite Polynomials in :math:`\tilde{\mu}(\theta)`.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    theta : float
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`

    Returns
    -------
    result : `~numpy.ndarray`
        Hermite Polynomial evaluated at angles :math:`\theta`.
    """
    return np.sum([a * tilda_mu(theta, alpha) ** l for l, a in
        zip(range(0, lmax + 1)[::-1], list(hermite(n=lmax)))], axis=0
    )


def _integral_phase_function(Psi, sin_abs_sort_alpha, sort_alpha, sort):
    """
    Integral phase function q for a generic, possibly asymmetric reflectivity
    map
    """
    return np.trapz(Psi[sort] * sin_abs_sort_alpha, sort_alpha)


def reflected_phase_curve(xi, omega, g, a_rp):
    """
    Reflected light phase curve for a homogeneous sphere by
    Heng, Morris & Kitzmann (2021).

    Parameters
    ----------
    xi : `~np.ndarray`
        Orbital phases of each observation defined on (-pi, pi)
    omega : tensor-like
        Single-scattering albedo as defined in
    g : tensor-like
        Scattering asymmetry factor, ranges from (-1, 1).
    a_rp : float, tensor-like
        Semimajor axis scaled by the planetary radius

    Returns
    -------
    flux_ratio_ppm : `~np.ndarray`
        Flux ratio between the reflected planetary flux and the stellar flux in
        units of ppm.
    A_g : float
        Geometric albedo derived for the planet given {omega, g}.
    q : float
        Integral phase function
    """
    phases = (xi + np.pi) / 2 / np.pi

    # Convert orbital phase on (0, 1) to "alpha" on (0, np.pi)
    alpha = (2 * np.pi * phases - np.pi)
    abs_alpha = np.abs(alpha)
    alpha_sort_order = np.argsort(alpha)
    sin_abs_sort_alpha = np.sin(abs_alpha[alpha_sort_order])
    sort_alpha = alpha[alpha_sort_order]

    gamma = np.sqrt(1 - omega)
    eps = (1 - gamma) / (1 + gamma)

    # Equation 34 for Henyey-Greestein
    P_star = (1 - g ** 2) / (1 + g ** 2 +
                             2 * g * np.cos(alpha)) ** 1.5
    # Equation 36
    P_0 = (1 - g) / (1 + g) ** 2

    # Equation 10:
    Rho_S = P_star - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_S_0 = P_0 - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_L = 0.5 * eps * (2 - eps) * (1 + eps) ** 2
    Rho_C = eps ** 2 * (1 + eps) ** 2

    alpha_plus = np.sin(abs_alpha / 2) + np.cos(abs_alpha / 2)
    alpha_minus = np.sin(abs_alpha / 2) - np.cos(abs_alpha / 2)

    valid_conditions = (
        (alpha_minus != -1) & (alpha_plus != 1) & (alpha_plus != -1) &
        (alpha_minus != 1)
    )
    num1 = np.where(
        valid_conditions,
        (1 + alpha_minus),
        1
    )
    num2 = np.where(
        valid_conditions,
        (alpha_plus - 1),
        1
    )
    denom1 = np.where(
        valid_conditions,
        (1 + alpha_plus),
        1
    )
    denom2 = np.where(
        valid_conditions,
        (1 - alpha_minus),
        1
    )

    # Equation 11:
    Psi_0 = np.where(
        valid_conditions,
        np.log(num1 * num2 / denom1 / denom2),
        0
    )

    Psi_S = 1 - 0.5 * (np.cos(abs_alpha / 2) -
                       1.0 / np.cos(abs_alpha / 2)) * Psi_0
    Psi_L = (np.sin(abs_alpha) + (np.pi - abs_alpha) *
             np.cos(abs_alpha)) / np.pi
    Psi_C = (-1 + 5 / 3 * np.cos(abs_alpha / 2) ** 2 - 0.5 *
             np.tan(abs_alpha / 2) * np.sin(abs_alpha / 2) ** 3 * Psi_0)

    # Fix the case when the phase angle is exactly 0 or pi
    Psi_S[abs_alpha == 0] = 1
    Psi_C[abs_alpha == 0] = 2 / 3
    Psi_S[abs_alpha == np.pi] = 0
    Psi_C[abs_alpha == np.pi] = 0

    # Equation 8:
    A_g = omega / 8 * (P_0 - 1) + eps / 2 + eps ** 2 / 6 + eps ** 3 / 24

    # Equation 9:
    Psi = ((12 * Rho_S * Psi_S + 16 * Rho_L *
            Psi_L + 9 * Rho_C * Psi_C) /
           (12 * Rho_S_0 + 16 * Rho_L + 6 * Rho_C))

    flux_ratio_ppm = 1e6 * (a_rp ** -2 * A_g * Psi)

    q = _integral_phase_function(
        Psi, sin_abs_sort_alpha, sort_alpha, alpha_sort_order
    )

    return flux_ratio_ppm, A_g, q


class Model(object):
    """
    Planetary system object for generating phase curves
    """

    def __init__(self, hotspot_offset=None, alpha=None, omega_drag=None,
                 A_B=None, C_ml=None, lmax=None, a_rs=None, rp_a=None, T_s=None,
                 planet=None, filt=None, stellar_spectrum=None):
        r"""
        Parameters
        ----------
        hotspot_offset : float
            Angle of hotspot offset [radians]
        alpha : float
            Dimensionless fluid number
        omega_drag : float
            Dimensionless drag frequency
        A_B : float
            Bond albedo
        C_ml : array-like, list
            Spherical harmonic coefficients
        lmax : int
            Maximum :math:`\ell` in spherical harmonic expansion
        a_rs : float
            Semimajor axis in units of stellar radii
        rp_a : float
            Planet radius normalized by the semimajor axis
        T_s : float [K]
            Stellar effective temperature
        planet : `~kelp.Planet`
            Planet instance which can be specified instead of the three
            previous parameters
        filt : `~kelp.Filter`
            Filter of observations
        stellar_spectrum : `~kelp.StellarSpectrum`
            Stellar spectrum (if not supplied, assumes a Planck function at
            temperature ``T_s``.)
        """
        self.hotspot_offset = hotspot_offset
        self.alpha = alpha
        self.omega_drag = omega_drag
        self.A_B = A_B

        if len(C_ml) != lmax + 1:
            raise ValueError('Length of C_ml must be lmax+1')

        self.C_ml = C_ml
        self.lmax = lmax
        self.filt = filt

        if planet is not None:
            rp_a = planet.rp_a
            a_rs = planet.a
            T_s = planet.T_s

        self.rp_a = rp_a
        self.a_rs = a_rs
        self.T_s = T_s

        if stellar_spectrum is not None:
            self.stellar_spectrum = stellar_spectrum
        else:
            self.stellar_spectrum = StellarSpectrum.from_zeros()

    def tilda_mu(self, theta):
        r"""
        The normalized quantity
        :math:`\tilde{\mu} = \alpha \mu(\theta)`

        Parameters
        ----------
        theta : `~numpy.ndarray`
            Angle :math:`\theta`
        """
        return self.alpha * self.mu(theta)

    def mu(self, theta):
        r"""
        Angle :math:`\mu = \cos(\theta)`

        .. note::

            It is assumed that ``theta`` is linearly spaced and always
            increasing.

        Parameters
        ----------
        theta : `~numpy.ndarray`
            Angle :math:`\theta`
        """
        return np.cos(theta)

    def H(self, l, theta):
        r"""
        Hermite Polynomials in :math:`\tilde{\mu}(\theta)`.

        Parameters
        ----------
        l : int
            Implemented through :math:`\ell \leq 7`.
        theta : float
            Angle :math:`\theta`

        Returns
        -------
        result : `~numpy.ndarray`
            Hermite Polynomial evaluated at angles :math:`\theta`.
        """
        if l < 52:
            return H(l, theta, self.alpha)

        else:
            raise NotImplementedError('H only implemented to l=51, l={0}'
                                      .format(l))

    def h_ml(self, m, l, theta, phi):
        r"""
        The :math:`h_{m\ell}` basis function.

        .. note::

            It is assumed that ``theta`` and ``phi`` are linearly spaced and
            always increasing.

        Parameters
        ----------
        m : int
            Spherical harmonic ``m`` index
        l : int
            Spherical harmonic ``l`` index
        theta : `~numpy.ndarray`
            Latitudinal coordinate
        phi : `~numpy.ndarray`
            Longitudinal coordinate

        Returns
        -------
        hml : `~numpy.ndarray`
            :math:`h_{m\ell}` basis function.
        """
        if m == 0:
            return 0 * np.zeros((theta.shape[1], phi.shape[0]))

        prefactor = (self.C_ml[l][m] /
                     (self.omega_drag ** 2 * self.alpha ** 4 + m ** 2) *
                     np.exp(-self.tilda_mu(theta) ** 2 / 2))

        a = self.mu(theta) * m * self.H(l, theta) * np.cos(m * phi)

        b = self.alpha * self.omega_drag * (self.tilda_mu(theta) *
                                            self.H(l, theta) -
                                            self.H(l + 1, theta))

        c = np.sin(m * phi)
        hml = prefactor * (a + b * c)
        return hml.T

    def temperature_map(self, n_theta, n_phi, f=2 ** -0.5, cython=True):
        """
        Temperature map as a function of latitude (theta) and longitude (phi).

        Parameters
        ----------
        n_theta : int
            Number of grid points in latitude
        n_phi : int
            Number of grid points in longitude
        f : float
            Greenhouse parameter (typically 1/sqrt(2)).
        cython : bool
            Use cython implementation of the hml basis. Default is True,
            yields a factor of ~two speedup.

        Returns
        -------
        T : `~np.ndarray`
            Temperature map evaluated precisely at each theta, phi
        theta : `~np.ndarray`
            Latitudes over which temperature map is computed
        phi : `~np.ndarray`
            Longitudes over which temperature map is computed
        """
        from .fast import _h_ml_sum_cy

        phase_offset = np.pi / 2
        phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)

        theta2d, phi2d = np.meshgrid(theta, phi)

        if cython:
            # Cython alternative to the pure python implementation:
            h_ml_sum = _h_ml_sum_cy(self.hotspot_offset, self.omega_drag,
                                    self.alpha, theta2d, phi2d, self.C_ml,
                                    self.lmax)
        else:
            # Slow loops, which have since been cythonized:
            h_ml_sum = np.zeros((n_theta, n_phi))

            for l in range(1, self.lmax + 1):
                for m in range(-l, l + 1):
                    h_ml_sum += self.h_ml(m, l,
                                          theta2d, phi2d + phase_offset +
                                          self.hotspot_offset)

        T_eq = f * self.T_s * np.sqrt(1 / self.a_rs)

        T = T_eq * (1 - self.A_B) ** 0.25 * (1 + h_ml_sum)

        return T, theta, phi

    def albedo_redist(self, temp_map, theta, phi):
        """
        Compute the Bond albedo and heat redistribution efficiency for
        ``temp_map``.

        Parameters
        ----------
        temp_map : `~np.ndarray`
            Temperature map produced by e.g. `~kelp.Model.temperature_map`.
        theta : `~np.ndarray`
            Latitudes produced by e.g. `~kelp.Model.temperature_map`.
        phi : `~np.ndarray`
            Longitudes produced by e.g. `~kelp.Model.temperature_map`.

        Returns
        -------
        bond_albedo : float
            Bond albedo
        epsilon : float
            Heat redistribution efficiency
        """
        theta2d, phi2d = np.meshgrid(theta, phi)

        flux_planet_total = trapz2d(
            temp_map.T[..., None] ** 4 * np.sin(theta2d[..., None]) *
            (phi2d[..., None] > 0),
            phi, theta
        )[0]
        flux_star = np.pi * self.T_s ** 4

        bond_albedo = 1 - self.a_rs ** 2 * flux_planet_total / flux_star

        mask_dayside = np.abs(phi2d) < np.pi / 2
        mask_nightside = np.abs(phi2d + np.pi) < np.pi / 2

        flux_dayside = np.sum(
            (temp_map.T * mask_dayside) ** 4) / np.sum(mask_dayside)
        flux_nightside = np.sum(
            (temp_map.T * mask_nightside) ** 4) / np.sum(
            mask_nightside)

        epsilon = flux_nightside / flux_dayside

        return bond_albedo, epsilon

    def integrated_blackbody(self, n_theta, n_phi, f=2 ** -0.5, cython=True):
        """
        Integral of the blackbody function convolved with a filter bandpass.

        Parameters
        ----------
        n_theta : int
            Number of grid points in latitude
        n_phi : int
            Number of grid points in longitude
        f : float
            Greenhouse parameter (typically 1/sqrt(2)).

        Returns
        -------
        interp_bb : function
            Interpolation function for the blackbody map as a function of
            latitude (theta) and longitude (phi)
        """
        from .fast import _integrated_blackbody

        if cython:
            int_bb, func = _integrated_blackbody(self.hotspot_offset,
                                                 self.omega_drag,
                                                 self.alpha, self.C_ml,
                                                 self.lmax,
                                                 self.T_s, self.a_rs, self.rp_a,
                                                 self.A_B, n_theta, n_phi,
                                                 self.filt.wavelength.to(
                                                     u.m).value,
                                                 self.filt.transmittance,
                                                 f=f)
            return int_bb, func

        else:
            T, theta_grid, phi_grid = self.temperature_map(n_theta, n_phi, f,
                                                           cython=cython)
            if (T < 0).any():
                return lambda theta, phi: np.inf

            bb_t = BlackBody(temperature=T * u.K)
            int_bb = np.trapz(bb_t(self.filt.wavelength[:, None, None]) *
                              self.filt.transmittance[:, None, None],
                              self.filt.wavelength, axis=0
                              ).si.value
            interp_bb = RectBivariateSpline(theta_grid, phi_grid, int_bb,
                                            kx=1, ky=1)
            return int_bb, lambda theta, phi: interp_bb(theta, phi)[0][0]

    def reflected_phase_curve(self, xi, omega, g):
        r"""
        Reflected light phase curve for a homogeneous sphere by
        Heng, Morris & Kitzmann (2021).

        Parameters
        ----------
        xi : `~np.ndarray`
            Orbital phases of each observation defined on (-pi, pi)
        omega : float
            Single-scattering albedo as defined in
        g : float
            Scattering asymmetry factor, ranges from (-1, 1).

        Returns
        -------
        phase_curve : `~kelp.PhaseCurve`
            Flux ratio between the reflected planetary flux and the stellar flux in
            units of ppm.
        A_g : float
            Geometric albedo derived for the planet given {omega, g}.
        q : float
            Integral phase function
        """
        reflected_light_ppm, A_g, q = reflected_phase_curve(
            xi, omega, g, 1/self.rp_a
        )
        return PhaseCurve(xi, reflected_light_ppm), A_g, q

    def phase_curve(self, xi, omega, g, n_theta=20, n_phi=200, f=2 ** -0.5,
                    cython=True, quad=False, check_sorted=True):
        r"""
        Reflected light phase curve for a homogeneous sphere by
        Heng, Morris & Kitzmann (2021) with the thermal phase curve for a planet
        represented with a spherical harmonic expansion by Morris et al.
        (in prep).

        Parameters
        ----------
        xi : `~np.ndarray`
            Orbital phases of each observation defined on (-pi, pi)
        omega : float
            Single-scattering albedo as defined in
        g : float
            Scattering asymmetry factor, ranges from (-1, 1).
        n_theta : int
            Number of grid points in latitude
        n_phi : int
            Number of grid points in longitude
        f : float
            Greenhouse parameter (typically 1/sqrt(2)).
        cython : bool (default is True)
            Use Cython implementation of the `integrated_blackbody` function
            (deprecated). Default is True.
        quad : bool (default is False)
            Use `dblquad` to integrate the temperature map if True,
            else use trapezoidal approximation.
        check_sorted : bool (default is True)
            Check that the ``xi`` values are sorted before passing to cython
            (carefully turning this off will speed things up a bit)

        Returns
        -------
        phase_curve : `~kelp.PhaseCurve`
            Flux ratio between the reflected planetary flux and the stellar flux in
            units of ppm.
        A_g : float
            Geometric albedo derived for the planet given {omega, g}.
        q : float
            Integral phase function
        """

        reflected, A_g, q = self.reflected_phase_curve(
            xi, omega, g
        )

        thermal = self.thermal_phase_curve(
            xi, n_theta=n_theta, n_phi=n_phi, f=f, cython=cython, quad=quad,
            check_sorted=check_sorted
        )

        return PhaseCurve(xi, thermal.flux + reflected.flux), A_g, q

    def thermal_phase_curve(self, xi, n_theta=20, n_phi=200, f=2 ** -0.5,
                            cython=True, quad=False, check_sorted=True):
        r"""
        Compute the thermal phase curve of the system as a function
        of observer angle ``xi``.

        .. note::

            The ``xi`` axis is assumed to be monotonically increasing when
            ``check_sorted=False``, ``cython=True`` and ``quad=False``.

        Parameters
        ----------
        xi : array-like
            Orbital phase angle
        n_theta : int
            Number of grid points in latitude
        n_phi : int
            Number of grid points in longitude
        f : float
            Greenhouse parameter (typically 1/sqrt(2)).
        cython : bool
            Use Cython implementation of the `integrated_blackbody` function
            (deprecated). Default is True.
        quad : bool
            Use `dblquad` to integrate the temperature map if True,
            else use trapezoidal approximation.
        check_sorted : bool
            Check that the ``xi`` values are sorted before passing to cython
            (carefully turning this off will speed things up a bit)

        Returns
        -------
        phase_curve : `~kelp.PhaseCurve`
            System fluxes as a function of phase angle :math:`\xi`.
        """
        from .fast import _phase_curve

        rp_rs2 = (self.rp_a * self.a_rs) ** 2

        if quad:
            fluxes = np.zeros(len(xi))

            int_bb, interp_blackbody = self.integrated_blackbody(n_theta, n_phi,
                                                                 f, cython)

            def integrand(phi, theta, xi):
                return (interp_blackbody(theta, phi) * sin(theta) ** 2 *
                        cos(phi + xi))

            bb_ts = BlackBody(temperature=self.T_s * u.K)
            planck_star = np.trapz(self.filt.transmittance *
                                   bb_ts(self.filt.wavelength),
                                   self.filt.wavelength).si.value

            for i in range(len(xi)):
                fluxes[i] = dblquad(integrand, 0, np.pi,
                                    lambda x: -xi[i] - np.pi / 2,
                                    lambda x: -xi[i] + np.pi / 2,
                                    epsrel=100, args=(xi[i],)
                                    )[0] * rp_rs2 / np.pi / planck_star
        else:

            if check_sorted:
                if not np.all(np.diff(xi) >= 0):
                    raise ValueError("xi array must be sorted")

            fluxes = _phase_curve(
                xi.astype('double'),
                self.hotspot_offset,
                self.omega_drag,
                self.alpha, self.C_ml, self.lmax,
                self.T_s, self.a_rs, self.rp_a,
                self.A_B, n_theta, n_phi,
                self.filt.wavelength.to(u.m).value.astype('double'),
                self.filt.transmittance.astype('double'),
                f,
                self.stellar_spectrum.wavelength.to(
                    u.m).value.astype('double'),
                self.stellar_spectrum.spectral_flux_density.to(
                    u.W/u.m**3).value.astype('double')
            )

        return PhaseCurve(xi, 1e6 * fluxes, channel=self.filt.name)

    def integrated_temperatures(self, n_theta=100, n_phi=100, f=2 ** -0.5):
        """
        Compute the integrated dayside and nightside temperatures for the
        temperature map.

        .. note::
            The dayside/nightside integrated temperatures reported by this
            function are weighted by their emitted power, i.e. we take the
            1/4 root of the mean of the temperature raised to the fourth power.

        Parameters
        ----------
        n_theta : int
            Number of grid points in latitude
        n_phi : int
            Number of grid points in longitude
        f : float
            Greenhouse parameter (typically 1/sqrt(2)).
        cython : bool
            Use cython implementation of the hml basis. Default is True,
            yields a factor of ~two speedup.

        Returns
        -------
        dayside : float
            Integrated dayside temperature [K]
        nightside : float
            Integrated nightside temperature [K]
        """
        T, theta, phi = self.temperature_map(n_theta=n_theta, n_phi=n_phi, f=f)

        theta2d, phi2d = np.meshgrid(theta, phi)

        dayside_hemisphere = (phi < np.pi / 2) & (phi > -np.pi / 2)
        nightside_hemisphere = (phi > np.pi / 2) & (phi < 3 / 2 * np.pi)

        integrand_dayside = np.max(
            [np.sin(theta2d) ** 2 * np.cos(phi2d),
             np.zeros_like(theta2d)],
            axis=0
        ).T
        integrand_nightside = np.max(
            [np.sin(theta2d) ** 2 * np.cos(phi2d + np.pi),
             np.zeros_like(theta2d)], axis=0
        ).T

        dayside_integrated_temperature = np.average(
            T[:, dayside_hemisphere] ** 4,
            weights=integrand_dayside[:, dayside_hemisphere]
        ) ** (1/4)
        nightside_integrated_temperature = np.average(
            T[:, nightside_hemisphere] ** 4,
            weights=integrand_nightside[:, nightside_hemisphere]
        ) ** (1/4)

        return dayside_integrated_temperature, nightside_integrated_temperature


class StellarSpectrum(object):
    def __init__(self, wavelength, spectral_flux_density):
        """
        Parameters
        ----------
        wavelength : `~astropy.units.Quantity`
            Wavelength array for stellar spectrum
        spectral_flux_density : `~astropy.units.Quantity`
            Spectral flux density as a function of wavelength. Should have units
            compatible with W/m^2/micron, for example.
        """
        self.wavelength = wavelength
        self.spectral_flux_density = spectral_flux_density

    @classmethod
    def from_zeros(cls, size=10):
        """
        Construct a stellar spectrum with all zeros as the wavelength and
        spectrum arrays.

        This effectively turns off the custom stellar spectrum
        feature.
        """
        return cls(np.zeros(size) * u.m, np.zeros(size) * u.W/u.m**3)

    @classmethod
    def from_phoenix(cls, T_s, log_g=4.5, cache=False):
        """
        Return PHOENIX model stellar spectrum for a star with a given effective
        temperature ``T_s`` and surface gravity ``log_g``.

        Parameters
        ----------
        T_s : int
            Effective temperature
        log_g : float
            Surface gravity
        cache : bool
            Cache the retrieved stellar spectrum.
        """
        from expecto import get_spectrum

        spectrum = get_spectrum(T_s, log_g, cache=cache)

        return cls(spectrum.wavelength, spectrum.flux)

    @classmethod
    def from_blackbody(cls, T_s):
        """
        Return PHOENIX model stellar spectrum for a star with a given effective
        temperature ``T_s`` and surface gravity ``log_g``.

        Parameters
        ----------
        T_s : int
            Effective temperature
        log_g : float
            Surface gravity
        """
        from astropy.modeling.models import BlackBody

        wavelengths = np.linspace(500, 55000, 1000) * u.Angstrom
        bb = np.pi * u.sr * BlackBody(
            T_s * u.K, scale=1 * u.W / u.m ** 3 / u.sr
        )(wavelengths)

        return cls(wavelengths, bb)

    def plot(self, ax=None, **kwargs):
        """
        Plot the stellar spectrum.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axis object
        kwargs : dict
            Dictionary passed to the `~matplotlib.pyplot.plot` command

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Updated axis object
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support

        if ax is None:
            ax = plt.gca()

        with quantity_support():
            ax.plot(self.wavelength, self.spectral_flux_density, **kwargs)

        return ax
