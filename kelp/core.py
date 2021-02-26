from math import sin, cos

import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline

from .fast import _h_ml_sum_cy, _integrated_blackbody, _phase_curve
from .registries import PhaseCurve

__all__ = ['Model']


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
        return np.ones_like(theta)
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


class Model(object):
    """
    Planetary system object for generating phase curves
    """

    def __init__(self, hotspot_offset, alpha, omega_drag, A_B, C_ml, lmax,
                 a_rs=None, rp_a=None, T_s=None, planet=None, filt=None):
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
        if l < 8:
            return H(l, theta, self.alpha)

        else:
            raise NotImplementedError('H only implemented to l=7, l={0}'
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

    def phase_curve(self, xi, n_theta=20, n_phi=200, f=2 ** -0.5, cython=True,
                    quad=False, check_sorted=True):
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
        fluxes : `~numpy.ndarray`
            System fluxes as a function of phase angle :math:`\xi`.
        """
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
                xi.astype(np.float64), self.hotspot_offset,
                self.omega_drag,
                self.alpha, self.C_ml, self.lmax,
                self.T_s, self.a_rs, self.rp_a,
                self.A_B, n_theta, n_phi,
                self.filt.wavelength.to(u.m).value,
                self.filt.transmittance,
                f=f
            )

        return PhaseCurve(xi, fluxes, channel=self.filt.name)

    def integrated_temperatures(self, n_theta=100, n_phi=100, f=2 ** -0.5):
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
            T[:, dayside_hemisphere],
            weights=integrand_dayside[:, dayside_hemisphere]
        )
        nightside_integrated_temperature = np.average(
            T[:, nightside_hemisphere],
            weights=integrand_nightside[:, nightside_hemisphere]
        )

        return dayside_integrated_temperature, nightside_integrated_temperature
