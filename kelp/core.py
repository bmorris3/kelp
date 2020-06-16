import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline

# from numba import njit

from astropy.modeling.models import BlackBody
import astropy.units as u

__all__ = ['Model']


# @njit
def mu(theta):
    r"""
    Angle :math:`\mu = \cos(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    """
    return np.cos(theta)


# @njit
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


# @njit
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

    def temperature_map(self, n_theta, n_phi, f=2**-0.5):
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
        """
        phase_offset = np.pi / 2
        phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)

        theta2d, phi2d = np.meshgrid(theta, phi)
        h_ml_sum = np.zeros((n_theta, n_phi))

        for l in range(1, self.lmax + 1):
            for m in range(-l, l + 1):
                h_ml_sum += self.h_ml(m, l,
                                      theta2d, phi2d + phase_offset +
                                      self.hotspot_offset)
        T_eq = f * self.T_s * np.sqrt(1 / self.a_rs)

        T = T_eq * (1 - self.A_B)**0.25 * (1 + h_ml_sum)

        return T, theta, phi

    def integrated_blackbody(self, n_theta, n_phi, f=2**-0.5):
        """
        Integral of the blackbody function convolved with a filter bandpass.

        Time for WASP-121 benchmark test: 63.7 ms

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
        T, theta, phi = self.temperature_map(n_theta, n_phi, f)
        if (T < 0).any():
            return lambda theta, phi: np.inf

        bb_t = BlackBody(temperature=T * u.K)
        bb_ts = BlackBody(temperature=self.T_s * u.K)

        bb_ratio = (bb_t(self.filt.wavelength[:, None, None]) /
                    bb_ts(self.filt.wavelength[:, None, None]))

        int_bb = np.trapz(bb_ratio *
                          self.filt.transmittance[:, None, None],
                          self.filt.wavelength.value, axis=0
                          ).value
        rp_rs = self.rp_a * self.a_rs
        interp_bb = RectBivariateSpline(theta, phi, int_bb * rp_rs**2)
        return lambda theta, phi: interp_bb(theta, phi)[0][0]

    def reflected(self, xi):
        """
        Symmetric reflection component of the phase curve.

        Parameters
        ----------
        xi : array-like
            Orbital phase angle

        Returns
        -------
        f_R_sym : array-like
            Reflected light (symmetric) component of the orbital phase curve.
        """
        A_g = 2 / 3 * self.A_B
        return (self.rp_a ** 2 * A_g / np.pi * (np.sin(np.abs(xi)) +
                (np.pi - np.abs(xi)) * np.cos(np.abs(xi))))

    def phase_curve(self, xi, n_theta=30, n_phi=30, f=1 / np.sqrt(2),
                    reflected=False):
        r"""
        Compute the thermal phase curve of the system as a function
        of observer angle ``xi``.

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
        reflected : bool
            Include reflected light component

        Returns
        -------
        fluxes : `~numpy.ndarray`
            System fluxes as a function of phase angle :math:`\xi`.
        """
        interp_blackbody = self.integrated_blackbody(n_theta, n_phi, f)

        def integrand(phi, theta, xi):
            return (interp_blackbody(theta, phi) * np.sin(theta)**2 *
                    np.cos(phi + xi))

        fluxes = np.zeros(len(xi))
        for i in range(len(xi)):
            fluxes[i] = dblquad(integrand, 0, np.pi,
                                lambda x: -xi[i] - np.pi / 2,
                                lambda x: -xi[i] + np.pi / 2,
                                epsrel=100, args=(xi[i],))[0]
        if reflected:
            fluxes += self.reflected(xi)

        return fluxes
