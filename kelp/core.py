import numpy as np
from astropy.modeling.blackbody import blackbody_lambda
import astropy.units as u
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

__all__ = ['System']


class System(object):
    """
    Planetary system object for generating phase curves
    """
    def __init__(self, alpha, omega_drag, A_B, C_ml, lmax, a_rs, T_s, filt):
        """
        Parameters
        ----------
        alpha : float
            Dimensionless fluid number
        omega_drag : float
            Dimensionless drag frequency
        A_B : float
            Bond albedo
        C_ml : array-like, list
            Spherical harmonic coefficients
        lmax : int
            Maximum `l` in spherical harmonic expansion
        a_rs : float
            Semimajor axis in units of stellar radii
        T_s : float [K]
            Stellar effective temperature
        filt : `~tynt.Filter`
            Filter of observations
        """
        self.alpha = alpha
        self.omega_drag = omega_drag
        self.a_rs = a_rs
        self.T_s = T_s
        self.A_B = A_B
        if len(C_ml) != lmax + 1:
            raise ValueError('Length of C_ml must be lmax+1')
        self.C_ml = np.diag(C_ml)
        self.lmax = lmax
        self.filt = filt

    def tilda_mu(self, theta):
        return self.alpha * self.mu(theta)

    def mu(self, theta):
        return np.cos(theta)

    def H(self, l, theta):
        if l < 0:
            return 0
        elif l == 0:
            return 1
        elif l == 1:
            return 2 * self.tilda_mu(theta)
        elif l == 2:
            return 4 * self.tilda_mu(theta)**2 - 2
        elif l == 3:
            return 8 * self.tilda_mu(theta)**3 - 12 * self.tilda_mu(theta)
        elif l == 4:
            return (16 * self.tilda_mu(theta)**4 - 48 *
                    self.tilda_mu(theta)**2 + 12)
        else:
            raise ValueError('H only implemented to l=4, l={0}'.format(l))

    def h_ml(self, omega_drag, alpha, m, l, theta, phi):
        if m == 0:
            return 0 * np.zeros((theta.shape[0], phi.shape[1]))
        prefactor = (self.C_ml[m][l] / (omega_drag**2 * alpha**4 + m**2) *
                     np.exp(-self.tilda_mu(theta)**2 / 2))
        return prefactor * (self.mu(theta) * m * self.H(l, theta) *
                            np.cos(m * phi) + alpha * omega_drag *
                            (2 * l * self.H(l-1, theta) -
                             self.tilda_mu(theta) *
                             self.H(l, theta)) * np.sin(m*phi))

    def temperature_map(self, n_theta, n_phi, f):
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
        phi = np.linspace(-2*np.pi, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)

        theta2d, phi2d = np.meshgrid(theta, phi)
        h_ml_sum = np.zeros((n_phi, n_theta))

        for l in range(0, self.lmax+1):
            for m in range(-self.lmax, self.lmax+1):
                h_ml_sum += self.h_ml(self.omega_drag, self.alpha, m, l,
                                      theta2d, phi2d)
        T_eq = f * self.T_s * np.sqrt(1 / self.a_rs)

        T = T_eq * (1 - self.A_B)**0.25 * (1 + h_ml_sum)

        return T, theta, phi

    def integrated_blackbody(self, n_theta, n_phi, f):
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
        T, theta, phi = self.temperature_map(n_theta, n_phi, f)
        int_bb = np.trapz(blackbody_lambda(
                              self.filt.wavelength[:, np.newaxis, np.newaxis],
                              T) /
                          blackbody_lambda(
                              self.filt.wavelength[:, np.newaxis, np.newaxis],
                              self.T_s) *
                          u.sr *
                          self.filt.wavelength[:, np.newaxis, np.newaxis] *
                          self.filt.transmittance[:, np.newaxis, np.newaxis],
                          self.filt.wavelength.value, axis=0
                          ).value
        interp_bb = RectBivariateSpline(theta, phi, int_bb.T)
        return lambda theta, phi: interp_bb(theta, phi)[0][0]

    def phase_curve(self, xi, n_theta=30, n_phi=30, f=1 / np.sqrt(2)):
        """
        Compute the optical thermal phase curve of the system as a function
        of observer angle `xi`

        Parameters
        ----------
        xi : array-like
        n_theta : int
            Number of grid points in latitude
        n_phi : int
            Number of grid points in longitude
        f : float
            Greenhouse parameter (typically 1/sqrt(2)).

        Returns
        -------
        fluxes : `~numpy.ndarray`
            System fluxes as a function of phase angle `xi`.
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
        return fluxes

    def plot_temperature_maps(self, n_theta=10, n_phi=30, f=1 / np.sqrt(2)):
        """
        Plot the temperature map.

        Parameters
        ----------
        n_theta : int
        n_phi : int
        f : float
        """
        from mpl_toolkits.mplot3d import Axes3D

        T, theta, phi = self.temperature_map(n_theta, n_phi, f)
        T_norm = (T - T.min()) / T.ptp()
        theta2d, phi2d = np.meshgrid(theta, phi)

        x = np.sin(theta2d) * np.cos(phi2d)
        y = np.sin(theta2d) * np.sin(phi2d)
        z = np.cos(theta2d)

        cax = plt.imshow(T.T, extent=[phi.min()/np.pi, phi.max()/np.pi,
                                      theta.min()/np.pi, theta.max()/np.pi])
        plt.colorbar(cax, extend='both', label='Temperature [K]')
        plt.xlabel('$\\phi/\\pi$')
        plt.ylabel('$\\theta/\\pi$')
        plt.show()

        fig = plt.figure(figsize=(3, 3))
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, shade=False,
                        facecolors=plt.cm.viridis(T_norm))
        ax.view_init(0, 90)
        # Turn off the axis planes
        ax.set_axis_off()
        ax.set_aspect(1.0)
        plt.show()
