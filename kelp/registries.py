import os
from json import load
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from astropy.io import fits
import astropy.units as u

__all__ = ['Planet', 'Filter', 'PhaseCurve']

planets_path = os.path.join(os.path.dirname(__file__), 'data', 'planets.json')
filters_path = os.path.join(os.path.dirname(__file__), 'data', 'filters.json')
pc_path = os.path.join(os.path.dirname(__file__), 'data', 'lightcurves.fits')


class Planet(object):
    """
    Transiting planet parameters.

    This is meant to be a duck-type drop-in for the ``batman`` package's
    transiting exoplanet parameters ``TransitParams`` object.
    """
    with open(planets_path, 'r') as _f:
        _planets = load(_f)

    def __init__(self, per=None, t0=None, inc=None, rp=None, ecc=None, w=None,
                 a=None, u=None, fp=None, t_secondary=None, T_s=None, rp_a=None,
                 limb_dark='quadratic', name=None):
        self.per = per
        self.t0 = t0
        self.inc = inc
        self.rp = rp
        self.ecc = ecc
        self.w = w
        self.a = a
        self.u = u
        self.limb_dark = limb_dark
        self.fp = fp
        self.t_secondary = t_secondary
        self.T_s = T_s
        self.rp_a = rp_a
        self.name = name

    @classmethod
    def from_name(cls, name):
        """
        Initialize a Planet instance from the target name.

        There's a small (but growing?) database of planets pre-defined in the
        ``kelp/data/planets.json`` file. If your favorite planet is missing,
        pull requests are welcome!

        Parameters
        ----------
        name : str (i.e.: "Kepler-7" or "KELT-9")
             Name of the planet
        """

        return cls(name=name, **cls._planets[name])

    def eclipse_model(self, xi):
        r"""
        Compute eclipse model at orbital phases ``xi``.

        Parameters
        ----------
        xi : `~numpy.ndarray`
            Orbital phase angle :math:`\xi`

        Returns
        -------
        eclipse : `~numpy.ndarray`
            Eclipse model normalized such that flux is zero in eclipse.
        """
        from batman import TransitModel

        xi_over_pi = xi / np.pi
        eclipse = TransitModel(self, xi_over_pi, transittype='secondary',
                               exp_time=xi_over_pi[1] - xi_over_pi[0],
                               supersample_factor=3,
                               ).light_curve(self)
        eclipse -= eclipse.min()
        return eclipse


class Filter(object):
    """
    Astronomical filter object.
    """
    with open(filters_path, 'r') as _f:
        _filters = load(_f)

    def __init__(self, wavelength, transmittance, name=None):
        """
        Parameters
        ----------
        wavelength : `~numpy.ndarray`
            Wavelength array
        transmittance : `~numpy.ndarray`
            Transmittance array
        """
        self.wavelength = wavelength
        self.transmittance = transmittance
        self.name = name

    @classmethod
    def from_name(cls, name):
        """
        Initialize a Filter instance from the filter name.

        Parameters
        ----------
        name : str (i.e.: "IRAC 1" or "IRAC 2")
             Name of the filter
        """
        return cls(np.array(cls._filters[name]['wavelength']) * u.um,
                   np.array(cls._filters[name]['transmittance']),
                   name)

    def plot(self, ax=None, **kwargs):
        """
        Plot the filter transmittance curve.

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
        if ax is None:
            ax = plt.gca()
        ax.set(title=self.name)
        ax.plot(self.wavelength, self.transmittance, **kwargs)

        return ax

    def bin_down(self, bins=10):
        """
        Bin down the filter bandpass wavelengths and transmittances (shortcut
        for faster integration over the bandpass).
        """
        bs = binned_statistic(self.wavelength.value, self.transmittance,
                              bins=bins, statistic='median')
        bincenters = 0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])

        self.wavelength = bincenters * self.wavelength.unit
        self.transmittance = bs.statistic


class PhaseCurve(object):
    """
    Thermal phase curve.
    """
    fits_file = None
    available = []

    def __init__(self, xi, flux, name=None, channel=None, year=None,
                 renormalize=False):
        """
        Parameters
        ----------
        xi : `~numpy.ndarray`
            Times
        flux : `~numpy.ndarray`
            Flux measurements
        name : str
            Name of the host star
        channel : str
            Name of the Spitzer channel
        year : int
            Year of the observations (for disambiguating)
        renormalize : bool
            Re-normalize the phase curve such that it is represented as
            :math:`F_p/F_s`, in units of ppm
        """
        self.xi = xi[np.argsort(xi)]

        if renormalize:
            in_eclipse = np.abs(xi) < 0.1
            flux_in_eclipse = np.nanmedian(flux[in_eclipse])
            flux = 1e6 * (flux - flux_in_eclipse)

        self.flux = flux[np.argsort(xi)]
        self.name = name
        self.channel = channel
        self.year = year

    @classmethod
    def from_name(cls, name, channel, year=None):
        """
        Initialize a Filter instance from the filter name.

        Parameters
        ----------
        name : str (i.e.: "WASP-18", "KELT-9")
            Name of the host star
        channel : str (i.e.: "1" or "2")
            Name of the filter (IRAC channel number)
        year : int
            Year of the observations (when
            multiple observations are available)
        """
        if cls.fits_file is None:
            with fits.open(pc_path) as fitsfile:
                cls.fits_file = deepcopy(fitsfile)
            for hdu in cls.fits_file[1:]:
                cls.available.append("{0} (Ch {1}; year {2})"
                                     .format(hdu.header['NAME'],
                                             hdu.header['CHANNEL'],
                                             hdu.header['YEAR']))

        recarray = None
        for hdu in cls.fits_file[1:]:
            if (hdu.header['NAME'] == name and
                    hdu.header['CHANNEL'] == channel and
                    hdu.header['YEAR'] == year):
                recarray = hdu.data

        if recarray is not None:
            return cls(recarray['xi'], recarray['flux'],
                       name=name, channel=channel, year=year,
                       renormalize=False)
        else:
            raise KeyError(('Target {1} (Ch {0}, {2}) not ' +
                            'found in FITS registry, ' +
                            'which contains: {3}'
                            ).format(channel, name, year,
                                     ', '.join(sorted(cls.available))))

    def plot(self, ax=None, mask=None, **kwargs):
        """
        Plot the phase curve.

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
        if ax is None:
            ax = plt.gca()

        if mask is None:
            mask = np.ones_like(self.xi).astype(bool)

        ax.plot(self.xi[mask], self.flux[mask], **kwargs)

        return ax

    def _add_to_fits(self, fitsfile):
        """
        Add this phase curve to a FITS archive ``fitsfile``.

        Parameters
        ----------
        fitsfile : FITS file stream
            Open FITS file stream
        """
        ra = np.recarray(len(self.xi), names=["xi", "flux"],
                         formats=['f8', 'f8'])
        ra['xi'] = self.xi
        ra['flux'] = self.flux
        header = fits.Header(dict(YEAR=self.year,
                                  CHANNEL=self.channel,
                                  NAME=self.name))
        fitsfile.append(fits.BinTableHDU(ra, header))
