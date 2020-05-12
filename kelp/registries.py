import os
from json import load
from batman import TransitModel

__all__ = ['Planet', 'Filter']

json_path = os.path.join(os.path.dirname(__file__), 'data', 'planets.json')
filters_path = os.path.join(os.path.dirname(__file__), 'data', 'filters.json')


class Planet(object):
    """
    Transiting planet parameters.

    This is meant to be a duck-type drop-in for the ``batman`` package's
    transiting exoplanet parameters ``TransitParams`` object.
    """
    def __init__(self, per=None, t0=None, inc=None, rp=None, ecc=None, w=None,
                 a=None, u=None, fp=None, t_secondary=None,
                 limb_dark='quadratic'):
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
        with open(json_path, 'r') as f:
            planets = load(f)

        return cls(**planets[name])

    def eclipse_model(self, xi_over_pi):
        r"""
        Compute eclipse model at orbital phases ``xi``.

        Parameters
        ----------
        xi_over_pi : `~numpy.ndarray`
            Orbital phase angle :math:`\xi/\pi` normalized on [-1, 1]

        Returns
        -------
        eclipse : `~numpy.ndarray`
            Eclipse model normalized such that flux is zero in eclipse.
        """
        eclipse = TransitModel(self, xi_over_pi, transittype='secondary'
                               ).light_curve(self)
        eclipse -= eclipse.min()
        return eclipse


class Filter(object):
    """
    Astronomical filter object.
    """
    def __init__(self, wavelength, transmittance):
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

    @classmethod
    def from_name(cls, name):
        """
        Initialize a Filter instance from the filter name.

        Parameters
        ----------
        name : str (i.e.: "IRAC 1" or "IRAC 2")
             Name of the filter
        """
        with open(filters_path, 'r') as f:
            filters = load(f)

        return cls(**filters[name])
