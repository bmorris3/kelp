from batman import TransitParams, TransitModel
import astropy.units as u
from astropy.constants import R_sun

__all__ = ['kepler7_eclipse']


def kepler7_eclipse(times):
    """
    Return secondary eclipse model for Kepler-7 b.

    Parameters
    ----------
    times : times at which to evaluate the model.

    Returns
    -------
    eclipse : `~numpy.ndarray`
        Fluxes near secondary eclipse.
    """
    params = TransitParams()
    a = 0.06224 * u.AU
    R_star = 1.843 * R_sun
    a_rs = float(a / R_star)
    params.t0 = 1               # Central time of PRIMARY transit [days]
    params.per = 2       # Period [days]
    params.inc = 85.161            # Inclination [degrees]
    params.ecc = 0.0              # Eccentricity
    params.w = 90                 # Argument of periastron
    params.rp = 0.08294            # Planet to star radius ratio
    params.a = a_rs              # Semi-major axis scaled by stellar radius
    params.limb_dark = 'quadratic'
    params.u = [0.4, 0.2]
    params.fp = 1.0
    params.t_secondary = 0

    eclipse = TransitModel(params, times, transittype='secondary'
                           ).light_curve(params)
    eclipse -= eclipse.min()

    return eclipse
