******************************
Albedo and Heat Redistribution
******************************

Following Cowan and Agol (2011), the dayside and nightside integrated
temperatures can be expressed as a function of the day-to-night heat
redistribution efficiency :math:`\epsilon`:

.. math::

    \begin{eqnarray}
    T_{d} &=& T_{\star} \left(1 - A_{B}\right)^{1/4} \sqrt{\frac{R_\star}{a}}  \left(\frac{2}{3} - \frac{5 \epsilon}{12}\right)^{1/4}\\
    T_{n} &=& T_{\star}  \left(1 - A_{B}\right)^{1/4} \sqrt{\frac{R_\star}{a}} \left(\frac{\epsilon}{4}\right)^{1/4}
    \end{eqnarray}

with the ratio of the semimajor axis :math:`a` to the stellar radius
:math:`R_\star`, the Bond albedo :math:`A_B`, and the stellar effective
temperature :math:`T_\star`. From these equations, it follows that:

.. math::

    \begin{eqnarray}
    \epsilon &=& \frac{8 T_{n}^{4}}{3 T_{d}^{4} + 5 T_{n}^{4}}\\
    A_B &=& 1 -\left(\frac{a}{R_\star}\right)^{2} \left(\frac{3 T_{d}^{4} + 5 T_{n}^{4}}{2T_{\star}^{4}}\right)
    \end{eqnarray}

The dayside and nightside integrated temperatures can be derived from the
temperature maps which we compute for a given combination of
:math:`\alpha, \omega_\mathrm{drag}, C_{11}`, so by varying :math:`C_{11}`
freely and varying :math:`\alpha` and :math:`\omega_\mathrm{drag}` within their
priors, we can directly compute the corresponding :math:`f` or :math:`\epsilon`
and :math:`A_B` with the relations above.

In the Cython implementation of kelp, the above looks like this:

.. code-block:: python

    >>> import numpy as np

    >>> from kelp import Model, Planet, Filter

    >>> # Construct planet, filter and model objects:
    >>> p = Planet.from_name("KELT-9")
    >>> filt = Filter.from_name("IRAC 2")
    >>> m = Model(0, 0.6, 4.5, 0.5, [[0], [0, 0.2, 0.0]], 1, planet=p, filt=filt)

    >>> # Construct a temperature map from which we'll compute the relevant parameters:
    >>> temp_map, theta, phi = m.temperature_map(n_theta=150, n_phi=300)
    >>> phi2d, theta2d = np.meshgrid(phi, theta)

    >>> # Compute the integrated dayside and nightside temperatures:
    >>> integrand_dayside = np.max(
    ...     [np.sin(theta2d) ** 2 * np.cos(phi2d),
    ...      np.zeros_like(theta2d)],
    ...     axis=0
    ... )
    >>> integrand_nightside = np.max(
    ...     [np.sin(theta2d) ** 2 * np.cos(phi2d + np.pi),
    ...      np.zeros_like(theta2d)],
    ...     axis=0
    ... )

    >>> dayside = np.sum(integrand_dayside * temp_map) / np.sum(integrand_dayside)
    >>> nightside = np.sum(integrand_nightside * temp_map) / np.sum(integrand_nightside)

    >>> # Compute the Bond albedo, heat redistribution and Greenhouse parameter:
    >>> A_B = (
    ...     p.T_s**4 - p.a**2 *
    ...     (3 * dayside**4 + 5*nightside**4) / 2
    ... ) / p.T_s**4
    >>>
    >>> epsilon = 8 * nightside**4 / (3 * dayside**4 + 5 * nightside**4)
    >>> f = (2/3 - epsilon * 5 / 12) ** 0.25
    >>>
    >>> print(A_B, epsilon, f)  # doctest: +FLOAT_CMP
    0.47854876402969493 0.3836298300218966 0.8437496969138535
