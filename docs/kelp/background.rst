Background
==========

Perturbed spherical harmonic basis: temperature map
---------------------------------------------------

The photospheric temperature of an exoplanet, as a function of the
planetary latitude :math:`\theta` and longitude :math:`\phi` can be defined as:

.. math::

    T = T_\mathrm{eq} (1 - A_B)^{1/4} \left( 1 + \sum_{m, \ell}^{\ell_{\rm max}} h_{m\ell}(\theta, \phi) \right)


such that :math:`T_\mathrm{eq} = f T_\mathrm{eff} \sqrt{R_\star/a}` is the equilibrium
temperature for greenhouse factor :math:`f`, stellar effective temperature
:math:`T_\mathrm{eff}`, and normalized semimajor axis :math:`a/R_\star`;
:math:`A_B` is the Bond albedo. The prefactor on the left acts as a constant
scaling term for the absolute temperature field, on which the :math:`h_{ml}`
terms are a perturbation.

The :math:`h_{m\ell}(\alpha, \omega_\mathrm{drag})` terms are defined by:

.. math::

    \begin{split}
    h_{m\ell} = \frac{C_{m\ell}}{\omega_\mathrm{drag}^2 \alpha^4 + m^2} e^{-\tilde{\mu}/2} [ \mu m H_{\ell} \cos(m \phi) \\
    + \alpha \omega_\mathrm{drag} (2lH_{\ell-1} - \tilde{\mu}H_\ell) \sin(m\phi) ],
    \end{split}

where

.. math::

    \alpha = \mathcal{R}^{-1/2} \mathcal{P}^{-1/4}

is the dimensionless fluid number of Heng & Workman (2014), and is a function of the
Reynold's number :math:`\mathcal{R}` and the Prandtl number :math:`\mathcal{P}`.
:math:`\omega_\mathrm{drag}` is the dimensionless drag frequency,
:math:`\mu = \cos\theta`, :math:`\tilde{\mu}=\alpha \mu`,
:math:`H_\ell(\tilde{\mu})` are the Hermite polynomials:

.. math::

    \begin{eqnarray}
    H_0 &=& 1\\
    H_1 &=& 2\tilde{\mu}\\
    H_3 &=& 8\tilde{\mu}^3 - 12 \tilde{\mu}\\
    H_4 &=& 16\tilde{\mu}^4 - 48\tilde{\mu}^2 + 12.
    \end{eqnarray}

Phase curve
-----------

We can then compute the thermal flux emitted by the planet at any orbital phase
:math:`\xi`, which is normalized from zero at secondary eclipse and
:math:`\pm\pi` at transit:

.. math::

       F_p = R_p^2 \int_{-\xi-\pi/2}^{-\xi+\pi/2} \int_0^\pi \mathcal{B}(T) \sin^2\theta \cos(\phi + \xi)d\theta d\phi

given the blackbody function defined as:

.. math::

       \mathcal{B}(T) = \int_{\lambda_1}^{\lambda_2} B_\lambda(T(\theta, \phi)) \mathcal{F}_\lambda d\lambda

where :math:`T(\theta, \phi)` is the temperature map described with the
perturbed spherical harmonic basis functions in the previous section, and
:math:`\mathcal{F_\lambda}` is the filter throughput.

The observation that we seek to fit is the infrared phase curve of the exoplanet,
typically normalized as a ratio of the thermal flux of the planet normalized by
the thermal flux of the star, like so:

.. math::

    \frac{F_p}{F_\star} = \left(\frac{R_p}{R_\star}\right)^2 \int_{-\xi-\pi/2}^{-\xi+\pi/2} \int_0^\pi \frac{\mathcal{B}(T_p)}{\mathcal{B}(T_s)} d\theta d\phi

