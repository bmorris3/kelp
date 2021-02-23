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

    F_p/F_\star = \left(\frac{R_p}{R_\star}\right)^2 \int_0^\pi \int_{-\xi-\pi/2}^{-\xi+\pi/2} \frac{I_p(\theta, \phi)}{I_\star(\theta, \phi)} \cos(\phi+\xi) \sin^2(\theta) d\phi d\theta \label{eqn:diskint}

where the intensity :math:`I` is given by

.. math::

    I = \int \mathcal{F}_\lambda \mathcal{B}_\lambda(T(\theta, \phi) d\lambda

for a filter bandpass transmittance function :math:`\mathcal{F}_\lambda`.

Example temperature fields
--------------------------

The first several terms in the spherical harmonic expansion of the temperature
map in the :math:`h_{m\ell}` basis. Each subplot shows the temperature perturbation (purple
to yellow is cold to hot) as a function of latitude and longitude (shown in
Mollweide projections such that the substellar longitude is in the center of
the plot). The :math:`\ell = 0` terms are always zero. The :math:`m=2` terms are asymmetric
about the equator and therefore do not represent typical GCM results, so we
keep all :math:`m=2` terms fixed to zero in the subsequent fits. These maps were
generated with :math:`\alpha=0.6` and :math:`\omega_\mathrm{drag} = 4.5`.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.units as u
    from kelp import Model, Filter, Planet

    import astropy.units as u
    from astropy.constants import R_jup, R_sun

    filt = Filter.from_name('IRAC 1')

    n_phi = 100
    n_theta = 100
    lmax = 3

    p = Planet.from_name('HD 189733')

    def indexer(m, l):
        C_ml = [[0],
                [0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
        C_ml[m][l] = 1
        return C_ml

    def generate_temp_map(a, m, l):
        hotspot_offset = 0
        C_ml = indexer(m, l)


        alpha = 0.6
        omega_drag = 4.5
        rp_a = float(R_jup / (a * u.AU))
        a_rs = float(a * u.AU / R_sun)
        A_B = 0
        T_s = 5770

        model = Model(hotspot_offset, alpha, omega_drag, A_B,
                       C_ml, lmax, a_rs=a_rs, rp_a=rp_a, T_s=T_s, filt=filt)

        phase_offset = np.pi / 2
        f = 1 / np.sqrt(2)
        T, theta, phi = model.temperature_map(n_theta, n_phi, f=f)
        return T, theta, phi

    cml_example = indexer(1, 0)

    fig = plt.figure(figsize=(10, 4))

    ax = np.array(
        [fig.add_subplot(
            len(cml_example),
            len(cml_example[-1]),
            1+i,
            projection="mollweide")
         for i in range(len(cml_example[-1]) * len(cml_example))]
    ).reshape((len(cml_example), len(cml_example[-1])))

    for m in range(0, lmax + 1):
        for l in range(-m, m + 1):
            temperature, theta, phi = generate_temp_map(0.17, m, l)
            phirange = (-np.pi <= phi) & (np.pi >= phi)
            cax = ax[m, l + len(cml_example[-1])//2].pcolormesh(
                phi[phirange], (theta - np.pi/2), temperature[:, phirange],
                rasterized=True
            )
            ax[m, l + len(cml_example[-1])//2].set_title(f'$m = {m},\,\ell = {l}$')
            ax[m, l + len(cml_example[-1])//2].grid(False)

    for i in range(len(cml_example)):
        for j in range(len(cml_example[-1])):
            ax[i, j].axis('off')

    plt.tight_layout(h_pad=0.8, w_pad=0.3)
    plt.show()

Below is the same as above, but this time for :math:`\alpha=0.9` and
:math:`\omega_\mathrm{drag} = 1.5` -- note that when the drag is set to a
smaller value, the chevron shape becomes more pronounced as a
perturbation on the temperature maps with :math:`\ell \neq 0`.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.units as u
    from kelp import Model, Filter, Planet

    import astropy.units as u
    from astropy.constants import R_jup, R_sun

    filt = Filter.from_name('IRAC 1')

    n_phi = 100
    n_theta = 100
    lmax = 3

    p = Planet.from_name('HD 189733')

    def indexer(m, l):
        C_ml = [[0],
                [0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
        C_ml[m][l] = 1
        return C_ml

    def generate_temp_map(a, m, l):
        hotspot_offset = 0
        C_ml = indexer(m, l)


        alpha = 0.9
        omega_drag = 1.5
        rp_a = float(R_jup / (a * u.AU))
        a_rs = float(a * u.AU / R_sun)
        A_B = 0
        T_s = 5770

        model = Model(hotspot_offset, alpha, omega_drag, A_B,
                       C_ml, lmax, a_rs=a_rs, rp_a=rp_a, T_s=T_s, filt=filt)

        phase_offset = np.pi / 2
        f = 1 / np.sqrt(2)
        T, theta, phi = model.temperature_map(n_theta, n_phi, f=f)
        return T, theta, phi

    cml_example = indexer(1, 0)

    fig = plt.figure(figsize=(10, 4))

    ax = np.array(
        [fig.add_subplot(
            len(cml_example),
            len(cml_example[-1]),
            1+i,
            projection="mollweide")
         for i in range(len(cml_example[-1]) * len(cml_example))]
    ).reshape((len(cml_example), len(cml_example[-1])))

    for m in range(0, lmax + 1):
        for l in range(-m, m + 1):
            temperature, theta, phi = generate_temp_map(0.17, m, l)
            phirange = (-np.pi <= phi) & (np.pi >= phi)
            cax = ax[m, l + len(cml_example[-1])//2].pcolormesh(
                phi[phirange], (theta - np.pi/2), temperature[:, phirange],
                rasterized=True
            )
            ax[m, l + len(cml_example[-1])//2].set_title(f'$m = {m},\,\ell = {l}$')
            ax[m, l + len(cml_example[-1])//2].grid(False)

    for i in range(len(cml_example)):
        for j in range(len(cml_example[-1])):
            ax[i, j].axis('off')

    plt.tight_layout(h_pad=0.8, w_pad=0.3)
    plt.show()
