***************
Reflected Light
***************

kelp implements the reflected light phase curve parameterization for an
homogeneous planetary atmospheres as derived by
`Heng, Morris & Kitzmann (HMK; 2021) <https://arxiv.org/abs/2103.02673>`_.
The HMK reflected light model is a function of two fundamental
parameters which describe the scattering in the planetary atmosphere:
the single-scattering albedo :math:`\omega`; and the scattering
asymmetry factor :math:`g`.

Below, we'll show how to use the reflected light
formulation for a homogeneous planetary atmosphere to fit the Kepler phase curve
of HAT-P-7 b; then we'll move on to the case of an inhomogeneous atmosphere with
the phase curve of Kepler-7 b.


Homogeneous reflectivity map
----------------------------

We can implement a model for the Kepler phase curve of HAT-P-7 b, which
combines all aspects of a challenging phase curve (except atmospheric
inhomogeneity, for this see the next section). HAT-P-7 b is a 2700 K planet with
strong ellipsoidal variations, nearly-equal contributions to the Kepler phase
curve from thermal emission and reflected light, plus detectable Doppler
beaming. We'll implement each of these contributions in the example below, and
solve for the maximum-likelihood single-scattering albedo and scattering
asymmetry factor, as well as the amplitudes of the Doppler beaming and
ellipsoidal variations.

First we'll import the necessary packages (there are quite a few!):

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    import theano.tensor as tt
    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx

    from corner import corner

    from lightkurve import search_lightcurve

    import astropy.units as u
    from astropy.constants import R_sun, R_earth
    from astropy.stats import sigma_clip, mad_std

    from kelp import Filter
    from kelp.theano import reflected_phase_curve, thermal_phase_curve

Note that we have imported the phase curve functions from the ``theano`` module,
because we will be using PyMC3 to construct the model and draw posterior
samples.

Now we will define the system parmaeters that we will use to construct the phase
curvec and phase fold the light curve:

.. code-block:: python

    floatX = 'float64'
    t0 = 2454954.357462  # Bonomo 2017
    period = 2.204740    # Stassun 2017
    rp = 16.9 * R_earth  # Stassun 2017
    rstar = 1.991 * R_sun  # Berger 2017
    a = 4.13 * rstar     # Stassun 2017
    duration = 4.0398 / 24  # Holczer 2016
    b = 0.4960           # Esteves 2015
    rho_star = 0.27 * u.g / u.cm ** 3  # Stassun 2017
    T_s = 6449           # Berger 2018

    a_rs = float(a / rstar)
    a_rp = float(a / rp)
    rp_rstar = float(rp / rstar)
    eclipse_half_dur = duration / period / 2

Now we will use `~lightkurve.search.search_lightcurve` to download the long
cadence light curve from Quarter 10 of the Kepler observations of HAT-P-7 b.
We'll then "flatten" the light curve, which applies a savgol filter to the light
curve to do some long-term detrending on the out-of-transit portions of the
phase curve. We will also sigma-clip the fluxes to remove outliers:

.. code-block:: python

    lcf = search_lightcurve(
        "HAT-P-7", mission="Kepler", cadence="long", quarter=10
    ).download_all()

    slc = lcf.stitch()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
                phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    slc = slc.flatten(
        polyorder=3, break_tolerance=10, window_length=1001, mask=~out_of_transit
    ).remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
                phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    sc = sigma_clip(
        np.ascontiguousarray(slc.flux[out_of_transit], dtype=np.float64),
        maxiters=100, sigma=8, stdfunc=mad_std
    )

Next we will compute the masked phases, times, and the normalized fluxes
:math:`F_p/F_\mathrm{star}` in units of ppm:

.. code-block:: python

    phase = np.ascontiguousarray(
        phases[out_of_transit][~sc.mask], dtype=np.float64
    )
    time = np.ascontiguousarray(
        slc.time.jd[out_of_transit][~sc.mask], dtype=np.float64
    )

    bin_in_eclipse = np.abs(phase - 0.5) < eclipse_half_dur
    unbinned_flux_mean = np.mean(sc[~sc.mask].data)

    unbinned_flux_mean_ppm = 1e6 * (unbinned_flux_mean - 1)
    flux_normed = np.ascontiguousarray(
        1e6 * (sc[~sc.mask].data / unbinned_flux_mean - 1.0), dtype=np.float64
    )
    flux_normed_err = np.ascontiguousarray(
        1e6 * slc.flux_err[out_of_transit][~sc.mask].value, dtype=np.float64
    )

Now we will median-bin the phase folded Kepler light curve:

.. code-block:: python

    bins = 100
    bs = binned_statistic(
        phase, flux_normed, statistic=np.median, bins=bins
    )

    bs_err = binned_statistic(
        phase, flux_normed_err,
        statistic=lambda x: 3 * np.median(x) / len(x) ** 0.5, bins=bins
    )

    binphase = 0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
    # Normalize the binned fluxes by the in-eclipse flux:
    binflux = bs.statistic - np.median(bs.statistic[np.abs(binphase - 0.5) < 0.01])
    binerror = bs_err.statistic

Now we will use the `~kelp.registries.Filter` object to define the filter
transmittance curve for Kepler:

.. code-block:: python

    filt = Filter.from_name("Kepler")
    filt.bin_down(6)   # This speeds up integration by orders of magnitude
    filt_wavelength, filt_trans = filt.wavelength.to(u.m).value, filt.transmittance


Next we construct the PyMC3 model. This is a long block of code, so let's
jump straight into in-line comments:

.. code-block:: python

    with pm.Model() as model:
        # Define a Keplerian orbit using `exoplanet`:
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=0, b=b, rho_star=rho_star.to(u.g / u.cm ** 3),
            r_star=float(rstar / R_sun)
        )

        # Compute the eclipse model (no limb-darkening):
        eclipse_light_curves = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
            orbit=orbit._flip(rp_rstar), r=orbit.r_star,
            t=binphase * period,
            texp=(30 * u.min).to(u.d).value
        )

        # Normalize the eclipse model to unity out of eclipse and
        # zero in-eclipse
        eclipse = 1 + pm.math.sum(eclipse_light_curves, axis=-1)

        # Define reflected light phase curve model according to
        # Heng, Morris & Kitzmann (2021)
        omega = pm.Uniform('omega', lower=0, upper=1)
        g = pm.TruncatedNormal('g', lower=0, upper=1, mu=0, sigma=0.4)

        reflected_ppm, A_g, q = reflected_phase_curve(binphase, omega, g, a_rp)

        # Define the ellipsoidal variation parameterization (simple sinusoid)
        ellipsoidal_amp = pm.Uniform('ellip_amp', lower=0, upper=50)
        ellipsoidal_model_ppm = - ellipsoidal_amp * tt.cos(
            4 * np.pi * (binphase - 0.5)) + ellipsoidal_amp

        # Define the doppler variation parameterization (simple sinusoid)
        doppler_amp = pm.Uniform('doppler_amp', lower=0, upper=50)
        doppler_model_ppm = doppler_amp * tt.sin(
            2 * np.pi * binphase)

        # Define the thermal emission model according to description in
        # Morris et al. (in prep)
        xi = 2 * np.pi * (binphase - 0.5)
        n_phi = 75
        n_theta = 5
        phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi, dtype=floatX)
        theta = np.linspace(0, np.pi, n_theta, dtype=floatX)
        theta2d, phi2d = np.meshgrid(theta, phi)

        ln_C_11_kepler = -2.6
        C_11_kepler = tt.exp(ln_C_11_kepler)
        hml_eps = 0.72
        hml_f = (2/3 - hml_eps * 5 / 12) ** 0.25
        delta_phi = 0

        A_B = pm.Deterministic('A_B', q * A_g)

        # Compute the thermal phase curve with zero phase offset
        thermal, T = thermal_phase_curve(
            xi, delta_phi, 4.5, 0.575, C_11_kepler, T_s, a_rs, 1 / a_rp, A_B,
            theta2d, phi2d, filt_wavelength, filt_trans, 2 ** -0.5
        )

        # Define the composite phase curve model
        flux_norm = eclipse * (
                reflected_ppm + ellipsoidal_model_ppm +
                doppler_model_ppm + 1e6 * thermal
        )

        # Keep track of the geometric albedo and integral phase function at
        # each step in the chain
        pm.Deterministic('A_g', A_g)
        pm.Deterministic('q', q)

        # Define the likelihood
        pm.Normal('obs', mu=flux_norm, sigma=binerror, observed=binflux)

        # Optimize a fast maximum-likelihood solution to seed posterior draws:
        map_soln = pm.find_MAP()

Now our model is set up, and we are ready to draw posterior samples from the
model given the data, which we will do with
`pymc3-ext <https://github.com/exoplanet-dev/pymc3-ext>`_ for the most efficient
posterior sampling of our degenerate phase curve parameterization. This will take
up to a minute:

.. code-block:: python

    with model:
        trace = pmx.sample(
            draws=100, tune=10, start=map_soln, compute_convergence_checks=False,
            target_accept=0.95, initial_accept=0.2,
            return_inferencedata=False
        )

Let's finally plot the final results:

.. code-block:: python

    with model:
        corner(pm.trace_to_dataframe(trace));
        plt.show()

    plt.errorbar(binphase, binflux, binerror, fmt='.', color='k', ecolor='silver')

    with model:
        for sample in xo.get_samples_from_trace(trace, size=10):
            plt.plot(binphase, xo.eval_in_model(flux_norm, sample), alpha=0.5,
                     color='r', zorder=10)

        plt.plot(binphase, xo.eval_in_model(reflected_ppm, sample),
                 color='DodgerBlue', zorder=10, label='reflected')
        plt.plot(binphase, xo.eval_in_model(1e6 * thermal, sample), color='m',
                 zorder=10, label='thermal')
        plt.plot(binphase, xo.eval_in_model(ellipsoidal_model_ppm, sample),
                 color='b', zorder=10, label='ellipsoidal')
        plt.plot(binphase, xo.eval_in_model(doppler_model_ppm, sample), color='g',
                 zorder=10, label='doppler')

    plt.legend()
    plt.ylim([-30, 110])
    for sp in ['right', 'top']:
        plt.gca().spines[sp].set_visible(False)
    plt.gca().set(xlabel='Phase', ylabel='$F_p/F_\mathrm{star}$ [ppm]',
                  title='HAT-P-7 b')
    plt.show()

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    import theano.tensor as tt
    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx

    from corner import corner

    from lightkurve import search_lightcurve

    import astropy.units as u
    from astropy.constants import R_sun, R_earth
    from astropy.stats import sigma_clip, mad_std

    from kelp import Filter
    from kelp.theano import reflected_phase_curve, thermal_phase_curve

    floatX = 'float64'
    t0 = 2454954.357462  # Bonomo 2017
    period = 2.204740    # Stassun 2017
    rp = 16.9 * R_earth  # Stassun 2017
    rstar = 1.991 * R_sun  # Berger 2017
    a = 4.13 * rstar     # Stassun 2017
    duration = 4.0398 / 24  # Holczer 2016
    b = 0.4960           # Esteves 2015
    rho_star = 0.27 * u.g / u.cm ** 3  # Stassun 2017
    T_s = 6449           # Berger 2018

    a_rs = float(a / rstar)
    a_rp = float(a / rp)
    rp_rstar = float(rp / rstar)
    eclipse_half_dur = duration / period / 2

    lcf = search_lightcurve(
        "HAT-P-7", mission="Kepler", cadence="long", quarter=10
    ).download_all()

    slc = lcf.stitch()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
                phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    slc = slc.flatten(
        polyorder=3, break_tolerance=10, window_length=1001, mask=~out_of_transit
    ).remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
                phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    sc = sigma_clip(
        np.ascontiguousarray(slc.flux[out_of_transit], dtype=np.float64),
        maxiters=100, sigma=8, stdfunc=mad_std
    )

    phase = np.ascontiguousarray(
        phases[out_of_transit][~sc.mask], dtype=np.float64
    )
    time = np.ascontiguousarray(
        slc.time.jd[out_of_transit][~sc.mask], dtype=np.float64
    )

    bin_in_eclipse = np.abs(phase - 0.5) < eclipse_half_dur
    unbinned_flux_mean = np.mean(sc[~sc.mask].data)  # .mean()

    unbinned_flux_mean_ppm = 1e6 * (unbinned_flux_mean - 1)
    flux_normed = np.ascontiguousarray(
        1e6 * (sc[~sc.mask].data / unbinned_flux_mean - 1.0), dtype=np.float64
    )
    flux_normed_err = np.ascontiguousarray(
        1e6 * slc.flux_err[out_of_transit][~sc.mask].value, dtype=np.float64
    )

    bins = 100
    bs = binned_statistic(
        phase, flux_normed, statistic=np.median, bins=bins
    )

    bs_err = binned_statistic(
        phase, flux_normed_err,
        statistic=lambda x: 3 * np.median(x) / len(x) ** 0.5, bins=bins
    )

    binphase = 0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
    # Normalize the binned fluxes by the in-eclipse flux:
    binflux = bs.statistic - np.median(bs.statistic[np.abs(binphase - 0.5) < 0.01])
    binerror = bs_err.statistic

    filt = Filter.from_name("Kepler")
    filt.bin_down(6)   # This speeds up integration by orders of magnitude
    filt_wavelength, filt_trans = filt.wavelength.to(u.m).value, filt.transmittance

    with pm.Model() as model:
        # Define a Keplerian orbit using `exoplanet`:
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=0, b=b, rho_star=rho_star.to(u.g / u.cm ** 3),
            r_star=float(rstar / R_sun)
        )

        # Compute the eclipse model (no limb-darkening):
        eclipse_light_curves = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
            orbit=orbit._flip(rp_rstar), r=orbit.r_star,
            t=binphase * period,
            texp=(30 * u.min).to(u.d).value
        )

        # Normalize the eclipse model to unity out of eclipse and
        # zero in-eclipse
        eclipse = 1 + pm.math.sum(eclipse_light_curves, axis=-1)

        # Define reflected light phase curve model according to
        # Heng, Morris & Kitzmann (2021)
        omega = pm.Uniform('omega', lower=0, upper=1)
        g = pm.TruncatedNormal('g', lower=0, upper=1, mu=0, sigma=0.4)

        reflected_ppm, A_g, q = reflected_phase_curve(binphase, omega, g, a_rp)

        # Define the ellipsoidal variation parameterization (simple sinusoid)
        ellipsoidal_amp = pm.Uniform('ellip_amp', lower=0, upper=50)
        ellipsoidal_model_ppm = - ellipsoidal_amp * tt.cos(
            4 * np.pi * (binphase - 0.5)) + ellipsoidal_amp

        # Define the doppler variation parameterization (simple sinusoid)
        doppler_amp = pm.Uniform('doppler_amp', lower=0, upper=50)
        doppler_model_ppm = doppler_amp * tt.sin(
            2 * np.pi * binphase)

        # Define the thermal emission model according to description in
        # Morris et al. (in prep)
        xi = 2 * np.pi * (binphase - 0.5)
        n_phi = 75
        n_theta = 5
        phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi, dtype=floatX)
        theta = np.linspace(0, np.pi, n_theta, dtype=floatX)
        theta2d, phi2d = np.meshgrid(theta, phi)

        ln_C_11_kepler = -2.6
        C_11_kepler = tt.exp(ln_C_11_kepler)
        hml_eps = 0.72
        hml_f = (2/3 - hml_eps * 5 / 12) ** 0.25
        delta_phi = 0

        A_B = pm.Deterministic('A_B', q * A_g)

        # Compute the thermal phase curve with zero phase offset
        thermal, T = thermal_phase_curve(
            xi, delta_phi, 4.5, 0.575, C_11_kepler, T_s, a_rs, 1 / a_rp, A_B,
            theta2d, phi2d, filt_wavelength, filt_trans, 2 ** -0.5
        )

        # Define the composite phase curve model
        flux_norm = eclipse * (
                reflected_ppm + ellipsoidal_model_ppm +
                doppler_model_ppm + 1e6 * thermal
        )

        # Keep track of the geometric albedo and integral phase function at
        # each step in the chain
        pm.Deterministic('A_g', A_g)
        pm.Deterministic('q', q)

        # Define the likelihood
        pm.Normal('obs', mu=flux_norm, sigma=binerror, observed=binflux)

        # Optimize a fast maximum-likelihood solution to seed posterior draws:
        map_soln = pm.find_MAP()

    with model:
        trace = pmx.sample(
            draws=1000, tune=50, start=map_soln, compute_convergence_checks=False,
            target_accept=0.95, initial_accept=0.2,
            return_inferencedata=False,
            cores=1, chains=1
        )

    with model:
        corner(pm.trace_to_dataframe(trace));
        plt.show()

    plt.errorbar(binphase, binflux, binerror, fmt='.', color='k', ecolor='silver')

    with model:
        for i, sample in enumerate(xo.get_samples_from_trace(trace, size=10)):
            plt.plot(binphase, xo.eval_in_model(flux_norm, sample), alpha=0.5,
                     color='r', zorder=10)

            plt.plot(binphase, xo.eval_in_model(reflected_ppm, sample),
                     color='DodgerBlue', zorder=10, label='reflected' if i==0 else None)
            plt.plot(binphase, xo.eval_in_model(1e6 * thermal, sample), color='m',
                     zorder=10, label='thermal' if i==0 else None)
            plt.plot(binphase, xo.eval_in_model(ellipsoidal_model_ppm, sample),
                     color='b', zorder=10, label='ellipsoidal' if i==0 else None)
            plt.plot(binphase, xo.eval_in_model(doppler_model_ppm, sample), color='g',
                     zorder=10, label='doppler' if i==0 else None)

    plt.legend()
    plt.ylim([-30, 120])
    for sp in ['right', 'top']:
        plt.gca().spines[sp].set_visible(False)
    plt.gca().set(xlabel='Phase', ylabel='$F_p/F_\mathrm{star}$ [ppm]',
                  title='HAT-P-7 b')
    plt.show()

In the above corner plot, you'll see the joint posterior correlation plots
for each of the free parameters in the fit, including the single-scattering
albedo :math:`\omega`, the scattering asymmetry factor :math:`g`, and the derived
parameters including the Bond albedo :math:`A_B`, the geometric albedo
:math:`A_g`, and the integral phase function :math:`q`.

You'll also see a plot above with several draws from the posteriors for each
parameter plotted in light-curve space, showing the range of plausible
contributions from each phase curve component shown in different colors.

Inhomogeneous reflectivity map
------------------------------

Kepler-7 b is a warm Jupiter with an insignificant thermal emission contribution
to the Kepler phase curve, but with a significant phase curve asymmetry,
possibly resulting from an inhomogeneous albedo distribution on the surface of
the planet. In this example, we'll give two parameters for the single scattering
albedo in the brighter and darker regions, one scattering asymmetry factor, and
one geometric albedo.

.. note::

    The analysis presented in this documentation is meant to be a
    quick-and-dirty example that demonstrates the capabilities of kelp, but is
    not meant to precisely reproduce the results of Heng, Morris & Kitzmann
    (2021). The results presented in that paper require a more complex and
    expensive model, so we opt to show a simpler and cheaper model for this
    tutorial.

As with the previous example, we begin with some imports:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    import theano
    floatX = 'float64'
    theano.config.floatX = floatX

    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx
    from corner import corner

    from lightkurve import search_lightcurve

    import astropy.units as u
    from astropy.constants import R_jup, R_sun
    from astropy.stats import sigma_clip, mad_std

    from kelp.theano import reflected_phase_curve_inhomogeneous

We also define the system parameters:

.. code-block:: python

    t0 = 2454967.27687  # Esteves et al. 2015
    period = 4.8854892  # Esteves et al. 2015
    rp = 1.622 * R_jup  # Esteves et al. 2015
    rstar = 1.966 * R_sun  # ±0.013 (NASA Exoplanet Archive)
    a = 0.06067 * u.AU  # Esteves et al. 2015
    duration = 5.1313 / 24  # Morton et al. 2016
    b = 0.5599  # Esteves et al. 2015 +0.0045-0.0046
    rho_star = 0.238 * u.g / u.cm ** 3  # Southworth et al. 2012 ±0.010
    T_s = 5933  # NASA Exoplanet Archive
    a_rs = float(a / rstar)
    a_rp = float(a / rp)
    rp_rstar = float(rp / rstar)
    eclipse_half_dur = duration / period / 2


And we use ``lightkurve`` to download the entire Kepler-7 b long cadence light
curve over all quarters:

.. code-block:: python

    lcf = search_lightcurve(
        "Kepler-7", mission="Kepler", cadence="long",
    ).download_all()

    slc = lcf.stitch().remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.2 * eclipse_half_dur) | (
                phases > 1 - 1.2 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

Next we sigma clip, normalize, and bin the Kepler-7 b observations:

.. code-block:: python

    sc = sigma_clip(
        np.ascontiguousarray(slc.flux[out_of_transit], dtype=np.float64),
        maxiters=100, sigma=8, stdfunc=mad_std
    )

    phase = np.ascontiguousarray(phases[out_of_transit][~sc.mask], dtype=np.float64)
    time = np.ascontiguousarray(slc.time.jd[out_of_transit][~sc.mask],
                                dtype=np.float64)

    bin_in_eclipse = np.abs(phase - 0.5) < eclipse_half_dur
    unbinned_flux_mean = np.mean(sc[~sc.mask].data)

    unbinned_flux_mean_ppm = 1e6 * (unbinned_flux_mean - 1)
    flux_normed = np.ascontiguousarray(
        1e6 * (sc[~sc.mask].data / unbinned_flux_mean - 1.0), dtype=np.float64)
    flux_normed_err = np.ascontiguousarray(
        1e6 * slc.flux_err[out_of_transit][~sc.mask].value, dtype=np.float64)

    bins = 100
    bs = binned_statistic(phase, flux_normed, statistic=np.median, bins=bins)

    bs_err = binned_statistic(phase, flux_normed_err,
                              statistic=lambda x: np.median(x) / len(x) ** 0.5,
                              bins=bins)

    binphase = 0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
    binflux = bs.statistic - np.median(bs.statistic[np.abs(binphase - 0.5) < 0.01])
    binerror = bs_err.statistic

Now we construct a the phase curve model. This time there are only two
components: the eclipse and the inhomogeneous reflected light phase curve.

For the purposes of making this example fast, we will fix the single scattering
albedo of the darker region to :math:`\omega_0 = 0` and the single scattering
albedo of the more reflective region to :math:`\omega^\prime = 0.95`, and fix
the start and stop longitudes of the darker region ``x1, x2 = 0, 0.8`` radians.
We'll also add a constant ``offset`` term to the entire phase curve to account
for the light curve normalization.

.. code-block:: python

    with pm.Model() as model:
        # Define a Keplerian orbit:
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=0, b=b, rho_star=rho_star.to(u.g / u.cm ** 3),
            r_star=float(rstar / R_sun)
        )

        # Compute the eclipse model (no LD):
        eclipse_light_curves_kepler = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
            orbit=orbit._flip(rp_rstar), r=orbit.r_star,
            t=binphase.astype(floatX) * period,
            texp=(30 * u.min).to(u.d).value
        )

        # Normalize the eclipse model:
        eclipse_kepler = 1 + pm.math.sum(eclipse_light_curves_kepler, axis=-1)

        omega_0 = 0
        omega_prime = 0.95

        # Define the start and stop longitudes of the darker region
        x1 = 0  # [radians]
        x2 = 0.8  # [radians]

        # Sample for the geometric albedo
        A_g = pm.Uniform('A_g', lower=0, upper=1)

        # construct an inhomogeneous reflected light phase curve model
        flux_ratio_ppm, g, q = reflected_phase_curve_inhomogeneous(
            binphase, omega_0, omega_prime, x1, x2, A_g, a_rp
        )

        # Apply a constant offset to the entire phase curve to account for normalization bias
        offset = pm.Uniform('offset', lower=-20, upper=20)

        # Construct a composite phase curve model
        flux_norm = eclipse_kepler * flux_ratio_ppm + offset

        # Keep track of the q and g values at each step in the chains
        pm.Deterministic('q', q)
        pm.Deterministic('g', g)

        # Construct our likelihood
        pm.Normal('obs_kepler', mu=flux_norm, sigma=binerror, observed=binflux)

        # Solve for a quick best-fit using scipy:
        map_soln = pm.find_MAP()

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(binphase, binflux, 'k.')
        ax.plot(binphase, xo.eval_in_model(flux_norm, map_soln), 'r', lw=2,
                label='composite')
        plt.show()

Now that the model is constructed, we're ready to sample from the posterior
distribution for the geometric albedo, the integral phase function, and the
scattering asymmetry factor, which we do again with
`pymc3-ext <https://github.com/exoplanet-dev/pymc3-ext>`_ for the most efficient
posterior sampling of our degenerate phase curve parameterization:

.. code-block:: python

    with model:
        trace = pmx.sample(
            draws=1000, tune=100, compute_convergence_checks=False,
            target_accept=0.95, initial_accept=0.2,
            return_inferencedata=False,
        )

We can plot the results with the following commands:

.. code-block:: python

    corner(pm.trace_to_dataframe(trace));
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.errorbar(binphase, binflux, binerror, fmt='.', color='k', ecolor='silver')
    with model:
        for i, sample in enumerate(xo.get_samples_from_trace(trace, size=5)):
            ax.plot(binphase, xo.eval_in_model(flux_norm, sample), 'r')

    ax.set(xlabel='Phase', ylabel='$F_p/F_\mathrm{star}$ [ppm]',
              title='Kepler-7 b')

    for sp in ['right', 'top']:
        ax.spines[sp].set_visible(False)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    import theano
    floatX = 'float64'
    theano.config.floatX = floatX

    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx
    from corner import corner

    from lightkurve import search_lightcurve

    import astropy.units as u
    from astropy.constants import R_jup, R_sun

    from kelp.theano import reflected_phase_curve_inhomogeneous


    t0 = 2454967.27687  # Esteves et al. 2015
    period = 4.8854892  # Esteves et al. 2015
    rp = 1.622 * R_jup  # Esteves et al. 2015
    rstar = 1.966 * R_sun  # ±0.013 (NASA Exoplanet Archive)
    a = 0.06067 * u.AU  # Esteves et al. 2015
    duration = 5.1313 / 24  # Morton et al. 2016
    b = 0.5599  # Esteves et al. 2015 +0.0045-0.0046
    rho_star = 0.238 * u.g / u.cm ** 3  # Southworth et al. 2012 ±0.010
    T_s = 5933  # NASA Exoplanet Archive
    a_rs = float(a / rstar)
    a_rp = float(a / rp)
    rp_rstar = float(rp / rstar)
    eclipse_half_dur = duration / period / 2


    lcf = search_lightcurve(
        "Kepler-7", mission="Kepler", cadence="long",
    ).download_all()


    slc = lcf.stitch().remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.2 * eclipse_half_dur) | (
                phases > 1 - 1.2 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    from astropy.stats import sigma_clip, mad_std

    sc = sigma_clip(
        np.ascontiguousarray(slc.flux[out_of_transit], dtype=np.float64),
        maxiters=100, sigma=8, stdfunc=mad_std
    )

    phase = np.ascontiguousarray(phases[out_of_transit][~sc.mask], dtype=np.float64)
    time = np.ascontiguousarray(slc.time.jd[out_of_transit][~sc.mask],
                                dtype=np.float64)

    bin_in_eclipse = np.abs(phase - 0.5) < eclipse_half_dur
    unbinned_flux_mean = np.mean(sc[~sc.mask].data)  # .mean()

    unbinned_flux_mean_ppm = 1e6 * (unbinned_flux_mean - 1)
    flux_normed = np.ascontiguousarray(
        1e6 * (sc[~sc.mask].data / unbinned_flux_mean - 1.0), dtype=np.float64)
    flux_normed_err = np.ascontiguousarray(
        1e6 * slc.flux_err[out_of_transit][~sc.mask].value, dtype=np.float64)

    bins = 100
    bs = binned_statistic(phase, flux_normed, statistic=np.median, bins=bins)

    bs_err = binned_statistic(phase, flux_normed_err,
                              statistic=lambda x: np.median(x) / len(x) ** 0.5,
                              bins=bins)

    binphase = 0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
    binflux = bs.statistic - np.median(bs.statistic[np.abs(binphase - 0.5) < 0.01])
    binerror = bs_err.statistic


    with pm.Model() as model:
        # Define a Keplerian orbit:
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=0, b=b, rho_star=rho_star.to(u.g / u.cm ** 3),
            r_star=float(rstar / R_sun)
        )

        # Compute the eclipse model (no LD):
        eclipse_light_curves_kepler = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
            orbit=orbit._flip(rp_rstar), r=orbit.r_star,
            t=binphase.astype(floatX) * period,
            texp=(30 * u.min).to(u.d).value
        )

        # Normalize the eclipse model:
        eclipse_kepler = 1 + pm.math.sum(eclipse_light_curves_kepler, axis=-1)

        omega_0 = 0
        omega_prime = 0.95

        # Define the start and stop longitudes of the darker region
        x1 = 0  # [radians]
        x2 = 0.8  # [radians]

        # Sample for the geometric albedo
        A_g = pm.Uniform('A_g', lower=0, upper=1)

        # construct an inhomogeneous reflected light phase curve model
        flux_ratio_ppm, g, q = reflected_phase_curve_inhomogeneous(
            binphase, omega_0, omega_prime, x1, x2, A_g, a_rp
        )

        # Apply a constant offset to the entire phase curve to account for normalization bias
        offset = pm.Uniform('offset', lower=-20, upper=20)

        # Construct a composite phase curve model
        flux_norm = eclipse_kepler * flux_ratio_ppm + offset

        # Keep track of the q and g values at each step in the chains
        pm.Deterministic('q', q)
        pm.Deterministic('g', g)

        # Construct our likelihood
        pm.Normal('obs_kepler', mu=flux_norm, sigma=binerror, observed=binflux)

        # Solve for a quick best-fit using scipy:
        map_soln = pm.find_MAP()

    with model:
        trace = pmx.sample(
            draws=1000, tune=100, compute_convergence_checks=False,
            target_accept=0.95, initial_accept=0.2,
            return_inferencedata=False,
            cores=1, chains=1
        )


    corner(pm.trace_to_dataframe(trace));
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.errorbar(binphase, binflux, binerror, fmt='.', color='k', ecolor='silver')
    with model:
        for i, sample in enumerate(xo.get_samples_from_trace(trace, size=5)):
            ax.plot(binphase, xo.eval_in_model(flux_norm, sample), 'r',
                       label='composite' if i == 0 else None)

    ax.set(xlabel='Phase', ylabel='$F_p/F_\mathrm{star}$ [ppm]',
              title='Kepler-7 b')

    for sp in ['right', 'top']:
        ax.spines[sp].set_visible(False)

In the corner plot above, you'll see that the solution has a geometric albedo
near :math:`A_g = 0.24`, and a scattering asymmetry factor near zero. The
draws from the posteriors for each parameter produce phase curve models that
are asymmetric (as we intended) which match the shape of the observations well,
despite having only a few free parameters.
