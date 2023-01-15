***************
Reflected Light
***************

kelp implements the reflected light phase curve parameterization for an
homogeneous planetary atmospheres as derived by
`Heng, Morris & Kitzmann (HMK; 2021)
<https://ui.adsabs.harvard.edu/abs/2021NatAs...5.1001H/abstract>`_.
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

    import arviz
    from corner import corner
    import numpyro

    # Set the number of cores on your machine for parallelism:
    cpu_cores = 1
    numpyro.set_host_device_count(cpu_cores)

    from numpyro.infer import MCMC, NUTS
    from numpyro import distributions as dist

    from jax import numpy as jnp
    from jax.random import PRNGKey, split

    from lightkurve import search_lightcurve
    from batman import TransitModel

    import astropy.units as u
    from astropy.constants import R_sun, R_earth
    from astropy.stats import sigma_clip, mad_std

    from kelp import Filter, Planet
    from kelp.jax import reflected_phase_curve, thermal_phase_curve

Note that we have imported the phase curve functions from the ``jax`` module,
because we will be using numpyro to construct the model and draw posterior
samples.

Now we will define the system parmaeters that we will use to construct the phase
curvec and phase fold the light curve:

.. code-block:: python

    b = 0.4960  # Esteves 2015
    r_star = 1.991 * R_sun
    a_rs = 4.13 # Stassun 2017
    inc = np.degrees(np.arccos(b / a_rs))
    period = 2.204740  # Stassun 2017
    t0 = 2454954.357462  # Bonomo 2017
    rp_rs = float(16.9 * R_earth / r_star)  # Stassun 2017, Berger 2017
    a_rp = float(a_rs / rp_rs)
    T_s = 6449  # Berger 2018

    p = Planet(
        per=period,
        t0=t0,
        rp=rp_rs,
        a=a_rs,
        inc=inc,
        u=[0.0, 0.0],
        fp=1.0,
        t_secondary=t0 + period/2,
        T_s=T_s,
        rp_a=1/a_rp,
        w=90,
        ecc=0,
        name="HAT-P-7"
    )
    p.duration = 4.0398 / 24  # Holczer 2016
    rho_star = 0.27 * u.g / u.cm ** 3  # Stassun 2017
    eclipse_half_dur = p.duration / p.per / 2

Now we will use `~lightkurve.search.search_lightcurve` to download the long
cadence light curve from Quarter 10 of the Kepler observations of HAT-P-7 b.
We'll then "flatten" the light curve, which applies a savgol filter to the light
curve to do some long-term detrending on the out-of-transit portions of the
phase curve. We will also sigma-clip the fluxes to remove outliers:

.. code-block:: python

    lcf = search_lightcurve(
        p.name, mission="Kepler", cadence="long", quarter=10
    ).download_all()

    slc = lcf.stitch()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
            phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    slc = slc.flatten(
        polyorder=3, break_tolerance=10,
        window_length=1001, mask=~out_of_transit
    ).remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
            phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    sc = sigma_clip(
        slc.flux[out_of_transit],
        maxiters=100, sigma=8, stdfunc=mad_std
    )

Next we will compute the masked phases, times, and the normalized fluxes
:math:`F_p/F_\mathrm{star}` in units of ppm:

.. code-block:: python

    phase = phases[out_of_transit][~sc.mask]
    time = slc.time.jd[out_of_transit][~sc.mask]

    bin_in_eclipse = np.abs(phase - 0.5) < eclipse_half_dur
    unbinned_flux_mean = np.mean(sc[~sc.mask].data)

    unbinned_flux_mean_ppm = 1e6 * (unbinned_flux_mean - 1)
    flux_normed = (
        1e6 * (sc[~sc.mask].data / unbinned_flux_mean - 1.0)
    )
    flux_normed_err = (
        1e6 * slc.flux_err[out_of_transit][~sc.mask].value
    )

Now we will median-bin the phase folded Kepler light curve:

.. code-block:: python

    bins = 250
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


We compute the eclipse model once, it will not vary in the MCMC:


.. code-block:: python

    # Compute the eclipse model (no limb-darkening),
    # normalize the eclipse model to unity out of eclipse and
    # zero in-eclipse
    eclipse = TransitModel(
        p, binphase * period + t0,
        transittype='secondary',
    ).light_curve(p) - 1

Next we construct the numpyro model. This is a long block of code, so let's
jump straight into in-line comments:

.. code-block:: python

    def model():
        # Define reflected light phase curve model according to
        # Heng, Morris & Kitzmann (2021)
        omega = numpyro.sample(
            'omega', dist.Uniform(low=0, high=1)
        )
        g = numpyro.sample('g',
            dist.TwoSidedTruncatedDistribution(
                dist.Normal(loc=0, scale=0.4),
                low=0, high=1
            )
        )
        reflected_ppm, A_g, q = reflected_phase_curve(binphase, omega, g, a_rp)

        # Define the ellipsoidal variation parameterization (simple sinusoid)
        ellipsoidal_amp = numpyro.sample(
            'ellip_amp', dist.Uniform(
                low=0, high=50
            )
        )
        ellipsoidal_model_ppm = - ellipsoidal_amp * jnp.cos(
            4 * np.pi * (binphase - 0.5)
        ) + ellipsoidal_amp

        # Define the doppler variation parameterization (simple sinusoid)
        doppler_amp = numpyro.sample(
            'doppler_amp', dist.Uniform(low=0, high=50)
        )
        doppler_model_ppm = doppler_amp * jnp.sin(
            2 * np.pi * binphase
        )

        # Define the thermal emission model according to description in
        # Morris et al. (in prep)
        xi = 2 * np.pi * (binphase - 0.5)
        n_phi = 75
        n_theta = 5
        phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)
        theta2d, phi2d = np.meshgrid(theta, phi)

        ln_C_11_kepler = -2.6
        C_11_kepler = jnp.exp(ln_C_11_kepler)
        hml_eps = 0.72
        hml_f = (2 / 3 - hml_eps * 5 / 12) ** 0.25
        delta_phi = 0

        A_B = 0.0

        # Compute the thermal phase curve with zero phase offset
        thermal, T = thermal_phase_curve(
            xi, delta_phi, 4.5, 0.575, C_11_kepler, T_s, a_rs, 1 / a_rp, A_B,
            theta2d, phi2d, filt_wavelength, filt_trans, hml_f
        )

        # Define the composite phase curve model
        flux_norm = eclipse * (
            reflected_ppm + ellipsoidal_model_ppm +
            doppler_model_ppm + 1e6 * thermal
        )

        # Keep track of the geometric albedo and integral phase function at
        # each step in the chain
        numpyro.deterministic('A_g', A_g)
        numpyro.deterministic('q', q)
        numpyro.deterministic('light_curve', flux_norm)

        # Define the likelihood
        numpyro.sample(
            "obs", dist.Normal(
                loc=flux_norm,
                scale=binerror
            ), obs=binflux
        )


Now our model is set up, and we are ready to draw posterior samples from the
model given the data, which we will do with numpyro for the most efficient
posterior sampling of our degenerate phase curve parameterization. This will take
a few seconds:

.. code-block:: python

    # Random numbers in jax are generated like this:
    rng_seed = 42
    rng_keys = split(
        PRNGKey(rng_seed),
        cpu_cores
    )

    # Define a sampler, using here the No U-Turn Sampler (NUTS)
    # with a dense mass matrix:
    sampler = NUTS(
        model,
        dense_mass=True
    )

    # Monte Carlo sampling for a number of steps and parallel chains:
    mcmc = MCMC(
        sampler,
        num_warmup=500,
        num_samples=2_000,
        num_chains=cpu_cores
    )

    # Run the MCMC
    mcmc.run(rng_keys)

Let's finally plot the final results:

.. code-block:: python

    # arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
    result = arviz.from_numpyro(mcmc)

    corner(result, var_names=["~light_curve"])
    plt.show()

    plt.figure()
    plt.errorbar(
        binphase, binflux, binerror,
        fmt='.', color='k', ecolor='silver'
    )

    # plot a few samples of the model
    random_indices = np.random.randint(
        0, mcmc.num_samples, size=50
    )
    for i in random_indices:
        plt.plot(
            binphase,
            np.array(result.posterior['light_curve'][0, i]),
            color='DodgerBlue', alpha=0.2, zorder=10
        )


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

    import arviz
    from corner import corner
    import numpyro

    # Set the number of cores on your machine for parallelism:
    cpu_cores = 1
    numpyro.set_host_device_count(cpu_cores)

    from numpyro.infer import MCMC, NUTS
    from numpyro import distributions as dist

    from jax import numpy as jnp
    from jax.random import PRNGKey, split

    from lightkurve import search_lightcurve
    from batman import TransitModel

    import astropy.units as u
    from astropy.constants import R_sun, R_earth
    from astropy.stats import sigma_clip, mad_std

    from kelp import Filter, Planet
    from kelp.jax import reflected_phase_curve, thermal_phase_curve

    b = 0.4960  # Esteves 2015
    r_star = 1.991 * R_sun
    a_rs = 4.13 # Stassun 2017
    inc = np.degrees(np.arccos(b / a_rs))
    period = 2.204740  # Stassun 2017
    t0 = 2454954.357462  # Bonomo 2017
    rp_rs = float(16.9 * R_earth / r_star)  # Stassun 2017, Berger 2017
    a_rp = float(a_rs / rp_rs)
    T_s = 6449  # Berger 2018

    p = Planet(
        per=period,
        t0=t0,
        rp=rp_rs,
        a=a_rs,
        inc=inc,
        u=[0.0, 0.0],
        fp=1.0,
        t_secondary=t0 + period/2,
        T_s=T_s,
        rp_a=1/a_rp,
        w=90,
        ecc=0,
        name="HAT-P-7"
    )
    p.duration = 4.0398 / 24  # Holczer 2016
    rho_star = 0.27 * u.g / u.cm ** 3  # Stassun 2017
    eclipse_half_dur = p.duration / p.per / 2

    lcf = search_lightcurve(
        p.name, mission="Kepler", cadence="long", quarter=10
    ).download_all()

    slc = lcf.stitch()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
            phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    slc = slc.flatten(
        polyorder=3, break_tolerance=10,
        window_length=1001, mask=~out_of_transit
    ).remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
    in_eclipse = np.abs(phases - 0.5) < eclipse_half_dur
    in_transit = (phases < 1.5 * eclipse_half_dur) | (
            phases > 1 - 1.5 * eclipse_half_dur)
    out_of_transit = np.logical_not(in_transit)

    sc = np.array(slc.flux[out_of_transit])
    phase = phases[out_of_transit]
    time = slc.time.jd[out_of_transit]

    bin_in_eclipse = np.abs(phase - 0.5) < eclipse_half_dur
    unbinned_flux_mean = np.mean(sc.data)

    unbinned_flux_mean_ppm = 1e6 * (unbinned_flux_mean - 1)
    flux_normed = (
        1e6 * (sc.data / unbinned_flux_mean - 1.0)
    )
    flux_normed_err = (
        1e6 * slc.flux_err[out_of_transit].value
    )

    bins = 250
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
    filt.bin_down(6)  # This speeds up integration by orders of magnitude
    filt_wavelength, filt_trans = filt.wavelength.to(u.m).value, filt.transmittance

    # Compute the eclipse model (no limb-darkening),
    # normalize the eclipse model to unity out of eclipse and
    # zero in-eclipse
    eclipse = TransitModel(
        p, binphase * period + t0,
        transittype='secondary',
    ).light_curve(p) - 1


    def model():
        # Define reflected light phase curve model according to
        # Heng, Morris & Kitzmann (2021)
        omega = numpyro.sample(
            'omega', dist.Uniform(low=0, high=1)
        )
        g = numpyro.sample('g',
            dist.TwoSidedTruncatedDistribution(
                dist.Normal(loc=0, scale=0.4),
                low=0, high=1
            )
        )
        reflected_ppm, A_g, q = reflected_phase_curve(binphase, omega, g, a_rp)

        # Define the ellipsoidal variation parameterization (simple sinusoid)
        ellipsoidal_amp = numpyro.sample(
            'ellip_amp', dist.Uniform(
                low=0, high=50
            )
        )
        ellipsoidal_model_ppm = - ellipsoidal_amp * jnp.cos(
            4 * np.pi * (binphase - 0.5)
        ) + ellipsoidal_amp

        # Define the doppler variation parameterization (simple sinusoid)
        doppler_amp = numpyro.sample(
            'doppler_amp', dist.Uniform(low=0, high=50)
        )
        doppler_model_ppm = doppler_amp * jnp.sin(
            2 * np.pi * binphase
        )

        # Define the thermal emission model according to description in
        # Morris et al. (in prep)
        xi = 2 * np.pi * (binphase - 0.5)
        n_phi = 75
        n_theta = 5
        phi = np.linspace(-2 * np.pi, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)
        theta2d, phi2d = np.meshgrid(theta, phi)

        ln_C_11_kepler = -2.6
        C_11_kepler = jnp.exp(ln_C_11_kepler)
        hml_eps = 0.72
        hml_f = (2 / 3 - hml_eps * 5 / 12) ** 0.25
        delta_phi = 0

        A_B = 0.0

        # Compute the thermal phase curve with zero phase offset
        thermal, T = thermal_phase_curve(
            xi, delta_phi, 4.5, 0.575, C_11_kepler, T_s, a_rs, 1 / a_rp, A_B,
            theta2d, phi2d, filt_wavelength, filt_trans, hml_f
        )

        # Define the composite phase curve model
        flux_norm = eclipse * (
            reflected_ppm + ellipsoidal_model_ppm +
            doppler_model_ppm + 1e6 * thermal
        )

        # Keep track of the geometric albedo and integral phase function at
        # each step in the chain
        numpyro.deterministic('A_g', A_g)
        numpyro.deterministic('q', q)
        numpyro.deterministic('light_curve', flux_norm)

        # Define the likelihood
        numpyro.sample(
            "obs", dist.Normal(
                loc=flux_norm,
                scale=binerror
            ), obs=binflux
        )

    # Random numbers in jax are generated like this:
    rng_seed = 42
    rng_keys = split(
        PRNGKey(rng_seed),
        cpu_cores
    )

    # Define a sampler, using here the No U-Turn Sampler (NUTS)
    # with a dense mass matrix:
    sampler = NUTS(
        model,
        dense_mass=True
    )

    # Monte Carlo sampling for a number of steps and parallel chains:
    mcmc = MCMC(
        sampler,
        num_warmup=500,
        num_samples=2_000,
        num_chains=cpu_cores
    )

    # Run the MCMC
    mcmc.run(rng_keys)

    # arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
    result = arviz.from_numpyro(mcmc)

    corner(result, var_names=["~light_curve"])
    plt.show()

    plt.figure()
    plt.errorbar(
        binphase, binflux, binerror,
        fmt='.', color='k', ecolor='silver'
    )

    # plot a few samples of the model
    random_indices = np.random.randint(
        0, mcmc.num_samples, size=50
    )
    for i in random_indices:
        plt.plot(
            binphase,
            np.array(result.posterior['light_curve'][0, i]),
            color='DodgerBlue', alpha=0.2, zorder=10
        )


    plt.legend()
    plt.ylim([-30, 110])
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
