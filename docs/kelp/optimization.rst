************
Optimization
************

First let's import the necessary packages:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    from kelp import Model, Planet, Filter

    from scipy.optimize import minimize

    from emcee import EnsembleSampler
    from multiprocessing import Pool
    from corner import corner

    np.random.seed(42)

Next let's set up the properties of the `~kelp.Planet`, `~kelp.Filter`, and
the `~kelp.Model`:

.. code-block:: python

    planet = Planet.from_name('HD 189733')
    filt = Filter.from_name("IRAC 1")
    filt.bin_down(10)

    xi = np.linspace(-np.pi, np.pi, 50)

    hotspot_offset = np.radians(40)
    alpha = 0.6
    omega = 4.5
    A_B = 0
    lmax = 1
    C = [[0],
         [0, 0.6, 0]]
    obs_err = 1e-8

Now we'll initialize a noisy instance of the model to be our "truth" in
this example, which we will recover with optimization techniques:

.. code-block:: python

    model = Model(hotspot_offset, alpha, omega,
                  A_B, C, lmax, planet=planet, filt=filt)
    obs = model.phase_curve(xi).flux + obs_err * np.random.randn(xi.shape[0])

    errkwargs = dict(color='k', fmt='.', ecolor='silver')
    plt.errorbar(xi / np.pi, obs, obs_err, **errkwargs)
    plt.xlabel('$\\xi/\\pi$')
    plt.ylabel('$\\rm F_p/F_s$')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    from kelp import Model, Planet, Filter

    np.random.seed(42)

    planet = Planet.from_name('HD 189733')
    filt = Filter.from_name("IRAC 1")
    filt.bin_down(10)

    xi = np.linspace(-np.pi, np.pi, 50)

    hotspot_offset = np.radians(40)
    alpha = 0.6
    omega = 4.5
    A_B = 0
    lmax = 1
    C = [[0],
         [0, 0.6, 0]]
    obs_err = 1e-8
    model = Model(hotspot_offset, alpha, omega,
                  A_B, C, lmax, planet=planet, filt=filt)
    obs = model.phase_curve(xi).flux + obs_err * np.random.randn(xi.shape[0])

    errkwargs = dict(color='k', fmt='.', ecolor='silver')
    plt.errorbar(xi / np.pi, obs, obs_err, **errkwargs)
    plt.xlabel('$\\xi/\\pi$')
    plt.ylabel('$\\rm F_p/F_s$')
    plt.show()

The simulated observations ``obs`` have small scale, uncorrelated scatter
about the phase curve mean model. Next we'll use ``scipy`` to find the best-fit
values for the hotspot offset :math:`\Delta \phi` and the power in the
:math:`C_{11}` spherical harmonic coefficient term:

.. code-block:: python

    def pc_model(p, x):
        """
        Phase curve model
        """
        offset, c_11 = p
        C = [[0],
             [0, c_11, 0]]
        model = Model(hotspot_offset=offset, alpha=alpha,
                      omega_drag=omega, A_B=A_B, C_ml=C, lmax=1,
                      planet=planet, filt=filt)
        return model.phase_curve(x).flux

    def lnprior(p):
        """
        Log prior
        """
        offset, c_11 = p

        if (offset > np.pi or offset < -np.pi or
            c_11 > 1 or c_11 < 0):
            return -np.inf

        return 0

    def lnlike(p, x, y, yerr):
        """
        Log-likelihood
        """
        return -0.5 * np.sum((pc_model(p, x) - y)**2 / yerr**2)

    def lnprob(p, x, y, yerr):
        """
        Log probability: sum of lnlike and lnprior
        """
        lp = lnprior(p)

        if np.isfinite(lp):
            return lp + lnlike(p, x, y, yerr)
        return -np.inf


    initp = np.array([0.7, 0.6])

    bounds = [[0, 2], [0.1, 1]]

    soln = minimize(lambda *args: -lnprob(*args),
                    initp, args=(xi, obs, obs_err),
                    method='powell')


``soln.x`` now contains the best-fit (:math:`\Delta \phi`, :math:`C_{11}`)
parameters from Powell's method. With this guess at the maximum a posteriori
values for the free parameters, let's now use Markov Chain Monte Carlo to
measure the uncertainty on the maximum-likelihood parameters:

.. code-block:: python

    ndim = 2
    nwalkers = 2 * ndim

    p0 = [soln.x + 0.1 * np.random.randn(ndim)
          for i in range(nwalkers)]

    with Pool() as pool:
        sampler = EnsembleSampler(nwalkers, ndim, lnprob,
                                  args=(xi, obs, obs_err),
                                  pool=pool)
        p1 = sampler.run_mcmc(p0, 100)
        sampler.reset()
        sampler.run_mcmc(p1, 500)

    corner(sampler.flatchain, truths=[hotspot_offset, C[1][1]],
           labels=['$\Delta \phi$', '$C_{11}$']))
    plt.show()

    p_map = sampler.flatchain[np.argmax(sampler.flatlnprobability)]

    plt.errorbar(xi/np.pi, obs, obs_err, **errkwargs)
    plt.plot(xi/np.pi, pc_model(p_map, xi), color='r')
    plt.xlabel('$\\xi/\\pi$')
    plt.ylabel('$\\rm F_p/F_s$')
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    from kelp import Model, Planet, Filter

    np.random.seed(42)

    planet = Planet.from_name('HD 189733')
    filt = Filter.from_name("IRAC 1")
    filt.bin_down(10)

    xi = np.linspace(-np.pi, np.pi, 50)

    hotspot_offset = np.radians(40)
    alpha = 0.6
    omega = 4.5
    A_B = 0
    lmax = 1
    C = [[0],
         [0, 0.6, 0]]
    obs_err = 1e-8
    model = Model(hotspot_offset, alpha, omega,
                  A_B, C, lmax, planet=planet, filt=filt)
    obs = model.phase_curve(xi).flux + obs_err * np.random.randn(xi.shape[0])

    def pc_model(p, x):
        """
        Phase curve model
        """
        offset, c_11 = p
        C = [[0],
             [0, c_11, 0]]
        model = Model(hotspot_offset=offset, alpha=alpha,
                      omega_drag=omega, A_B=A_B, C_ml=C, lmax=1,
                      planet=planet, filt=filt)
        return model.phase_curve(x).flux

    def lnprior(p):
        """
        Log prior
        """
        offset, c_11 = p

        if (offset > np.pi or offset < -np.pi or
            c_11 > 1 or c_11 < 0):
            return -np.inf

        return 0

    def lnlike(p, x, y, yerr):
        """
        Log-likelihood
        """
        return -0.5 * np.sum((pc_model(p, x) - y)**2 / yerr**2)

    def lnprob(p, x, y, yerr):
        """
        Log probability: sum of lnlike and lnprior
        """
        lp = lnprior(p)

        if np.isfinite(lp):
            return lp + lnlike(p, x, y, yerr)
        return -np.inf


    from scipy.optimize import minimize

    initp = np.array([0.7, 0.6])

    bounds = [[0, 2], [0.1, 1]]

    soln = minimize(lambda *args: -lnprob(*args),
                    initp, args=(xi, obs, obs_err),
                    method='powell')

    from emcee import EnsembleSampler
    from multiprocessing import Pool
    from corner import corner

    ndim = 2
    nwalkers = 2 * ndim

    p0 = [soln.x + 0.1 * np.random.randn(ndim)
          for i in range(nwalkers)]


    sampler = EnsembleSampler(nwalkers, ndim, lnprob,
                              args=(xi, obs, obs_err))
    p1 = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(p1, 500)

    corner(sampler.flatchain, truths=[hotspot_offset, C[1][1]],
           labels=['$\Delta \phi$', '$C_{11}$'])
    plt.show()

    p_map = sampler.flatchain[np.argmax(sampler.flatlnprobability)]

    errkwargs = dict(color='k', fmt='.', ecolor='silver')
    plt.errorbar(xi/np.pi, obs, obs_err, **errkwargs)
    plt.plot(xi/np.pi, pc_model(p_map, xi), color='r')
    plt.xlabel('$\\xi/\\pi$')
    plt.ylabel('$\\rm F_p/F_s$')
    plt.show()

The blue lines on the corner plot represent the "true" values which we used to
construct the simulated observations. The recovered (:math:`\Delta \phi`,
:math:`C_{11}`) parameters are consistent with their true values. The maximum
likelihood parameters generate a model (red) that looks very consistent with the
observations (black). Good job, team!
