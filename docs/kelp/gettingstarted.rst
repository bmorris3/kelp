Getting Started
===============

Representing a planetary system with the :math:`h_{m\ell}` basis
----------------------------------------------------------------

First, we'll import the necessary packages:

.. jupyter-execute::

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np

    from kelp import Filter, Planet, Model

Next we set some parameters for the model:

.. jupyter-execute::

    # Set phase curve parameters
    hotspot_offset = 0  # hotspot offset
    alpha = 0.6         # alpha ~ 0.5
    ln_omega_drag = 0   # omega_drag ~ 1
    ln_c11 = -5         # Spherical harmonic power C_{m=1, l=1}
    ln_c13 = -3         # Spherical harmonic power C_{m=1, l=3}

    # Set observation parameters
    n = 'HD 189733'     # name of the planetary system
    channel = 1         # Spitzer IRAC channel of observations
    n_theta = 20        # number of latitudes to simulate
    n_phi = 25          # number of longitudes to simulate

We initialize a `~kelp.Planet` and `~kelp.Filter` object for the model:

.. jupyter-execute::

    # Import planet properties
    p = Planet.from_name(n)

    # Import IRAC filter properties
    filt = Filter.from_name(f"IRAC {channel}")
    filt.bin_down()  # this speeds up the example

We specify the :math:`C_{m\ell}` terms like so:

.. jupyter-execute::

    # These elements will be accessed like C_ml[m][l]:
    C_ml = [[0],
            [0, np.exp(ln_c11), 0],
            [0, 0, 0, 0, 0],
            [0, -np.exp(ln_c13), 0, 0, 0, 0, 0, 0]]

Let's construct a `~kelp.Model` object,

.. jupyter-execute::

    # Generate an h_ml basis model representation of the system:
    model = Model(hotspot_offset=hotspot_offset,
                  omega_drag=np.exp(ln_omega_drag),
                  alpha=alpha, C_ml=C_ml, lmax=3, A_B=0,
                  planet=p, filt=filt)

and plot the temperature map using `~kelp.Model.temperature_map`:

.. jupyter-execute::

    # Compute the temperature map:
    T, theta, phi = model.temperature_map(n_theta, n_phi, f=2**-0.5)

    # Plot the temperature map
    cax = plt.pcolormesh(phi / np.pi, theta / np.pi, T)
    plt.colorbar(cax, label='T [K]')
    plt.xlabel('$\\phi/\\pi$')
    plt.ylabel('$\\theta/\\pi$')
    plt.show()

and plot the phase curve that results from this temperature map using
`~kelp.Model.phase_curve`:

.. jupyter-execute::

    # Compute the phase curve:
    xi = np.linspace(-2.7, 2.7, 50)
    phase_curve = model.phase_curve(xi)

    # Plot the phase curve
    plt.plot(xi / np.pi, phase_curve)
    plt.xlabel('$\\xi/\\pi$')
    plt.ylabel('$F_p/F_s$')
    plt.show()

Spherical harmonics components
------------------------------

In this example we'll plot the contributions from each of the spherical harmonic
perturbations to the temperature field. First, let's import the necessary
packages:

.. jupyter-execute::

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.units as u

    from kelp import Model, Filter, Planet

Next, let's set up an instance of the `~kelp.Planet` and `~kelp.Filter`, and
a grid of :math:`\theta` and :math:`\phi` on which to plot the temperature
field:

.. jupyter-execute::

    p = Planet.from_name('HD 189733')
    filt = Filter.from_name('IRAC 1')

    hotspot_offset = 0
    alpha = 0.6
    ln_omega_drag = 6
    f = 2 ** -0.5
    lmax = 3

    n_phi = 50
    n_theta = 30

next we'll write a few helper functions that will generate pretty temperature
field plots:

.. jupyter-execute::

    def indexer(m, l):
        C_ml = [[0],
                [0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
        C_ml[m][l] = 1
        return C_ml

    def generate_temp_map(m, l, A_B=0):
        C_ml = indexer(m, l)
        model = Model(hotspot_offset, alpha, np.exp(ln_omega_drag), A_B,
                      C_ml, lmax, planet=p, filt=filt)

        T, _, _ = model.temperature_map(n_theta, n_phi, f)
        return T

and we'll build the plot:

.. jupyter-execute::

    example = indexer(1, 0)
    fig, ax = plt.subplots(len(example), len(example[-1]), figsize=(20, 10))

    for m in range(0, lmax + 1):
        for l in range(-m, m + 1):
            temperature = generate_temp_map(m, l)

            cax = ax[m, l + len(example[-1])//2].imshow(temperature)
            ax[m, l + len(example[-1])//2].set_title(f'$m = {m},\,\ell = {l}$')

    for i in range(len(example)):
        for j in range(len(example[-1])):
            ax[i, j].axis('off')

    plt.tight_layout()
    plt.show()
