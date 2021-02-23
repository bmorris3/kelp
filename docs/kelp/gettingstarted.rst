Getting Started
===============

Representing a planetary system with the :math:`h_{m\ell}` basis
----------------------------------------------------------------

First, we'll import the necessary packages:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    from kelp import Filter, Planet, Model

Next we set some parameters for the model:

.. code-block:: python

    # Set phase curve parameters
    hotspot_offset = 0  # hotspot offset
    alpha = 0.6
    omega_drag = 4.5
    ln_c11 = -1         # Spherical harmonic power C_{m=1, l=1}

    # Set observation parameters
    n = 'HD 189733'     # name of the planetary system
    channel = 1         # Spitzer IRAC channel of observations
    n_theta = 100        # number of latitudes to simulate
    n_phi = 50          # number of longitudes to simulate

We initialize a `~kelp.Planet` and `~kelp.Filter` object for the model:

.. code-block:: python

    # Import planet properties
    p = Planet.from_name(n)

    # Import IRAC filter properties
    filt = Filter.from_name(f"IRAC {channel}")
    filt.bin_down()  # this speeds up the example

We specify the :math:`C_{m\ell}` terms like so:

.. code-block:: python

    # These elements will be accessed like C_ml[m][l]:
    C_ml = [[0],
            [0, np.exp(ln_c11), 0]]

Let's construct a `~kelp.Model` object,

.. code-block:: python

    # Generate an h_ml basis model representation of the system:
    model = Model(hotspot_offset=hotspot_offset,
                  omega_drag=omega_drag,
                  alpha=alpha, C_ml=C_ml, lmax=1, A_B=0,
                  planet=p, filt=filt)

and plot the temperature map using `~kelp.Model.temperature_map`:

.. code-block:: python

    # Compute the temperature map:
    T, theta, phi = model.temperature_map(n_theta, n_phi, f=2**-0.5)

    # Plot the temperature map
    cax = plt.pcolormesh(phi / np.pi, theta / np.pi, T)
    plt.colorbar(cax, label='T [K]')
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    from kelp import Filter, Planet, Model

    # Set phase curve parameters
    hotspot_offset = 0  # hotspot offset
    alpha = 0.6
    omega_drag = 4.5
    ln_c11 = -1         # Spherical harmonic power C_{m=1, l=1}

    # Set observation parameters
    n = 'HD 189733'     # name of the planetary system
    channel = 1         # Spitzer IRAC channel of observations
    n_theta = 100       # number of latitudes to simulate
    n_phi = 50          # number of longitudes to simulate

    # Import planet properties
    p = Planet.from_name(n)

    # Import IRAC filter properties
    filt = Filter.from_name(f"IRAC {channel}")
    filt.bin_down()  # this speeds up the example

    # These elements will be accessed like C_ml[m][l]:
    C_ml = [[0],
            [0, np.exp(ln_c11), 0]]

    # Generate an h_ml basis model representation of the system:
    model = Model(hotspot_offset=hotspot_offset,
                  omega_drag=omega_drag,
                  alpha=alpha, C_ml=C_ml, lmax=1, A_B=0,
                  planet=p, filt=filt)

    # Compute the temperature map:
    T, theta, phi = model.temperature_map(n_theta, n_phi, f=2**-0.5)

    # Plot the temperature map
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    cax = ax.pcolormesh(phi, theta - np.pi/2, T)
    plt.colorbar(cax, label='T [K]', ax=ax)
    ax.set(xlabel='$\\phi/\\pi$', ylabel='$\\theta/\\pi$')
    plt.show()

and plot the phase curve that results from this temperature map using
`~kelp.Model.phase_curve`:

.. code-block:: python

    # Compute the phase curve:
    xi = np.linspace(-np.pi, np.pi, 50)
    phase_curve = model.phase_curve(xi)

    # Plot the phase curve
    phase_curve.plot()
    plt.xlabel('$\\xi/\\pi$')
    plt.ylabel('$F_p/F_s$')
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    from kelp import Filter, Planet, Model

    # Set phase curve parameters
    hotspot_offset = 0  # hotspot offset
    alpha = 0.6
    omega_drag = 4.5
    ln_c11 = -1         # Spherical harmonic power C_{m=1, l=1}

    # Set observation parameters
    n = 'HD 189733'     # name of the planetary system
    channel = 1         # Spitzer IRAC channel of observations

    # Import planet properties
    p = Planet.from_name(n)

    # Import IRAC filter properties
    filt = Filter.from_name(f"IRAC {channel}")
    filt.bin_down()  # this speeds up the example

    # These elements will be accessed like C_ml[m][l]:
    C_ml = [[0],
            [0, np.exp(ln_c11), 0]]

    # Generate an h_ml basis model representation of the system:
    model = Model(hotspot_offset=hotspot_offset,
                  omega_drag=omega_drag,
                  alpha=alpha, C_ml=C_ml, lmax=1, A_B=0,
                  planet=p, filt=filt)

    # Compute the phase curve:
    xi = np.linspace(-np.pi, np.pi, 50)
    phase_curve = model.phase_curve(xi)

    # Plot the phase curve
    phase_curve.plot()
    plt.xlabel('$\\xi/\\pi$')
    plt.ylabel('$\\rm F_p/F_s$')
    plt.show()
