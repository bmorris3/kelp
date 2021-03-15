if __name__ == '__main__':
    import geoviews as gv
    import xarray as xr
    from kelp import Model, Planet, Filter
    import numpy as np
    gv.extension('bokeh')

    n_phi = 30
    n_theta = 15
    omega_range = np.linspace(0.5, 5.5, 10)
    c11_range = np.linspace(0.1, 0.3, 10)
    temperature_maps = np.zeros(
        (n_phi, n_theta, len(omega_range), len(c11_range)), dtype=int
    )
    p = Planet.from_name('KELT-9')
    filt = Filter.from_name("CHEOPS")
    for i, omega in enumerate(omega_range):
        for j, c11 in enumerate(c11_range):
            m = Model(0, 0.575, omega, 0, [[0], [0, c11, 0]],
                      1, planet=p, filt=filt)
            T, theta, phi = m.temperature_map(n_theta, n_phi)
            temperature_maps[..., i, j] = T.T
    condition = (phi <= 2*np.pi) & (phi >= 0)

    arr = xr.DataArray(
        temperature_maps[condition, ...],
        coords=[np.degrees(phi)[condition], np.degrees(theta) - 90,
                omega_range, c11_range],
        dims=['longitude', 'latitude', 'omega', 'C_11'],
        name='surface_temperature'
    )

    dataset = gv.Dataset(arr, ['longitude', 'latitude', 'omega', 'C_11'],
                         'surface_temperature')
    images = dataset.to(gv.Image, ['longitude', 'latitude'],
                        'surface_temperature', ['omega', 'C_11'])
    images.opts(cmap='viridis', colorbar=True, width=400, height=400)
    gv.save(images, 'kelp_geoviews', fmt='widgets')