if __name__ == '__main__':
    import geoviews as gv
    import xarray as xr
    from kelp import Model, Planet, Filter
    import numpy as np
    import holoviews as hv
    import param
    import panel as pn

    gv.extension('bokeh')

    n_phi = 30
    n_theta = 15
    xi = np.linspace(-np.pi, np.pi, 25)
    omega_range = np.linspace(0.5, 5.5, 5)
    c11_range = np.linspace(0.1, 0.3, 5)
    delta_phi_range = np.linspace(-np.pi, 0, 5)

    temperature_maps = np.zeros(
        (n_phi, n_theta, len(omega_range), len(c11_range), len(delta_phi_range)),
        dtype=int)
    p = Planet.from_name('KELT-9')
    filt = Filter.from_name("CHEOPS")
    phase_curves = np.zeros(
        (len(xi), len(omega_range), len(c11_range), len(delta_phi_range)))
    for i, omega in enumerate(omega_range):
        for j, c11 in enumerate(c11_range):
            for k, delta_phi in enumerate(delta_phi_range):
                m = Model(delta_phi, 0.575, omega, 0, [[0], [0, c11, 0]], 1,
                          planet=p, filt=filt)
                T, theta, phi = m.temperature_map(n_theta, n_phi)
                temperature_maps[..., i, j, k] = T.T
                phase_curves[:, i, j, k] = m.thermal_phase_curve(xi).flux

    condition = (phi <= 2 * np.pi) & (phi >= 0)

    arr_temp = xr.DataArray(
        temperature_maps[condition, ...],
        coords=[np.degrees(phi)[condition], np.degrees(theta) - 90, omega_range,
                c11_range, delta_phi_range],
        dims=['longitude', 'latitude', 'omega', 'C_11', 'delta_phi'],
        name='surface_temperature'
    )

    arr_pc = xr.DataArray(
        phase_curves,
        coords=[xi, omega_range, c11_range, delta_phi_range],
        dims=['xi', 'omega', 'C_11', 'delta_phi'],
        name='phase_curve'
    )


    class InteractiveKelpMap(param.Parameterized):
        omega = param.ObjectSelector(default=omega_range[-1], objects=omega_range)

        c11 = param.ObjectSelector(default=c11_range[0], objects=c11_range)

        delta_phi = param.ObjectSelector(default=delta_phi_range[-1],
                                         objects=delta_phi_range)

        @param.depends('omega', 'c11', 'delta_phi')
        def load_symbol(self):
            sample = arr_pc.where(
                (arr_temp.omega == self.omega) & (arr_temp.C_11 == self.c11) & (
                            arr_temp.delta_phi == self.delta_phi),
                drop=True).values.ravel()
            return hv.Curve(np.vstack([xi, sample]).T, 'xi', 'Fp/Fstar').opts(
                framewise=True, width=400, height=250)

        def view_pc(self):
            stocks = hv.DynamicMap(self.load_symbol)
            return stocks

        @param.depends('omega', 'c11', 'delta_phi')
        def view_map(self):
            sample_map = arr_temp.where(
                (arr_temp.omega == self.omega) & (arr_temp.C_11 == self.c11) & (
                            arr_temp.delta_phi == self.delta_phi), drop=True)
            dataset = gv.Dataset(sample_map,
                                 ['longitude', 'latitude', 'omega', 'C_11'],
                                 'surface_temperature')
            images = dataset.to(gv.Image, ['longitude', 'latitude'],
                                'surface_temperature')
            images.opts(cmap='viridis', colorbar=True, width=400, height=250)
            return images


    explorer = InteractiveKelpMap()

    stock_dmap = hv.DynamicMap(explorer.load_symbol)

    row = pn.Column(pn.panel(explorer.param,
                             parameters=['omega', 'c11', 'delta_phi']),
                    explorer.view_map, explorer.view_pc)

    row.save('_kelp_vis.html', embed=True)
