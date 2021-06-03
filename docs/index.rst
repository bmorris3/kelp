.. image:: _static/kelp.svg
  :align: right
  :width: 200

====
kelp
====

This is the documentation for ``kelp``, a photometric phase curve
package written in Python. The source code is available on
`GitHub <https://github.com/bmorris3/kelp>`_.

.. toctree::
  :maxdepth: 2

  kelp/installation.rst
  kelp/background.rst
  kelp/gettingstarted.rst
  kelp/interactive.rst
  kelp/optimization.rst
  kelp/reflected.rst
  kelp/albedo_redistribution.rst
  kelp/index.rst


Interactive demo
^^^^^^^^^^^^^^^^

The :math:`h_{m\ell}` basis is a mathematical shorthand for describing
physically-motivated temperature maps for exoplanets. Three parameters which
can be tuned to fit most phase curves include the the dimensionless drag
frequency :math:`\omega`, the power in the :math:`m=\ell=1` spherical harmonic
mode :math:`C_{11}`, and the arbitrary rotational offset of the coordinate
system :math:`\Delta\phi`. Tweak these parameters below to see how they affect
the corresponding maps and phase curves:

.. raw:: html
    :file: _kelp_vis.html