Equations
=========

Adirondax solves the following equations:


Magnetohydrodynamics
--------------------

TODO

Schrodinger-Poisson
-------------------

TODO

Cylindrical geometry
--------------------

Setting ``mesh.geometry`` to ``'cylindrical'`` solves the Euler equations on an
axisymmetric :math:`(R,z)` mesh, with :math:`\partial/\partial\phi = 0`. The
domain is :math:`R \in [0, L_R]`, :math:`z \in [0, L_z]`, and the ``'axis'``
boundary condition places the symmetry axis at :math:`R=0` and a reflecting
wall at :math:`R = L_R`.

The equations are

.. math::

   \frac{\partial \rho}{\partial t}
     + \frac{1}{R}\frac{\partial}{\partial R}\left(R \rho v_R\right)
     + \frac{\partial}{\partial z}\left(\rho v_z\right) = 0

.. math::

   \frac{\partial (\rho v_R)}{\partial t}
     + \frac{1}{R}\frac{\partial}{\partial R}\left(R \left[\rho v_R^2 + P\right]\right)
     + \frac{\partial}{\partial z}\left(\rho v_R v_z\right)
     = \frac{P}{R} + \frac{\rho v_\phi^2}{R}

.. math::

   \frac{\partial (\rho v_z)}{\partial t}
     + \frac{1}{R}\frac{\partial}{\partial R}\left(R \rho v_R v_z\right)
     + \frac{\partial}{\partial z}\left(\rho v_z^2 + P\right) = 0

.. math::

   \frac{\partial E}{\partial t}
     + \frac{1}{R}\frac{\partial}{\partial R}\left(R \left[E + P\right] v_R\right)
     + \frac{\partial}{\partial z}\left(\left[E + P\right] v_z\right) = 0

with :math:`E = P/(\gamma-1) + \tfrac{1}{2}\rho\left(v_R^2 + v_z^2 + v_\phi^2\right)`.

Discretization
^^^^^^^^^^^^^^

The solver is a finite-volume scheme written in terms of cell volumes and face
areas, so the same routines cover both geometries. For a cell spanning
:math:`[R_{i-1/2}, R_{i+1/2}]`,

.. math::

   V_i = \pi\left(R_{i+1/2}^2 - R_{i-1/2}^2\right)\Delta z, \quad
   A^R_{i\pm 1/2} = 2\pi R_{i\pm 1/2}\,\Delta z, \quad
   A^z_i = \pi\left(R_{i+1/2}^2 - R_{i-1/2}^2\right)

Cartesian geometry is the special case in which these are the constants
:math:`\Delta x \Delta y`, :math:`\Delta y` and :math:`\Delta x`.

The radial pressure term is the only part of the system not in divergence form.
It is discretized as a difference of face areas rather than as :math:`P/R`,

.. math::

   S_{\rho v_R} = P_i \, \frac{A^R_{i+1/2} - A^R_{i-1/2}}{V_i}

which reduces to :math:`P/R` in the continuum limit but has the stronger
discrete property of cancelling the pressure part of the flux divergence
exactly for a uniform state. Hydrostatic equilibrium is therefore held to
machine precision, including in the cell touching the axis, where
:math:`A^R_{-1/2} = 0`. That vanishing inner face area is also what prevents
any flux from crossing the axis.

Terms that genuinely require a :math:`1/R` use the volume-centroid radius

.. math::

   \bar{R}_i = \frac{2}{3}\,
     \frac{R_{i+1/2}^3 - R_{i-1/2}^3}{R_{i+1/2}^2 - R_{i-1/2}^2}

which equals :math:`\tfrac{2}{3}\Delta R` in the cell on the axis and so is
never zero.

Because the azimuthal direction is not resolved, the timestep restriction is
the usual :math:`\Delta t \le C\,\min(\Delta R, \Delta z)/(c_s + |v|)`; there is
no :math:`1/R` narrowing of cells near the axis.

Rotation
^^^^^^^^

Switching on ``physics.rotation`` evolves the azimuthal velocity
:math:`v_\phi` as well. It is carried as the angular momentum density
:math:`\rho R v_\phi`, which obeys a source-free conservation law,

.. math::

   \frac{\partial (\rho R v_\phi)}{\partial t}
     + \frac{1}{R}\frac{\partial}{\partial R}\left(R \cdot R v_\phi \rho v_R\right)
     + \frac{\partial}{\partial z}\left(R v_\phi \rho v_z\right) = 0

so angular momentum is conserved to round-off rather than only to truncation
error. The azimuthal velocity feeds back on the radial momentum through the
centrifugal term :math:`\rho v_\phi^2 / R` and contributes
:math:`\tfrac{1}{2}\rho v_\phi^2` to the total energy.

Self-gravity, magnetic fields and the Schrodinger field are not yet available
in cylindrical geometry.
