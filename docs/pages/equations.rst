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

Radiation transport
-------------------

Switching on ``physics.radiation`` evolves the frequency-integrated specific
intensities :math:`I_f(\mathbf{n})` of a set of frequency groups along a set of
discrete directions :math:`\mathbf{n}`, is based on the implicit scheme of
Jiang (2021, ApJS 253, 49) and its multigroup extension (Jiang 2022,
arXiv:2209.06240). In the lab frame,

.. math::

   \frac{\partial I_f}{\partial t} + c\,\mathbf{n}\cdot\nabla I_f
     = c\,\Gamma^{-3}\left[\sigma_{t,f}\left(J_{0,f} - I_{0,f}\right)
     + \sigma_{p,f}\left(\epsilon_f(T) - J_{0,f}\right)\right]

where :math:`I_{0,f} = \Gamma^4 I_f` is the intensity in the fluid rest frame,
:math:`\Gamma = \gamma(1 - \mathbf{n}\cdot\mathbf{v}/c)`, :math:`J_{0,f}` is its
angle average in that frame, :math:`\sigma_{t,f} = \rho(\kappa_s + \kappa_{a,f})`
and :math:`\sigma_{p,f} = \rho\kappa_{p,f}` are the total (Rosseland mean) and
Planck mean opacities of the group, and :math:`\epsilon_f(T)` is the blackbody
emission integrated over the group, which sums to :math:`a_r T^4/4\pi`. The
intensities are in units of energy density per steradian, so the radiation
energy density is :math:`E_r = 4\pi\sum_n w_n I_n` for quadrature weights
:math:`w_n` that sum to one. The gas temperature is
:math:`T = P/(\rho\,\mathcal{R})` with ``radiation.gas_constant``
:math:`\mathcal{R} = k_B/(\mu m_p)`.

The transport and the coupling are solved together with backward Euler, so the
timestep is set by the hydrodynamics alone. The face fluxes are a modified HLLE
flux that is upwind in optically thin cells and centered, with a small
diffusion, in optically thick ones, which recovers the diffusion limit. Unlike
Jiang (2021), the optical depth of a face is taken from the more transparent of
its two cells, which keeps the flux upwind, and the intensities positive, at the
surface of an optically thick body. The
energy and momentum the radiation loses to the coupling are given to the gas,
so the total energy and momentum of both are conserved to round-off. The
radiation is advanced for half a timestep before and after the hydrodynamics.
The gas density and velocity are held fixed during the radiation solve, which
assumes the inertia of the radiation is small next to that of the gas,
:math:`E_r \ll \rho c^2`, as it is in any nonrelativistic flow.

The implicit equations of each cell (all its groups and directions, and its
temperature) are solved directly, which reduces to one scalar equation for the
temperature. This local solve preconditions a Newton-Krylov (GMRES) iteration
over the whole mesh (``radiation.solver = 'gmres'``), or is iterated on its
own as in Jiang (2021) (``'jacobi'``). The solve is differentiated implicitly,
so gradients through it cost one more linear solve, and do not depend on how
many iterations it took.

Directions
^^^^^^^^^^

The directions are :math:`\mathbf{n} = (\sin\zeta\cos\psi, \cos\zeta,
\sin\zeta\sin\psi)` in the :math:`(x, y, \perp)` basis of the mesh: a
Gauss-Legendre grid of ``num_polar`` values of :math:`\cos\zeta` times
``num_azimuthal`` values of :math:`\psi` in :math:`(0, \pi)`. Both 2D
geometries are mirror symmetric through the mesh plane, so the directions with
:math:`\psi \in (\pi, 2\pi)` are their mirror images and need not be evolved.
The quadrature integrates the moments up to second order exactly.

In cylindrical geometry the basis is the local :math:`(R, z, \phi)` basis of
each cell, which turns as a ray crosses the mesh. In conservative form

.. math::

   \mathbf{n}\cdot\nabla I
     = \frac{1}{R}\frac{\partial}{\partial R}\left(R\sin\zeta\cos\psi\, I\right)
     + \frac{\partial}{\partial z}\left(\cos\zeta\, I\right)
     - \frac{\sin\zeta}{R}\frac{\partial}{\partial\psi}\left(\sin\psi\, I\right)

The last term moves photons towards :math:`\psi = 0`, the radially outward
direction. It is upwinded in :math:`\psi`, with :math:`1/R` taken as
:math:`(A^R_{i+1/2} - A^R_{i-1/2})/V_i` and :math:`\sin\psi` on the bin faces
as the running sum of :math:`\cos\psi\,\Delta\psi`, so that it conserves the
radiation energy exactly and balances the radial flux of an isotropic field to
round-off.

Not yet included are the Doppler shift of photons between frequency groups
(exact with one group) and Compton scattering.
