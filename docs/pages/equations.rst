Equations
=========

Adirondax couples a set of physics modules, each switched on in the
``physics`` parameters:

* ``hydro``: the Euler equations of an ideal gas,
* ``magnetic``: ideal magnetohydrodynamics (with ``hydro``),
* ``rotation``: the out-of-plane velocity (and field), for 2.5D problems,
* ``gravity`` and ``external_potential``: self-gravity and a fixed potential,
* ``quantum``: the Schrodinger-Poisson equations of a scalar field (fuzzy dark
  matter),
* ``radiation``: radiation transport, by discrete ordinates or flux-limited
  diffusion, coupled to the gas (or to a static material, without ``hydro``).

All of them work on a uniform 2D mesh, in Cartesian :math:`(x, y)` or
axisymmetric cylindrical :math:`(R, z)` geometry, and are written in pure JAX,
so every solver can be differentiated end to end.


Units
-----

The ``units`` parameters set the size of the code units of length, mass, time
and temperature, in cm, g, s and K; the default is cgs. Every physical constant
the solvers need (the speed of light, the gravitational constant, the Boltzmann
constant, the proton mass, the radiation constant, ...) is converted from cgs
into these units, so all of the physics modules share one consistent set. For
example, SI units are

.. code-block:: python

   "units": {"length": 1.0e2, "mass": 1.0e3, "time": 1.0, "temperature": 1.0}

Problems posed in terms of dimensionless numbers can instead set constants
directly, in code units, in the ``constants`` parameters, which take precedence
over the units. For instance, the ratio of the speed of light to the sound
speed, with :math:`k_B = m_p = 1` so that :math:`T = P/\rho`:

.. code-block:: python

   "constants": {"speed_of_light": 1.9e5, "boltzmann_constant": 1.0, "proton_mass": 1.0}

The constants in code units are available as ``sim.constants``.

Likewise, the Schrodinger field uses :math:`m/\hbar` in code units, derived
from the axion mass ``quantum.m_eV`` (:math:`10^{-22}` eV by default) and the
units.

The magnetic field is carried in units in which :math:`\mu_0 = 1`, so that
the magnetic pressure is :math:`B^2/2`.


Time integration
----------------

Each timestep of size :math:`\Delta t` is split symmetrically into the
operators of the active modules:

.. math::

   \mathcal{R}\left(\tfrac{\Delta t}{2}\right)\,
   \mathcal{K}\left(\tfrac{\Delta t}{2}\right)\,
   \mathcal{D}\left(\Delta t\right)\,
   \mathcal{K}\left(\tfrac{\Delta t}{2}\right)\,
   \mathcal{R}\left(\tfrac{\Delta t}{2}\right)

where the drift :math:`\mathcal{D}` advances the (magneto)hydrodynamics and the
kinetic term of the Schrodinger equation, the kick :math:`\mathcal{K}` applies
the gravitational potential, and :math:`\mathcal{R}` is the implicit radiation
update.

With ``time.num_timesteps`` set, the timesteps are equal. Otherwise the
timestep adapts to the state, as the smallest of the limits of the active
modules: the Courant condition of the (magneto)hydrodynamics,

.. math::

   \Delta t = C\,\min_{\rm cells}
     \frac{\min(\Delta x, \Delta y)}{|\mathbf{v}| + c_f}

with the Courant number :math:`C` = ``hydro.cfl`` and :math:`c_f` the sound
speed (or, with a magnetic field, the bound :math:`\sqrt{c_s^2 + v_A^2}` on the
fast magnetosonic speed), and :math:`\Delta t \le (m/\hbar)\,
\min(\Delta x, \Delta y)^2 / 6` for the Schrodinger field. The radiation is
implicit, and does not limit the timestep.


Hydrodynamics
-------------

Switching on ``physics.hydro`` solves the Euler equations of an ideal gas,

.. math::

   \frac{\partial \rho}{\partial t} + \nabla\cdot\left(\rho\mathbf{v}\right) = 0

.. math::

   \frac{\partial (\rho\mathbf{v})}{\partial t}
     + \nabla\cdot\left(\rho\mathbf{v}\mathbf{v} + P\,\mathbf{I}\right)
     = -\rho\nabla\Phi

.. math::

   \frac{\partial E}{\partial t}
     + \nabla\cdot\left[\left(E + P\right)\mathbf{v}\right]
     = -\rho\mathbf{v}\cdot\nabla\Phi

with total energy density :math:`E = P/(\gamma - 1) + \tfrac{1}{2}\rho v^2`,
adiabatic index :math:`\gamma` = ``hydro.eos.gamma``, and :math:`\Phi` the
gravitational potential, if any. The state is stored as the primitive fields
``rho``, ``vx``, ``vy`` and ``P``.

The solver is a second-order, finite-volume MUSCL-Hancock scheme. Each step:

#. computes the gradients of the primitive fields by centered differences,
   optionally slope limited (``hydro.slope_limiting``) so that the
   reconstructed face values stay within the range of the neighbouring cells;
#. predicts the primitive fields half a timestep ahead, from the primitive
   form of the equations;
#. extrapolates them to the cell faces, and solves the Riemann problem between
   the two states that meet at each face: with the local Lax-Friedrichs
   (Rusanov) flux (``hydro.riemann_solver = 'llf'``), or the HLLC flux
   (``'hllc'``), which resolves contact discontinuities;
#. updates the conserved mass, momentum and energy with the face fluxes, so
   they are conserved to round-off.

The boundary conditions, set per dimension in ``mesh.boundary_condition``, are
``'periodic'``; ``'reflective'`` (a mirror); ``'outflow'`` (zero gradient);
``'wall'`` (a mirror that, with a magnetic field, is a perfect conductor);
``'axis'``, in cylindrical geometry; and ``'driven'``, whose ghost cells are
supplied by the user through ``sim.driven_boundary`` (for example, to feed in
a current from an external circuit).


Magnetohydrodynamics
--------------------

Switching on ``physics.magnetic`` as well solves the equations of ideal
magnetohydrodynamics, in units with :math:`\mu_0 = 1`,

.. math::

   \frac{\partial \rho}{\partial t} + \nabla\cdot\left(\rho\mathbf{v}\right) = 0

.. math::

   \frac{\partial (\rho\mathbf{v})}{\partial t}
     + \nabla\cdot\left(\rho\mathbf{v}\mathbf{v} - \mathbf{B}\mathbf{B}
     + P^*\mathbf{I}\right) = 0

.. math::

   \frac{\partial E}{\partial t}
     + \nabla\cdot\left[\left(E + P^*\right)\mathbf{v}
     - \mathbf{B}\left(\mathbf{B}\cdot\mathbf{v}\right)\right] = 0

.. math::

   \frac{\partial \mathbf{B}}{\partial t}
     - \nabla\times\left(\mathbf{v}\times\mathbf{B}\right) = 0,
   \qquad \nabla\cdot\mathbf{B} = 0

with total pressure :math:`P^* = P + \tfrac{1}{2}B^2` and total energy density
:math:`E = P/(\gamma - 1) + \tfrac{1}{2}\rho v^2 + \tfrac{1}{2}B^2`. Note that
with a magnetic field the state field ``P`` holds the total pressure
:math:`P^*`.

The fluid is advanced with the same MUSCL-Hancock scheme, with the local
Lax-Friedrichs flux (``'llf'``) or the HLLD flux (``'hlld'``), which resolves
contact and Alfven discontinuities. The in-plane magnetic field is evolved by constrained
transport: it is stored on the cell faces (``bx`` on the :math:`x`-faces,
``by`` on the :math:`y`-faces), and updated by the curl of the electric field
:math:`E_z` at the cell corners, which is the average of the four face fluxes
that meet there. The face-centered divergence of the field therefore stays at
its initial value (zero) to round-off.

Switching on ``physics.rotation`` evolves the out-of-plane velocity and field
too (``vz`` and ``bz``, or ``vphi`` and ``bphi`` in cylindrical geometry), for
2.5D problems such as Alfven waves or a Z-pinch. In cylindrical geometry the
induction equation for the poloidal field uses the cylindrical curl,
:math:`\partial b_z/\partial t = (1/R)\,\partial(R E_\phi)/\partial R`, and
the radial momentum feels the magnetic hoop stress :math:`-B_\phi^2/R`.


Gravity
-------

Switching on ``physics.gravity`` solves for the gravitational potential of the
gas (and of the Schrodinger field),

.. math::

   \nabla^2\Phi = 4\pi G\left(\rho - \bar{\rho}\right)

with the mean density :math:`\bar{\rho}` subtracted, as is usual for a
periodic domain. It is solved spectrally with FFTs, which requires periodic
boundaries. Switching on ``physics.external_potential`` adds a fixed potential
:math:`\Phi(x, y)` supplied by the user as ``sim.external_potential``. The
acceleration :math:`-\nabla\Phi` is computed spectrally, or by second-order
finite differences along non-periodic directions, and the kick adds it to the
momentum and its work to the energy.


Schrodinger-Poisson
-------------------

Switching on ``physics.quantum`` evolves a complex scalar field :math:`\psi`,
such as ultralight (fuzzy) dark matter, by the Schrodinger equation

.. math::

   i\hbar\frac{\partial \psi}{\partial t}
     = -\frac{\hbar^2}{2m}\nabla^2\psi + m\Phi\psi

normalized so that :math:`\rho = |\psi|^2` is its mass density, which sources
the gravitational potential :math:`\Phi` (see Gravity). The particle mass
:math:`m` is ``quantum.m_eV`` (see Units).

It is solved with the spectral split-step (kick-drift-kick) method: the kick
multiplies the field by the phase :math:`\exp(-i\,(m/\hbar)\,\Phi\,\Delta t/2)`,
and the drift multiplies its Fourier transform by
:math:`\exp(-i\,(\hbar/2m)\,k^2\,\Delta t)`. Both are exactly unitary, so
the total mass is conserved to round-off. The spectral method requires periodic
boundaries.

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

Magnetohydrodynamics (see above) and radiation transport (see below) are
available in cylindrical geometry. Self-gravity and the Schrodinger field are
not yet.

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
:math:`T = \mu m_p P/(k_B \rho)`, with the mean molecular weight :math:`\mu`
given by ``hydro.eos.mean_molecular_weight``. The speed of light, the radiation
constant and the rest come from the code units (see `Units`_).

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

Flux-limited diffusion
^^^^^^^^^^^^^^^^^^^^^^

Setting ``radiation.method`` to ``'diffusion'`` evolves only the radiation
energy density :math:`E_f` of each group (the state field ``Er``), which is
much cheaper, by

.. math::

   \frac{\partial E_f}{\partial t}
     = \nabla\cdot\left(D_f\nabla E_f\right)
     + c\,\sigma_{p,f}\left(4\pi\epsilon_f(T) - E_f\right),
   \qquad
   D_f = \frac{c\,\lambda(R)}{\sigma_{t,f}},
   \quad R = \frac{|\nabla E_f|}{\sigma_{t,f} E_f}

where the flux limiter :math:`\lambda(R)` (``radiation.flux_limiter``) goes to
1/3 in optically thick gas, which recovers the diffusion limit, and to
:math:`1/R` in optically thin gas, which caps the flux at :math:`cE_f`: that of
Levermore & Pomraning (1981), :math:`\lambda = (2+R)/(6+3R+R^2)`
(``'levermore_pomraning'``); :math:`(9+R^2)^{-1/2}` (``'larsen'``);
:math:`1/\max(3, R)` (``'max'``); or plain diffusion, :math:`\lambda = 1/3`
(``'none'``), which does not cap the flux. The diffusion, the flux limiter and
the exchange of energy with the gas are solved together, implicitly: the
limiter is evaluated from the energy density at the end of the step, as one
evaluated at its start lets the radiation diffuse without limit into gas it
has not yet reached. Like the transport, it conserves the total energy of the
radiation and the gas to round-off.

An ``'outflow'`` boundary is a vacuum (Marshak) boundary, through which the
radiation leaves at :math:`cE/2`, and a ``'driven'`` boundary also lets in
radiation of the energy density given by ``sim.driven_boundary``.

In a moving gas the radiation is carried along with it, to first order in
:math:`v/c`, and pushes on it with the force :math:`-\lambda\nabla E_f`: in
optically thick gas this is a radiation fluid of pressure :math:`E/3`, and in
optically thin gas the force vanishes.

Diffusion is accurate where the radiation is nearly isotropic, deep in
optically thick material. It cannot cast shadows, and it carries radiation
around obstacles and corners too easily: see the crooked pipe example, which
compares the two methods.

Radiation and a static material
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``physics.radiation`` on and ``physics.hydro`` off, the radiation
exchanges energy with a static material (``rho`` and ``P`` in the state,
which hold its density and, through the gas law, its temperature): it heats
and cools the material, but does not move it. There is then no Courant
condition, so ``time.num_timesteps`` must be set.


External circuits
-----------------

A simulation can be coupled to external circuits, such as the capacitor bank of
a pulsed power generator driving a Z-pinch, with ``adirondax.coupled_step``.
The domain and the circuits meet at ports: each port carries a current into the
domain and reports a voltage back. Each step, the domain (driven by the port
currents, for example through a ``'driven'`` boundary) and the circuits (driven
by the port voltages) are linearized about a first guess, using JAX to compute
the domain's port impedance and the circuits' port admittance, and their
responses are solved against each other with the Sherman-Morrison-Woodbury
formula, so that both sides take the step with the same currents and voltages.
