import jax
import jax.numpy as jnp
from functools import partial
import copy

from .constants import constants
from .hydro import update_hydro
from .quantum import quantum_kick, quantum_drift
from .gravity import calculate_gravitational_potential


class Simulation:
    """
    Simulation: The base class for a multi-physics simulation.

    Parameters
    ----------
      params (dict): The python dictionary that contains the simulation parameters.

    """

    def __init__(self, params):
        # simulation parameters
        self._params = copy.deepcopy(params)
        self._nt = params["simulation"]["n_timestep"]
        self._dt = params["simulation"]["timestep"]
        self._dim = len(params["mesh"]["resolution"])
        self._nx = params["mesh"]["resolution"][0]
        self._Lx = params["mesh"]["boxsize"][0]
        self._dx = self._Lx / self._nx
        if self._dim > 1:
            self._ny = params["mesh"]["resolution"][1]
            self._Ly = params["mesh"]["boxsize"][1]
            self._dy = self._Ly / self._ny
        if self._dim == 3:
            self._nz = params["mesh"]["resolution"][2]
            self._Lz = params["mesh"]["boxsize"][2]
            self._dz = self._Lz / self._nz
        self._vol = self._dx
        if self._dim > 1:
            self._vol *= self._dy
        if self._dim == 3:
            self._vol *= self._dz

        # simulation state
        self.state = {}
        self.state["t"] = 0.0
        if params["physics"]["hydro"]:
            self.state["rho"] = jnp.zeros((self._nx, self._ny))
            self.state["vx"] = jnp.zeros((self._nx, self._ny))
            self.state["vy"] = jnp.zeros((self._nx, self._ny))
            self.state["P"] = jnp.zeros((self._nx, self._ny))
        if params["physics"]["quantum"]:
            self.state["psi"] = jnp.zeros((self._nx, self._ny), dtype=jnp.complex64)

        # internal simulation state -- should not be touched by user!
        self._internal = {}
        if params["physics"]["gravity"]:
            self._internal["V"] = jnp.zeros((self._nx, self._ny))

    @property
    def nt(self):
        return self._nt

    @property
    def dt(self):
        return self._dt

    @property
    def dim(self):
        return self._dim

    @property
    def params(self):
        return self._params

    @property
    def mesh(self):
        dx = self._dx
        dy = self._dy
        xlin = jnp.linspace(0.5 * dx, self._Lx - 0.5 * dx, self._nx)
        ylin = jnp.linspace(0.5 * dy, self._Ly - 0.5 * dy, self._ny)
        X, Y = jnp.meshgrid(xlin, ylin, indexing="ij")
        return X, Y

    @property
    def kgrid(self):
        n = self._nx
        L = self._Lx
        klin = 2.0 * jnp.pi / L * jnp.arange(-n / 2, n / 2)
        kx, ky = jnp.meshgrid(klin, klin)
        kx = jnp.fft.ifftshift(kx)
        ky = jnp.fft.ifftshift(ky)
        return kx, ky

    def _calc_grav_potential(self, state, k_sq):
        G = 4000.0  # XXX
        rho_tot = 0.0
        if self.params["physics"]["quantum"]:
            rho_tot += jnp.abs(state["psi"]) ** 2
        if self.params["physics"]["hydro"]:
            rho_tot += state["rho"]
        rho_bar = jnp.mean(rho_tot)
        V = calculate_gravitational_potential(rho_tot, k_sq, G, rho_bar)
        return V

    @property
    def potential(self):
        kx, ky = self.kgrid
        k_sq = kx**2 + ky**2
        return self._calc_grav_potential(self.state, k_sq)

    @partial(jax.jit, static_argnames=["self"])
    def _evolve(self, state):
        """
        This function evolves the simulation state according to the simulation parameters/physics.

        Parameters
        ----------
        state: jax.pytree
          The current state of the simulation.

        Returns
        -------
        state: jax.pytree
          The evolved state of the simulation.
        """

        # Simulation parameters
        dt = self._dt
        nt = self._nt
        dx = self._dx
        vol = self._vol
        courant_fac = 0.4
        if self.params["physics"]["hydro"]:
            gamma = self.params["hydro"]["eos"]["gamma"]

        # Fourier space variables
        if self.params["physics"]["gravity"] or self.params["physics"]["quantum"]:
            kx, ky = self.kgrid
            k_sq = kx**2 + ky**2

        # initialize potential
        if self.params["physics"]["gravity"]:
            self._internal["V"] = self._calc_grav_potential(state, k_sq)

        def update(i, state):
            # Update the simulation state by one timestep
            # according to a 2nd-order `kick-drift-kick` scheme

            # Kick (half-step)
            if self.params["physics"]["quantum"] and self.params["physics"]["gravity"]:
                state["psi"] = quantum_kick(
                    state["psi"], self._internal["V"], 1.0, dt / 2.0
                )

            # Drift (full-step)
            if self.params["physics"]["quantum"]:
                state["psi"] = quantum_drift(state["psi"], k_sq, 1.0, dt)

            if self.params["physics"]["hydro"]:
                (
                    state["rho"],
                    state["vx"],
                    state["vy"],
                    state["P"],
                ) = update_hydro(
                    state["rho"],
                    state["vx"],
                    state["vy"],
                    state["P"],
                    vol,
                    dx,
                    gamma,
                    dt,
                )

            # update potential
            if self.params["physics"]["gravity"]:
                self._internal["V"] = self._calc_grav_potential(state, k_sq)

            # Kick (half-step)
            if self.params["physics"]["quantum"] and self.params["physics"]["gravity"]:
                state["psi"] = quantum_kick(
                    state["psi"], self._internal["V"], 1.0, dt / 2.0
                )

            # update time
            state["t"] += nt * dt

            return state

        # Simulation Main Loop
        state = jax.lax.fori_loop(0, nt, update, init_val=state)

        return state

    def run(self):
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
