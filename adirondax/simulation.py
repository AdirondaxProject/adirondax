import jax
import jax.numpy as jnp
from functools import partial
import copy

from .hydro import (
    get_conserved,
    get_primitive,
    get_gradient,
    extrapolate_to_face,
    apply_fluxes,
    get_flux,
    update_hydro,
)


class Simulation:
    """
    Simulation: The base class for an astrophysics simulation.

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

    @partial(jax.jit, static_argnames=["self"])
    def evolve(self, state):
        """
        This function evolves the simulation state according to the simulation parameters/physics.

        Parameters
        ----------
        state: jax.pytree
          The current state of the simulation.

        Returns
        -------
        psi: jax.pytree
          The evolved state of the simulation.
        """

        # Simulation parameters
        n = self._nx
        dt = self._dt
        nt = self._nt
        G = 4000.0  # gravitational constant
        L = 1.0  # domain size
        gamma = 5.0 / 3.0  # adiabatic index
        vol = self._vol
        courant_fac = 0.4
        dx = self._dx

        # Fourier space variables
        klin = 2.0 * jnp.pi / L * jnp.arange(-n / 2, n / 2)
        kx, ky = jnp.meshgrid(klin, klin)
        kx = jnp.fft.ifftshift(kx)
        ky = jnp.fft.ifftshift(ky)
        kSq = kx**2 + ky**2

        if self.params["physics"]["hydro"]:
            rho = state["rho"]
            vx = state["vx"]
            vy = state["vy"]
            P = state["P"]
            state["Mass"], state["Momx"], state["Momy"], state["Energy"] = (
                get_conserved(rho, vx, vy, P, gamma, vol)
            )

        def update(i, state):

            # TODO: move these to separate functions

            if self.params["physics"]["quantum"]:

                # drift
                psi = state["psi"]
                psihat = jnp.fft.fftn(psi)
                psihat = jnp.exp(dt * (-1.0j * kSq / 2.0)) * psihat
                psi = jnp.fft.ifftn(psihat)

                # update potential
                Vhat = -jnp.fft.fftn(4.0 * jnp.pi * G * (jnp.abs(psi) ** 2 - 1.0)) / (
                    kSq + (kSq == 0)
                )
                V = jnp.real(jnp.fft.ifftn(Vhat))

                # kick
                state["psi"] = jnp.exp(-1.0j * dt * V) * psi

            if self.params["physics"]["hydro"]:
                (
                    state["Mass"],
                    state["Momx"],
                    state["Momy"],
                    state["Energy"],
                    state["rho"],
                ) = update_hydro(
                    state["Mass"],
                    state["Momx"],
                    state["Momy"],
                    state["Energy"],
                    vol,
                    dx,
                    gamma,
                    dt,
                )

            # update time
            state["t"] += nt * dt

            return state

        # Simulation Main Loop
        state = jax.lax.fori_loop(0, nt, update, init_val=state)

        return state
