import jax
import jax.numpy as jnp
from functools import partial
import copy


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
        if self._dim > 1:
            self._ny = params["mesh"]["resolution"][1]
        if self._dim == 3:
            self._nz = params["mesh"]["resolution"][2]

        # simulation state
        self.state = {}
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

    @partial(jax.jit, static_argnames=["self", "dt", "nt"])
    def evolve(self, state, dt, nt):
        """
        TODO: make general, update description

        This function evolves the wave function psi.
        Assumes a periodic domain [0,1] x [0,1] and
        `<|psi|^2> = 1`.

        Parameters
        ----------
        psi: jax.numpy.ndarray
          The (complex) array holding the wave function for the simulation.
        dt: float
          The time step for the simulation.
        nt: int
          The number of time steps for the simulation.

        Returns
        -------
        psi: jax.numpy.ndarray
          The evolved wave function.
        """

        # Simulation parameters
        n = state["psi"].shape[0]
        G = 4000.0  # gravitational constant
        L = 1.0  # domain size

        # Fourier space variables
        klin = 2.0 * jnp.pi / L * jnp.arange(-n / 2, n / 2)
        kx, ky = jnp.meshgrid(klin, klin)
        kx = jnp.fft.ifftshift(kx)
        ky = jnp.fft.ifftshift(ky)
        kSq = kx**2 + ky**2

        def update(i, psi):

            # drift
            psihat = jnp.fft.fftn(psi)
            psihat = jnp.exp(dt * (-1.0j * kSq / 2.0)) * psihat
            psi = jnp.fft.ifftn(psihat)

            # update potential
            Vhat = -jnp.fft.fftn(4.0 * jnp.pi * G * (jnp.abs(psi) ** 2 - 1.0)) / (
                kSq + (kSq == 0)
            )
            V = jnp.real(jnp.fft.ifftn(Vhat))

            # kick
            psi = jnp.exp(-1.0j * dt * V) * psi

            return psi

        # Simulation Main Loop
        state["psi"] = jax.lax.fori_loop(0, nt, update, init_val=state["psi"])

        return state
