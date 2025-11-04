import jax
import jax.numpy as jnp
import copy

from .constants import constants
from .hydro.euler2d import hydro_euler2d_fluxes
from .hydro.mhd2d import hydro_mhd2d_fluxes
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

        # simulation state
        self.state = {}
        self.state["t"] = jnp.array(0.0)
        if params["physics"]["hydro"]:
            self.state["rho"] = jnp.zeros((self._nx, self._ny))
            self.state["vx"] = jnp.zeros((self._nx, self._ny))
            self.state["vy"] = jnp.zeros((self._nx, self._ny))
            self.state["P"] = jnp.zeros((self._nx, self._ny))
        if params["physics"]["magnetic"]:
            self.state["bx"] = jnp.zeros((self._nx, self._ny))
            self.state["by"] = jnp.zeros((self._nx, self._ny))
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

    @property
    def kgrid(self):
        n = self._nx
        L = self._Lx
        klin = 2.0 * jnp.pi / L * jnp.arange(-n / 2, n / 2)
        kx, ky = jnp.meshgrid(klin, klin)
        kx = jnp.fft.ifftshift(kx)
        ky = jnp.fft.ifftshift(ky)
        return kx, ky

    def _calc_grav_potential(self, state, k_sq, use_quantum, use_hydro):
        G = 4000.0  # XXX
        rho_tot = 0.0
        if use_quantum:
            rho_tot += jnp.abs(state["psi"]) ** 2
        if use_hydro:
            rho_tot += state["rho"]
        rho_bar = jnp.mean(rho_tot)
        V = calculate_gravitational_potential(rho_tot, k_sq, G, rho_bar)
        return V

    @property
    def potential(self):
        kx, ky = self.kgrid
        k_sq = kx**2 + ky**2
        return self._calc_grav_potential(
            self.state,
            k_sq,
            self.params["physics"]["quantum"],
            self.params["physics"]["hydro"],
        )

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

        # Physics flags
        use_hydro = self.params["physics"]["hydro"]
        use_magnetic = self.params["physics"]["magnetic"]
        use_quantum = self.params["physics"]["quantum"]
        use_gravity = self.params["physics"]["gravity"]

        gamma = self.params["hydro"]["eos"]["gamma"] if use_hydro else None

        # Precompute Fourier space variables
        k_sq = None
        if use_gravity or use_quantum:
            kx, ky = self.kgrid
            k_sq = kx**2 + ky**2

        # Initialize potential
        V = None
        if use_gravity:
            V = self._calc_grav_potential(state, k_sq, use_quantum, use_hydro)

        # Build the carry: (state, V)
        carry = (state, V)

        def step_fn(carry, _):
            """
            Pure step function: advances state by one timestep.
            Returns new carry and None (no stacked outputs).
            """
            state, V = carry

            # Create new state dict to avoid mutation
            new_state = {}

            # Kick (half-step) - quantum + gravity
            psi = state.get("psi")
            if use_quantum and use_gravity and psi is not None:
                psi = quantum_kick(psi, V, 1.0, dt / 2.0)

            # Drift (full-step) - quantum
            if use_quantum and psi is not None:
                psi = quantum_drift(psi, k_sq, 1.0, dt)

            if use_quantum:
                new_state["psi"] = psi

            # Drift (full-step) - hydro
            if use_hydro:
                if use_magnetic:
                    rho, vx, vy, P, bx, by = hydro_mhd2d_fluxes(
                        state["rho"],
                        state["vx"],
                        state["vy"],
                        state["P"],
                        state["bx"],
                        state["by"],
                        gamma,
                        dx,
                        dt,
                    )
                    new_state["rho"] = rho
                    new_state["vx"] = vx
                    new_state["vy"] = vy
                    new_state["P"] = P
                    new_state["bx"] = bx
                    new_state["by"] = by
                else:
                    rho, vx, vy, P = hydro_euler2d_fluxes(
                        state["rho"],
                        state["vx"],
                        state["vy"],
                        state["P"],
                        gamma,
                        dx,
                        dt,
                    )
                    new_state["rho"] = rho
                    new_state["vx"] = vx
                    new_state["vy"] = vy
                    new_state["P"] = P

            # Update potential
            new_V = V
            if use_gravity:
                new_V = self._calc_grav_potential(
                    new_state, k_sq, use_quantum, use_hydro
                )

            # Kick (half-step) - quantum + gravity
            if use_quantum and use_gravity:
                new_state["psi"] = quantum_kick(new_state["psi"], new_V, 1.0, dt / 2.0)

            # Update time
            new_state["t"] = state["t"] + dt

            return (new_state, new_V), None

        # Run the entire loop as a single JIT-compiled function
        def run_loop(carry):
            final_carry, _ = jax.lax.scan(step_fn, carry, xs=None, length=nt)
            return final_carry

        # Execute the compiled loop
        state, V = run_loop(carry)

        return state

    def run(self):
        """
        Run the simulation
        """
        self.state = self._evolve(self.state)
        if "psi" in self.state:
            jax.block_until_ready(self.state["psi"])
        elif "rho" in self.state:
            jax.block_until_ready(self.state["rho"])
        else:
            jax.block_until_ready(self.state["t"])
