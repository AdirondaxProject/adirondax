import jax
import jax.numpy as jnp

from .constants import constants
from .hydro.euler2d import hydro_euler2d_fluxes, hydro_euler2d_timestep
from .hydro.mhd2d import hydro_mhd2d_fluxes, hydro_mhd2d_timestep
from .quantum import quantum_kick, quantum_drift, quantum_timestep
from .gravity import calculate_gravitational_potential
from .utils import set_up_parameters, print_parameters


class Simulation:
    """
    Simulation: The base class for a multi-physics simulation.

    Parameters
    ----------
      params (dict): The python dictionary that contains the simulation parameters.

    """

    def __init__(self, params):
        # start from default simulation parameters and update with user params
        self._params = set_up_parameters(params)

        # additional checks (TODO: move these into separate function(s))
        if len(self.resolution) != len(self.box_size):
            raise ValueError("'resolution' and 'box_size' must have same shape")

        if self.dim == 3:
            raise NotImplementedError("3D is not yet implemented.")

        if self.params["hydro"]["riemann_solver"] not in ["llf", "hlld"]:
            raise ValueError("riemann solver does not exist")

        if (
            self.params["hydro"]["riemann_solver"] == "hlld"
            and not self.params["physics"]["magnetic"]
        ):
            raise ValueError("'hlld' riemann solver only exists for magnetic=True")

        # print info
        if jax.process_index() == 0:
            print("Simulation parameters:")
            print_parameters(self.params)

        # simulation state
        self.state = {}
        self.state["t"] = jnp.array(0.0) + jnp.nan
        if self.params["physics"]["hydro"]:
            self.state["rho"] = jnp.zeros(self.resolution) + jnp.nan
            self.state["vx"] = jnp.zeros(self.resolution) + jnp.nan
            self.state["vy"] = jnp.zeros(self.resolution) + jnp.nan
            self.state["P"] = jnp.zeros(self.resolution) + jnp.nan
        if self.params["physics"]["magnetic"]:
            self.state["bx"] = jnp.zeros(self.resolution) + jnp.nan
            self.state["by"] = jnp.zeros(self.resolution) + jnp.nan
        if self.params["physics"]["quantum"]:
            self.state["psi"] = (
                jnp.zeros(self.resolution, dtype=jnp.complex64) + jnp.nan
            )

        # extra info to keep track of
        self.state["steps_taken"] = jnp.array(0) + jnp.nan

        # functions
        self.external_potential = None

    @property
    def resolution(self):
        """
        Return the resolution (per dimension) of the simulation
        """
        return self.params["mesh"]["resolution"]

    @property
    def box_size(self):
        """
        Return the box size of the simulation
        """
        return self.params["mesh"]["box_size"]

    @property
    def dim(self):
        """
        Return the dimension of the simulation
        """
        return len(self.resolution)

    @property
    def steps_taken(self):
        """
        Return the number of steps taken in the simulation
        """
        return self.state["steps_taken"]

    @property
    def params(self):
        """
        Return the parameters of the simulation
        """
        return self._params

    @property
    def mesh(self):
        """
        Return the simulation mesh
        """
        Lx = self.box_size[0]
        Ly = self.box_size[1]
        nx = self.resolution[0]
        ny = self.resolution[1]
        dx = Lx / nx
        dy = Ly / ny
        xlin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
        ylin = jnp.linspace(0.5 * dy, Ly - 0.5 * dy, ny)
        X, Y = jnp.meshgrid(xlin, ylin, indexing="ij")
        return X, Y

    @property
    def kgrid(self):
        """
        Return the simulation spectral grid
        """
        Lx = self.box_size[0]
        nx = self.resolution[0]
        klin = (2.0 * jnp.pi / Lx) * jnp.arange(-nx / 2, nx / 2)
        kx, ky = jnp.meshgrid(klin, klin)
        kx = jnp.fft.ifftshift(kx)
        ky = jnp.fft.ifftshift(ky)
        return kx, ky

    def _calc_grav_potential(self, state, k_sq, G, use_quantum, use_hydro):
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
        """
        Return the gravitational potential
        """
        kx, ky = self.kgrid
        k_sq = kx**2 + ky**2
        return self._calc_grav_potential(
            self.state,
            k_sq,
            constants["gravitational_constant"],
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
        Lx = self.box_size[0]
        Ly = self.box_size[1]
        nx = self.resolution[0]
        ny = self.resolution[1]
        dx = Lx / nx
        dy = Ly / ny
        nt = self.params["time"]["num_timesteps"]
        t_span = self.params["time"]["span"]

        use_adaptive_timesteps = True if nt < 1 else False
        dt_ref = jnp.nan if use_adaptive_timesteps else t_span / nt

        # Physics flags
        use_hydro = self.params["physics"]["hydro"]
        use_magnetic = self.params["physics"]["magnetic"]
        use_quantum = self.params["physics"]["quantum"]
        use_gravity = self.params["physics"]["gravity"]
        use_external_potential = self.params["physics"]["external_potential"]

        G = constants["gravitational_constant"]

        gamma = self.params["hydro"]["eos"]["gamma"]
        cfl = self.params["hydro"]["cfl"]
        riemann_solver_type = self.params["hydro"]["riemann_solver"]
        use_slope_limiting = self.params["hydro"]["slope_limiting"]

        m_per_hbar = 1.0  # XXX

        # Precompute Fourier space variables
        k_sq = None
        if use_gravity or use_quantum:
            kx, ky = self.kgrid
            k_sq = kx**2 + ky**2

        # Build the carry:
        carry = (state, k_sq)

        def _get_timestep(state):
            dt = jnp.inf
            if use_hydro:
                if use_magnetic:
                    dt_hydro = hydro_mhd2d_timestep(
                        state["rho"],
                        state["vx"],
                        state["vy"],
                        state["P"],
                        state["bx"],
                        state["by"],
                        gamma,
                        dx,
                        dy,
                    )
                else:
                    dt_hydro = hydro_euler2d_timestep(
                        state["rho"],
                        state["vx"],
                        state["vy"],
                        state["P"],
                        gamma,
                        dx,
                        dy,
                    )
                dt = jnp.minimum(dt, cfl * dt_hydro)
            if use_quantum:
                dt_quantum = quantum_timestep(m_per_hbar, dx, dy)
                dt = jnp.minimum(dt, dt_quantum)
            dt = jnp.minimum(dt, t_span - state["t"])
            return dt

        def _kick(state, k_sq, dt):
            # Kick (half-step)

            # update potential
            if use_gravity:
                V = self._calc_grav_potential(state, k_sq, G, use_quantum, use_hydro)

            # apply
            if use_gravity or use_external_potential:
                if use_quantum:
                    state["psi"] = quantum_kick(state["psi"], V, m_per_hbar, dt / 2.0)

        def _drift(state, k_sq, dt):
            # Drift (full-step)

            if use_quantum:
                state["psi"] = quantum_drift(state["psi"], k_sq, m_per_hbar, dt)

            if use_hydro:
                if use_magnetic:
                    (
                        state["rho"],
                        state["vx"],
                        state["vy"],
                        state["P"],
                        state["bx"],
                        state["by"],
                    ) = hydro_mhd2d_fluxes(
                        state["rho"],
                        state["vx"],
                        state["vy"],
                        state["P"],
                        state["bx"],
                        state["by"],
                        gamma,
                        dx,
                        dy,
                        dt,
                        riemann_solver_type,
                        use_slope_limiting,
                    )
                else:
                    state["rho"], state["vx"], state["vy"], state["P"] = (
                        hydro_euler2d_fluxes(
                            state["rho"],
                            state["vx"],
                            state["vy"],
                            state["P"],
                            gamma,
                            dx,
                            dy,
                            dt,
                            riemann_solver_type,
                            use_slope_limiting,
                        )
                    )

        def step_fn(carry):
            """
            Pure step function: advances state by one timestep.
            """
            state, k_sq = carry

            # Get the timestep
            dt = dt_ref
            if use_adaptive_timesteps:
                dt = _get_timestep(state)

            # kick-drift-kick
            _kick(state, k_sq, 0.5 * dt)
            _drift(state, k_sq, dt)
            _kick(state, k_sq, 0.5 * dt)

            # Update time
            state["t"] = state["t"] + dt

            # Update diagnostics
            state["steps_taken"] = state["steps_taken"] + 1

            return (state, k_sq)

        # Run the entire loop as a single JIT-compiled function
        def run_loop(carry):
            if use_adaptive_timesteps:
                # def cond_fn(carry):
                #    state, _ = carry
                #    return state["t"] < t_span * (1.0 - 1e-10)

                # final_carry = jax.lax.while_loop(cond_fn, step_fn, carry)

                # do a simple while loop
                state, _ = carry
                while state["t"] < t_span * (1.0 - 1e-10):
                    carry = step_fn(carry)
                    state, _ = carry
                final_carry = carry
            else:

                def step_fn_stacked(carry, _):
                    # Returns new carry and None (no stacked outputs) for jax.lax.scan()
                    return step_fn(carry), None

                final_carry, _ = jax.lax.scan(
                    step_fn_stacked, carry, xs=None, length=nt
                )
            return final_carry

        # Execute the compiled loop
        state, _ = run_loop(carry)

        return state

    def run(self):
        """
        Run the simulation
        """
        self.state["steps_taken"] = 0
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
