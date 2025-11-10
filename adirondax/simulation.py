import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import json
import os

from .constants import constants
from .hydro.common2d import hydro_accelerate
from .hydro.euler2d import hydro_euler2d_fluxes, hydro_euler2d_timestep
from .hydro.mhd2d import hydro_mhd2d_fluxes, hydro_mhd2d_timestep
from .quantum import quantum_kick, quantum_drift, quantum_timestep
from .gravity import calculate_gravitational_potential, get_acceleration
from .utils import set_up_parameters, print_parameters
from .visualization import plot_sim


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

        if self.params["output"]["save"] and self.params["time"]["num_timesteps"] > 0:
            if (
                self.params["time"]["num_timesteps"]
                % self.params["output"]["num_checkpoints"]
                != 0
            ):
                raise ValueError("'num_checkpoints' must divide 'num_timesteps'")

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
        x_lin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
        y_lin = jnp.linspace(0.5 * dy, Ly - 0.5 * dy, ny)
        xx, yy = jnp.meshgrid(x_lin, y_lin, indexing="ij")
        return xx, yy

    @property
    def kgrid(self):
        """
        Return the simulation spectral grid
        """
        Lx = self.box_size[0]
        Ly = self.box_size[1]
        nx = self.resolution[0]
        ny = self.resolution[1]
        kx_lin = (2.0 * jnp.pi / Lx) * jnp.arange(-nx / 2, nx / 2)
        ky_lin = (2.0 * jnp.pi / Ly) * jnp.arange(-ny / 2, ny / 2)
        kx, ky = jnp.meshgrid(kx_lin, ky_lin, indexing="ij")
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

        # constants
        G = constants["gravitational_constant"]

        # physics variables
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

        # Checkpointer
        save = self.params["output"]["save"]
        num_checkpoints = self.params["output"]["num_checkpoints"]
        if save:
            checkpoint_dir = checkpoint_dir = os.path.join(
                os.getcwd(), self.params["output"]["path"]
            )
            path = os.path.join(os.getcwd(), checkpoint_dir)
            if jax.process_index() == 0:
                path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

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
            if use_gravity and use_external_potential:
                xx, yy = self.mesh
                V = self._calc_grav_potential(
                    state, k_sq, G, use_quantum, use_hydro
                ) + self.external_potential(xx, yy)
            elif use_gravity:
                V = self._calc_grav_potential(state, k_sq, G, use_quantum, use_hydro)
            elif use_external_potential:
                xx, yy = self.mesh
                V = self.external_potential(xx, yy)

            # apply
            if use_gravity or use_external_potential:
                if use_quantum:
                    state["psi"] = quantum_kick(state["psi"], V, m_per_hbar, dt / 2.0)
                if use_hydro:
                    kx, ky = self.kgrid
                    ax, ay = get_acceleration(V, kx, ky)
                    state["vx"], state["vy"] = hydro_accelerate(
                        state["vx"], state["vy"], ax, ay, dt
                    )

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
                if save:
                    raise NotImplementedError("implement me.")
                else:
                    # def cond_fn(carry):
                    #    state, _ = carry
                    #    return state["t"] < t_span * (1.0 - 1e-10)

                    # carry = jax.lax.while_loop(cond_fn, step_fn, carry)

                    # do a simple while loop
                    state, _ = carry
                    while state["t"] < t_span * (1.0 - 1e-10):
                        carry = step_fn(carry)
                        state, _ = carry
            else:

                def step_fn_stacked(carry, _):
                    # Returns new carry and None (no stacked outputs) for jax.lax.scan()
                    return step_fn(carry), None

                if save:
                    nt_sub = int(round(nt / num_checkpoints))
                    for i in range(1, num_checkpoints + 1):
                        carry, _ = jax.lax.scan(
                            step_fn_stacked, carry, xs=None, length=nt_sub
                        )
                        state, _ = carry
                        jax.block_until_ready(state)
                        # save state
                        plot_sim(state, checkpoint_dir, i, self.params)
                else:
                    carry, _ = jax.lax.scan(step_fn_stacked, carry, xs=None, length=nt)
            return carry

        # save initial state
        if jax.process_index() == 0:
            print(f"Starting simulation (res={self.resolution}, nt={nt}) ...")
        if self.params["output"]["save"]:
            with open(os.path.join(checkpoint_dir, "params.json"), "w") as f:
                json.dump(self.params, f, indent=2)
            plot_sim(state, checkpoint_dir, 0, self.params)

        # Simulation Main Loop
        state, _ = run_loop(carry)

        return state

    def run(self):
        """
        Run the simulation
        """
        self.state["steps_taken"] = 0
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
