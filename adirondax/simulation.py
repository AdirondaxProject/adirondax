import json
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from .constants import constants
from .gravity import calculate_gravitational_potential, get_acceleration
from .hydro.euler2d import (
    hydro_euler2d_accelerate,
    hydro_euler2d_fluxes,
    hydro_euler2d_timestep,
)
from .hydro.geometry import get_geometry
from .hydro.mhd2d import hydro_mhd2d_fluxes, hydro_mhd2d_timestep
from .quantum import quantum_drift, quantum_kick, quantum_timestep
from .utils import print_parameters, set_up_parameters
from .validation import validate_params
from .visualization import plot_sim


class Simulation:
    """
    Simulation: The base class for a multi-physics simulation.

    Parameters
    ----------
      params (dict): The python dictionary that contains the simulation parameters.

    Attributes
    ----------
      external_potential (callable): (x, y) -> V, the external gravitational
        potential, used when physics.external_potential is on.
      driven_boundary (callable): (state, axis) -> {key: (lo, hi)}, the ghost
        cells of a 'driven' boundary along the given axis, for each field.

    """

    def __init__(self, params):
        # start from default simulation parameters and update with user params
        self._params = set_up_parameters(params)

        validate_params(self._params)

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
        if self.params["physics"]["rotation"]:
            v_out, b_out = self.out_of_plane_keys
            self.state[v_out] = jnp.zeros(self.resolution) + jnp.nan
            if self.params["physics"]["magnetic"]:
                self.state[b_out] = jnp.zeros(self.resolution) + jnp.nan
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
        self.driven_boundary = None

        self._advance_cache = None

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
    def origin(self):
        """
        Return the lower corner of the simulation domain
        """
        return self.params["mesh"]["origin"]

    @property
    def geometry(self):
        """
        Return the geometry of the simulation mesh
        """
        return self.params["mesh"]["geometry"]

    @property
    def is_cylindrical(self):
        """
        Return whether the mesh is an axisymmetric cylindrical (R,z) mesh
        """
        return self.geometry == "cylindrical"

    @property
    def out_of_plane_keys(self):
        """
        Return the state keys of the out-of-plane velocity and magnetic field
        """
        return ("vphi", "bphi") if self.is_cylindrical else ("vz", "bz")

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
        Return the simulation mesh (cell centers).

        For cylindrical geometry the two returned arrays are (R, z), with
        R in (0, box_size[0]) and z in (0, box_size[1]).
        """
        Lx = self.box_size[0]
        Ly = self.box_size[1]
        nx = self.resolution[0]
        ny = self.resolution[1]
        dx = Lx / nx
        dy = Ly / ny
        x0, y0 = self.origin[0], self.origin[1]
        x_lin = jnp.linspace(x0 + 0.5 * dx, x0 + Lx - 0.5 * dx, nx)
        y_lin = jnp.linspace(y0 + 0.5 * dy, y0 + Ly - 0.5 * dy, ny)
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
        kx_lin = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=Lx / nx)
        ky_lin = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=Ly / ny)
        kx, ky = jnp.meshgrid(kx_lin, ky_lin, indexing="ij")
        return kx, ky

    @property
    def _hydro_keys(self):
        """
        Return the map from state keys to the field names of the hydro solvers
        """
        physics = self.params["physics"]
        v_out, b_out = self.out_of_plane_keys
        keys = {"rho": "rho", "vx": "vx", "vy": "vy", "P": "P"}
        if physics["magnetic"]:
            keys.update(bx="bx", by="by")
        if physics["rotation"]:
            keys[v_out] = "vz"
            if physics["magnetic"]:
                keys[b_out] = "bz"
        return keys

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

    def _get_advance(self):
        """
        Return the jit-compiled advance function, rebuilding it only when the
        parameters or the user-supplied functions have changed.
        """
        key = (
            json.dumps(self.params, sort_keys=True),
            self.external_potential,
            self.driven_boundary,
        )
        if self._advance_cache is None or self._advance_cache[0] != key:
            self._advance_cache = (key, jax.jit(self._build_advance()))
        return self._advance_cache[1]

    def _build_advance(self):
        """
        Build the function advance(state, t_target), which evolves the state up
        to t_target with adaptive timesteps, or else by a fixed number of
        timesteps (all of them, or those between two checkpoints).
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
        bc_x = self.params["mesh"]["boundary_condition"][0]
        bc_y = self.params["mesh"]["boundary_condition"][1]
        save = self.params["output"]["save"]
        num_checkpoints = self.params["output"]["num_checkpoints"]

        use_adaptive_timesteps = nt < 1
        if not use_adaptive_timesteps:
            dt_ref = t_span / nt
            num_steps = nt // num_checkpoints if save else nt

        x_has_ghosts = bc_x != "periodic"
        y_has_ghosts = bc_y != "periodic"

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
        riemann_solver = self.params["hydro"]["riemann_solver"]
        slope_limiting = self.params["hydro"]["slope_limiting"]

        m_per_hbar = 1.0  # XXX

        if use_magnetic:
            hydro_fluxes, hydro_timestep = hydro_mhd2d_fluxes, hydro_mhd2d_timestep
        else:
            hydro_fluxes, hydro_timestep = hydro_euler2d_fluxes, hydro_euler2d_timestep

        hydro_keys = self._hydro_keys
        external_potential = self.external_potential
        driven_boundary = self.driven_boundary

        def to_hydro(fields):
            return {h: fields[k] for k, h in hydro_keys.items()}

        def from_hydro(state, W):
            return {**state, **{k: W[h] for k, h in hydro_keys.items()}}

        def driven_ghosts(state, axis, bc):
            return to_hydro(driven_boundary(state, axis)) if bc == "driven" else None

        def advance(state, t_target):
            # mesh metric factors: 'geom' on the bare grid, 'geom_work' on the
            # ghost-extended grid the flux routine operates on
            geom = get_geometry(
                self.geometry, self.box_size, self.resolution, r_min=self.origin[0]
            )
            geom_work = get_geometry(
                self.geometry,
                self.box_size,
                self.resolution,
                num_ghost_x=1 if x_has_ghosts else 0,
                r_min=self.origin[0],
            )
            kx, ky = self.kgrid
            k_sq = kx**2 + ky**2
            if use_external_potential:
                V_ext = external_potential(*self.mesh)

            def get_timestep(state):
                dt = jnp.inf
                if use_hydro:
                    dt_hydro = hydro_timestep(to_hydro(state), gamma, dx, dy)
                    dt = jnp.minimum(dt, cfl * dt_hydro)
                if use_quantum:
                    dt_quantum = quantum_timestep(m_per_hbar, dx, dy)
                    dt = jnp.minimum(dt, dt_quantum)
                dt = jnp.minimum(dt, t_target - state["t"])
                return dt

            def kick(state, dt):
                if not (use_gravity or use_external_potential):
                    return state
                state = dict(state)

                if use_gravity:
                    V = self._calc_grav_potential(
                        state, k_sq, G, use_quantum, use_hydro
                    )
                    if use_external_potential:
                        V = V + V_ext
                else:
                    V = V_ext

                if use_quantum:
                    state["psi"] = quantum_kick(state["psi"], V, m_per_hbar, dt)
                if use_hydro:
                    ax, ay = get_acceleration(
                        V, kx, ky, dx, dy, x_has_ghosts, y_has_ghosts
                    )
                    W = hydro_euler2d_accelerate(
                        to_hydro(state), ax, ay, gamma, geom, dt
                    )
                    state = from_hydro(state, W)
                return state

            def drift(state, dt):
                state = dict(state)
                if use_quantum:
                    state["psi"] = quantum_drift(state["psi"], k_sq, m_per_hbar, dt)
                if use_hydro:
                    W = hydro_fluxes(
                        to_hydro(state),
                        geom_work,
                        dt,
                        gamma=gamma,
                        riemann_solver=riemann_solver,
                        slope_limiting=slope_limiting,
                        bc_x=bc_x,
                        bc_y=bc_y,
                        ghost_x=driven_ghosts(state, 0, bc_x),
                        ghost_y=driven_ghosts(state, 1, bc_y),
                    )
                    state = from_hydro(state, W)
                return state

            def step(state):
                dt = get_timestep(state) if use_adaptive_timesteps else dt_ref

                # kick-drift-kick
                state = kick(state, 0.5 * dt)
                state = drift(state, dt)
                state = kick(state, 0.5 * dt)

                return {
                    **state,
                    "t": state["t"] + dt,
                    "steps_taken": state["steps_taken"] + 1,
                }

            if use_adaptive_timesteps:
                return jax.lax.while_loop(
                    lambda state: state["t"] < t_target * (1.0 - 1e-10), step, state
                )
            state, _ = jax.lax.scan(
                lambda state, _: (step(state), None), state, xs=None, length=num_steps
            )
            return state

        return advance

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

        bcs = self.params["mesh"]["boundary_condition"]
        if "driven" in bcs and self.driven_boundary is None:
            raise ValueError(
                "a 'driven' boundary condition requires sim.driven_boundary"
            )

        nt = self.params["time"]["num_timesteps"]
        t_span = self.params["time"]["span"]
        save = self.params["output"]["save"]
        num_chunks = self.params["output"]["num_checkpoints"] if save else 1

        advance = self._get_advance()

        # Checkpointer
        if save:
            checkpoint_dir = os.path.join(os.getcwd(), self.params["output"]["path"])
            if jax.process_index() == 0:
                ocp.test_utils.erase_and_create_empty(checkpoint_dir)

        # save initial state
        if jax.process_index() == 0:
            print(f"Starting simulation (res={self.resolution}, nt={nt}) ...")
        if save:
            with open(os.path.join(checkpoint_dir, "params.json"), "w") as f:
                json.dump(self.params, f, indent=2)
            plot_sim(state, checkpoint_dir, 0, self.params)

        # Simulation Main Loop
        for i in range(1, num_chunks + 1):
            state = advance(state, jnp.asarray(t_span * i / num_chunks))
            if save:
                jax.block_until_ready(state)
                plot_sim(state, checkpoint_dir, i, self.params)

        return state

    def run(self):
        """
        Run the simulation
        """
        self.state["steps_taken"] = 0
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
