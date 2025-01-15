# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import os
import jax
import jax.numpy as jnp
import adirondax as adx
from jaxopt import ScipyMinimize
import matplotlib.image as img


class InverseProblemSuite:
    """
    An example benchmark that times the performance of an inverse problem.
    """

    def setup(self):
        file_path = os.path.join(
            os.path.dirname(__file__), "../../examples/schrodinger_poisson/target.png"
        )
        target_data = img.imread(file_path)[:, :, 0]
        self.rho_target = jnp.flipud(jnp.array(target_data, dtype=float))
        self.rho_target = 1.0 - 0.5 * (self.rho_target - 0.5)
        self.rho_target /= jnp.mean(self.rho_target)

        n = 128
        nt = 100
        t_stop = 0.03
        dt = t_stop / nt
        self.params = {
            "physics": {
                "hydro": False,
                "magnetic": False,
                "quantum": True,
                "gravity": True,
            },
            "mesh": {
                "type": "cartesian",
                "resolution": [n, n],
                "boxsize": [1.0, 1.0],
            },
            "simulation": {
                "stop_time": t_stop,
                "timestep": dt,
                "n_timestep": nt,
            },
        }

    def run_forward_model(self):
        sim = adx.Simulation(self.params)
        xlin = jnp.linspace(0.0, 1.0, self.params["mesh"]["resolution"][0])
        x, y = jnp.meshgrid(xlin, xlin, indexing="ij")
        theta = -jnp.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2))
        sim.state["t"] = 0.0
        sim.state["psi"] = jnp.exp(1.0j * theta)
        sim.run()
        theta = jnp.angle(sim.state["psi"])
        return jnp.mean(theta)

    def solve_inverse_problem(self):

        assert self.rho_target.shape[0] == 128

        sim = adx.Simulation(self.params)

        @jax.jit
        def loss_function(theta, rho_target):
            sim.state["t"] = 0.0
            sim.state["psi"] = jnp.exp(1.0j * theta)
            sim.run()
            rho = jnp.abs(sim.state["psi"]) ** 2
            return jnp.mean((rho - rho_target) ** 2)

        opt = ScipyMinimize(method="l-bfgs-b", fun=loss_function, tol=1e-5)
        theta = jnp.zeros_like(self.rho_target)
        sol = opt.run(theta, self.rho_target)
        theta = jnp.mod(sol.params, 2.0 * jnp.pi) - jnp.pi

        return jnp.mean(theta)

    def time_forward_model(self):
        self.run_forward_model()

    def time_inverse_problem(self):
        self.solve_inverse_problem()

    def peakmem_forward_model(self):
        self.run_forward_model()

    def peakmem_inverse_problem(self):
        self.solve_inverse_problem()
