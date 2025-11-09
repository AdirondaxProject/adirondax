import jax.numpy as jnp

# TODO: REMOVE THE FOLLOWING LINES
import sys

sys.path.append("../../")

import adirondax as adx
from adirondax.hydro.common2d import get_curl, get_avg
import time
import matplotlib.pyplot as plt

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Simulate the Orszag-Tang vortex

Philip Mocz (2025)
"""


def set_up_simulation():
    # Define the parameters for the simulation
    n = 512
    t_stop = 0.5
    gamma = 5.0 / 3.0
    box_size = 1.0
    dx = box_size / n

    params = {
        "physics": {
            "hydro": True,
            "magnetic": True,
        },
        "mesh": {
            "type": "cartesian",
            "resolution": [n, n],
            "box_size": [box_size, box_size],
        },
        "time": {
            "span": t_stop,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": gamma},
            "cfl": 0.6,
            "riemann_solver": "hlld",
            "slope_limiting": True,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions
    sim.state["t"] = jnp.array(0.0)
    X, Y = sim.mesh
    sim.state["rho"] = (gamma**2 / (4.0 * jnp.pi)) * jnp.ones(X.shape)
    sim.state["vx"] = -jnp.sin(2.0 * jnp.pi * Y)
    sim.state["vy"] = jnp.sin(2.0 * jnp.pi * X)
    P_gas = (gamma / (4.0 * jnp.pi)) * jnp.ones(X.shape)
    # (Az is at top-right node of each cell)
    xlin_node = jnp.linspace(dx, box_size, n)
    Xn, Yn = jnp.meshgrid(xlin_node, xlin_node, indexing="ij")
    Az = jnp.cos(4.0 * jnp.pi * Xn) / (4.0 * jnp.pi * jnp.sqrt(4.0 * jnp.pi)) + jnp.cos(
        2.0 * jnp.pi * Yn
    ) / (2.0 * jnp.pi * jnp.sqrt(4.0 * jnp.pi))
    bx, by = get_curl(Az, dx)
    Bx, By = get_avg(bx, by)
    P_tot = P_gas + 0.5 * (Bx**2 + By**2)
    sim.state["P"] = P_tot
    sim.state["bx"] = bx
    sim.state["by"] = by

    return sim


def make_plot(sim):
    # Plot the solution
    plt.figure(figsize=(6, 4), dpi=80)
    plt.imshow(jnp.rot90(sim.state["rho"]), cmap="jet", vmin=0.06, vmax=0.5)
    plt.colorbar(label="density")
    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():
    sim = set_up_simulation()

    # Evolve the system
    t0 = time.time()
    sim.run()
    print("Steps taken:", sim.steps_taken)
    print("Run time (s):", time.time() - t0)

    make_plot(sim)


if __name__ == "__main__":
    main()
