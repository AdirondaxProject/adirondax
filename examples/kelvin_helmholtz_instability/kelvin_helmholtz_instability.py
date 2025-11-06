import jax.numpy as jnp

# TODO: REMOVE THE FOLLOWING LINES
import sys

sys.path.append("../../")

import adirondax as adx
import time
import matplotlib.pyplot as plt

"""
Simulate the Kelvin-Helmholtz instability

Philip Mocz (2025)
"""


def set_up_simulation():
    # Define the parameters for the simulation
    n = 256
    nt = 1500 * int(n / 128)
    t_stop = 2.0

    params = {
        "physics": {
            "hydro": True,
        },
        "mesh": {
            "type": "cartesian",
            "resolution": [n, n],
            "box_size": [1.0, 1.0],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": 5.0 / 3.0},
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions
    # (opposite moving streams with perturbation)
    sim.state["t"] = 0.0
    X, Y = sim.mesh
    w0 = 0.1
    sigma = 0.05 / jnp.sqrt(2.0)
    sim.state["rho"] = 1.0 + (jnp.abs(Y - 0.5) < 0.25)
    sim.state["vx"] = -0.5 + (jnp.abs(Y - 0.5) < 0.25)
    sim.state["vy"] = (
        w0
        * jnp.sin(4.0 * jnp.pi * X)
        * (
            jnp.exp(-((Y - 0.25) ** 2) / (2.0 * sigma**2))
            + jnp.exp(-((Y - 0.75) ** 2) / (2.0 * sigma**2))
        )
    )
    sim.state["P"] = 2.5 * jnp.ones(X.shape)

    return sim


def make_plot(sim):
    # Plot the solution
    plt.figure(figsize=(6, 4), dpi=80)
    plt.imshow(jnp.rot90(sim.state["rho"]), cmap="jet", vmin=0.8, vmax=2.2)
    plt.colorbar(label="density")
    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():
    sim = set_up_simulation()

    # Evolve the system
    t0 = time.time()
    sim.run()
    print("Run time (s): ", time.time() - t0)

    make_plot(sim)


if __name__ == "__main__":
    main()
