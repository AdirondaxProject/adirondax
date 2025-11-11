import jax.numpy as jnp

# TODO: REMOVE THE FOLLOWING LINES
import sys

sys.path.append("../../")

import adirondax as adx
import time
import matplotlib.pyplot as plt

"""
Simulate the Rayleigh-Taylor Instability

Philip Mocz (2025)
"""


def set_up_simulation():
    # Define the parameters for the simulation
    nx = 64
    ny = 192
    nt = 13000  #  -1
    t_stop = 20.0

    params = {
        "physics": {
            "hydro": True,
            "external_potential": True,
        },
        "mesh": {
            "type": "cartesian",
            "resolution": [nx, ny],
            "box_size": [0.5, 1.5],
            "boundary_condition": ["periodic", "reflective"],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": 100,
            "save": True,
            "plot_dynamic_range": 2.0,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": 1.4},
            "slope_limiting": False,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions
    # (heavy fluid on top of light)
    sim.state["t"] = 0.0
    X, Y = sim.mesh
    w0 = 0.0025
    P0 = 2.5
    g = 0.1
    sim.state["rho"] = 1.0 + (Y > 0.75)
    sim.state["vx"] = jnp.zeros(X.shape)
    sim.state["vy"] = (
        w0 * (1.0 - jnp.cos(4.0 * jnp.pi * X)) * (1.0 - jnp.cos(4.0 * jnp.pi * Y / 3.0))
    )
    sim.state["P"] = P0 - g * (Y - 0.75) * sim.state["rho"]

    # external potential
    def external_potential(x, y):
        V = g * y
        return V

    sim.external_potential = external_potential

    return sim


def make_plot(sim):
    # Plot the solution
    plt.figure(figsize=(4, 6), dpi=80)
    plt.imshow(sim.state["rho"].T, cmap="jet", vmin=0.8, vmax=2.2)
    plt.gca().invert_yaxis()
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
    print("Steps taken:", sim.steps_taken)

    make_plot(sim)


if __name__ == "__main__":
    main()
