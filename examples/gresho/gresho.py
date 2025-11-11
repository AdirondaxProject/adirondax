import jax.numpy as jnp

# TODO: REMOVE THE FOLLOWING LINES
import sys

sys.path.append("../../")

import adirondax as adx
import time
import matplotlib.pyplot as plt

"""
Simulate the Gresho Vortex

Philip Mocz (2025)
"""


def set_up_simulation():
    # Define the parameters for the simulation
    nx = 128
    nt = -1
    t_stop = 3.0

    params = {
        "physics": {
            "hydro": True,
        },
        "mesh": {
            "type": "cartesian",
            "resolution": [nx, nx],
            "box_size": [1.0, 1.0],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": 100,
            "save": False,
            "plot_dynamic_range": 2.0,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": 5.0 / 3.0},
            "slope_limiting": True,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions
    sim.state["t"] = 0.0
    X, Y = sim.mesh
    R = jnp.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)

    def v_phi(r):
        return jnp.where(
            r < 0.2,
            5.0 * r,
            jnp.where(r < 0.4, 2.0 - 5.0 * r, 0.0),
        )

    def P0(r):
        return jnp.where(
            r < 0.2,
            5.0 + 12.5 * r**2,
            jnp.where(
                r < 0.4,
                9.0 + 12.5 * r**2 - 20.0 * r + 4.0 * jnp.log(r / 0.2),
                3.0 + 4.0 * jnp.log(2.0),
            ),
        )

    vx = -v_phi(R) * (Y - 0.5) / (R + (R == 0))
    vy = v_phi(R) * (X - 0.5) / (R + (R == 0))
    sim.state["rho"] = jnp.ones(X.shape)
    sim.state["vx"] = vx
    sim.state["vy"] = vy
    sim.state["P"] = P0(R)

    return sim


def make_plot(sim):
    # Plot the solution
    plt.figure(figsize=(6, 4), dpi=80)
    v_phi = jnp.sqrt(sim.state["vx"] ** 2 + sim.state["vy"] ** 2)
    plt.imshow(v_phi.T, cmap="jet", vmin=0.0, vmax=1.0)
    plt.gca().invert_yaxis()
    plt.colorbar(label="v_phi")
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
