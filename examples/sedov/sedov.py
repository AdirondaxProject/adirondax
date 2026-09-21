import time

import jax.numpy as jnp
import matplotlib.pyplot as plt

import adirondax as adx

"""
Simulate a Sedov-Taylor blast wave on an axisymmetric cylindrical (R,z) mesh

The explosion is spherical, so the solution must remain spherical even though
the mesh is not: this is a direct check of the geometric source term and of the
treatment of the R=0 axis.

Philip Mocz (2026)
"""


def set_up_simulation():
    # Define the parameters for the simulation
    nR = 128
    nz = 256
    nt = 400
    t_stop = 0.05

    params = {
        "physics": {
            "hydro": True,
        },
        "mesh": {
            "geometry": "cylindrical",
            "resolution": [nR, nz],
            "box_size": [0.5, 1.0],
            "boundary_condition": ["axis", "reflective"],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": 100,
            "save": True,
            "plot_dynamic_range": 10.0,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": 5.0 / 3.0},
            "slope_limiting": True,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions: a hot sphere at the origin of the (R,z) plane
    sim.state["t"] = 0.0
    R, z = sim.mesh
    r_sph = jnp.sqrt(R**2 + (z - 0.5) ** 2)

    sim.state["rho"] = jnp.ones(R.shape)
    sim.state["vx"] = jnp.zeros(R.shape)
    sim.state["vy"] = jnp.zeros(R.shape)
    sim.state["P"] = jnp.where(r_sph < 0.05, 100.0, 0.1)

    return sim


def make_plot(sim):
    # Plot the solution, mirrored about the axis
    rho = sim.state["rho"]
    LR = sim.box_size[0]
    Lz = sim.box_size[1]

    rho_full = jnp.concatenate((jnp.flip(rho, axis=0), rho), axis=0)

    plt.figure(figsize=(5, 5), dpi=80)
    plt.imshow(
        rho_full.T,
        cmap="viridis",
        origin="lower",
        extent=[-LR, LR, 0.0, Lz],
    )
    plt.colorbar(label="rho")
    plt.xlabel("R")
    plt.ylabel("z")
    plt.gca().set_aspect("equal")
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
