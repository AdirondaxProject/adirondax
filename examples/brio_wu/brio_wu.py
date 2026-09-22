import argparse
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import adirondax as adx
from adirondax.hydro.common2d import get_avg

"""
Simulate the Brio-Wu shock tube on a 2D cartesian mesh

The MHD analogue of Sod's problem.

Philip Mocz (2026)
"""

# Brio & Wu (1988), table 1
GAMMA = 2.0
X_INTERFACE = 0.5
BX = 0.75
RHO_L, P_GAS_L, BY_L = 1.000, 1.0, 1.0
RHO_R, P_GAS_R, BY_R = 0.125, 0.1, -1.0


def set_up_simulation(resolution_multiplier=4, save=True):
    # Define the parameters for the simulation
    nx = 100 * resolution_multiplier
    ny = 10 * resolution_multiplier
    nt = 200 * resolution_multiplier
    t_stop = 0.1

    params = {
        "physics": {
            "hydro": True,
            "magnetic": True,
        },
        "mesh": {
            "geometry": "cartesian",
            "resolution": [nx, ny],
            "box_size": [1.0, 0.1],
            "boundary_condition": ["outflow", "periodic"],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": 100,
            "save": save,
            "plot_dynamic_range": 10.0,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": GAMMA},
            "riemann_solver": "hlld",
            "slope_limiting": True,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions
    sim.state["t"] = jnp.array(0.0)
    X, _ = sim.mesh
    left = X < X_INTERFACE

    # Bx is uniform and By varies only with x, so the field is divergence-free
    # by construction and no vector potential is needed
    bx = BX * jnp.ones(X.shape)
    by = jnp.where(left, BY_L, BY_R)
    Bx, By = get_avg(bx, by)

    sim.state["rho"] = jnp.where(left, RHO_L, RHO_R)
    sim.state["vx"] = jnp.zeros(X.shape)
    sim.state["vy"] = jnp.zeros(X.shape)
    sim.state["bx"] = bx
    sim.state["by"] = by
    # the state carries the total (gas plus magnetic) pressure
    sim.state["P"] = jnp.where(left, P_GAS_L, P_GAS_R) + 0.5 * (Bx**2 + By**2)

    return sim


def make_plot(sim):
    # Every cell of the 2D mesh is scattered against its x coordinate, so a
    # solution that has stayed one-dimensional collapses onto a single curve.
    X, _ = sim.mesh
    x = np.asarray(X).ravel()
    t = float(sim.state["t"])

    Bx, By = get_avg(sim.state["bx"], sim.state["by"])
    P_gas = sim.state["P"] - 0.5 * (Bx**2 + By**2)

    fields = [
        ("rho", sim.state["rho"]),
        ("vx", sim.state["vx"]),
        ("vy", sim.state["vy"]),
        ("By", By),
        ("P_gas", P_gas),
    ]

    _, axes = plt.subplots(5, 1, figsize=(6, 11), dpi=80, sharex=True)
    for ax, (name, values) in zip(axes, fields):
        ax.scatter(
            x,
            np.asarray(values).ravel(),
            s=1.0,
            color="tab:blue",
            alpha=0.3,
            linewidths=0,
        )
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)

    axes[0].set_title(f"Brio-Wu shock tube at t = {t:.2f}")
    axes[-1].set_xlabel("x")
    axes[-1].set_xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=4, help="resolution multiplier")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write checkpoints",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make the summary plot",
    )
    args = parser.parse_args()

    sim = set_up_simulation(args.res, args.save)

    # Evolve the system
    t0 = time.time()
    sim.run()
    print("Run time (s): ", time.time() - t0)
    print("Steps taken:", sim.steps_taken)

    if args.plot:
        make_plot(sim)

    return sim


if __name__ == "__main__":
    main()
