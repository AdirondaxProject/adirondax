import argparse
import contextlib
import glob
import io
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import adirondax as adx

jax.config.update("jax_enable_x64", True)

"""
Simulate a Alfven wave on an axisymmetric cylindrical (R,z) mesh

Philip Mocz (2026)
"""

RHO = 1.0
B_Z = 1.0
P_GAS = 1.0
AMPLITUDE = 0.01
GAMMA = 5.0 / 3.0


def set_up_simulation(resolution_multiplier=4, save=True):
    # Define the parameters for the simulation
    nR = 8 * resolution_multiplier
    nz = 32 * resolution_multiplier
    nt = 600 * resolution_multiplier  # a multiple of num_checkpoints
    t_stop = 1.0  # one full period, so the wave returns to where it started

    params = {
        "physics": {
            "hydro": True,
            "magnetic": True,
            "rotation": True,
        },
        "mesh": {
            "geometry": "cylindrical",
            "resolution": [nR, nz],
            "box_size": [0.5, 1.0],
            "boundary_condition": ["axis", "periodic"],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": 100,
            "save": save,
            "plot_dynamic_range": 2.0,
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
    R, z = sim.mesh
    b = AMPLITUDE * jnp.sin(2.0 * jnp.pi * z)

    sim.state["rho"] = RHO * jnp.ones(R.shape)
    sim.state["vx"] = jnp.zeros(R.shape)  # v_R
    sim.state["vy"] = jnp.zeros(R.shape)  # v_z
    sim.state["bx"] = jnp.zeros(R.shape)  # b_R
    sim.state["by"] = B_Z * jnp.ones(R.shape)  # b_z
    # a wave travelling towards +z has v_phi = -B_phi / sqrt(rho)
    sim.state["bphi"] = b * R
    sim.state["vphi"] = -b * R / jnp.sqrt(RHO)
    # the state carries the total (gas plus magnetic) pressure
    sim.state["P"] = P_GAS + 0.5 * (B_Z**2 + (b * R) ** 2)

    return sim


def exact_bphi(R, z, t):
    """The wave travels along z at the Alfven speed without changing shape"""
    v_a = B_Z / np.sqrt(RHO)
    return AMPLITUDE * np.sin(2.0 * np.pi * (z - v_a * t)) * R


def make_plot(sim):
    R, z = sim.mesh
    t = float(sim.state["t"])
    R = np.asarray(R)
    z = np.asarray(z)
    bphi = np.asarray(sim.state["bphi"])

    # the twist is proportional to R, so scaling it out collapses every radius
    # onto a single curve
    z_line = z[0, :]
    exact = exact_bphi(1.0, z_line, t)

    _, axes = plt.subplots(2, 1, figsize=(6, 7), dpi=80)

    axes[0].scatter(
        z.ravel(),
        (bphi / R).ravel(),
        s=2.0,
        color="tab:blue",
        alpha=0.3,
        linewidths=0,
        label="simulation",
    )
    axes[0].plot(z_line, exact, "k-", lw=1.2, label="exact")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("B_phi / R")
    axes[0].set_title(f"Alfven wave at t = {t:.2f}")
    axes[0].legend(loc="upper right", framealpha=1.0, markerscale=4)

    im = axes[1].imshow(
        bphi.T,
        cmap="RdBu_r",
        origin="lower",
        extent=[0.0, sim.box_size[0], 0.0, sim.box_size[1]],
        vmin=-AMPLITUDE * sim.box_size[0],
        vmax=AMPLITUDE * sim.box_size[0],
        aspect="auto",
    )
    plt.colorbar(im, ax=axes[1], label="B_phi")
    axes[1].set_xlabel("R")
    axes[1].set_ylabel("z")

    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def plot_bphi(sim, filename):
    bphi = np.asarray(sim.state["bphi"])
    bphi_full = np.concatenate((-np.flip(bphi, axis=0), bphi), axis=0)
    nx, ny = bphi_full.shape
    lim = AMPLITUDE * sim.box_size[0]

    plt.clf()
    ax = plt.gca()
    ax.imshow(
        bphi_full.T,
        cmap="RdBu_r",
        origin="lower",
        vmin=-lim,
        vmax=lim,
        extent=[0, nx, 0, ny],
    )
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def run_simulation(sim):
    if not sim.params["output"]["save"]:
        sim.run()
        return

    checkpoint_dir = sim.params["output"]["path"]
    num_checkpoints = sim.params["output"]["num_checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(checkpoint_dir, "*.png")):
        os.remove(stale)

    nt_total = sim.params["time"]["num_timesteps"]
    t_total = sim.params["time"]["span"]

    sim.params["output"]["save"] = False
    sim.params["time"]["num_timesteps"] = nt_total // num_checkpoints
    sim.params["time"]["span"] = t_total / num_checkpoints

    plot_bphi(sim, os.path.join(checkpoint_dir, "bphi000.png"))
    steps = 0
    for i in range(1, num_checkpoints + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run()
        steps += int(sim.steps_taken)
        plot_bphi(sim, os.path.join(checkpoint_dir, f"bphi{i:03d}.png"))

    sim.params["output"]["save"] = True
    sim.params["time"]["num_timesteps"] = nt_total
    sim.params["time"]["span"] = t_total
    sim.state["steps_taken"] = steps


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
    run_simulation(sim)
    print("Run time (s): ", time.time() - t0)
    print("Steps taken:", sim.steps_taken)

    R, z = sim.mesh
    err = jnp.mean(jnp.abs(sim.state["bphi"] - exact_bphi(R, z, float(sim.state["t"]))))
    print("L1 error in B_phi after one period:", float(err))
    print(
        "max |v_R| (should stay at the O(amplitude^2) level):",
        float(jnp.max(jnp.abs(sim.state["vx"]))),
    )

    if args.plot:
        make_plot(sim)

    return sim


if __name__ == "__main__":
    main()
