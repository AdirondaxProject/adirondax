import argparse
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import adirondax as adx
from adirondax.radiation.rt2d import radiation_moments

"""
Simulate two beams of photons crossing in vacuum (radiation transport)

Two beams are injected at the bottom of a box with no opacity, and travel
diagonally upwards. The box is periodic in x, so the beams wrap around and cross
each other again and again on their way to the top. Each beam carries on
undisturbed through the crossings, which a radiation fluid (M1) cannot do: it
would merge the two beams into one.

With no opacity the implicit solver has only the transport to iterate on, which
makes this the hardest case for its convergence.

Philip Mocz (2026)
"""


# Units: the box is 4 tall and light crosses it in 4e-3
C_LIGHT = 1.0e3
I_BEAM = 0.8  # intensity of the beams
X_BEAM = 0.1  # the beams enter at x = -X_BEAM and x = +X_BEAM


def set_up_simulation(resolution_multiplier=4, save=True):
    # Define the parameters for the simulation
    nx = 16 * resolution_multiplier
    ny = 64 * resolution_multiplier
    nt = 10 * resolution_multiplier

    # two light-crossing times of the box
    t_stop = 2.0 * 4.0 / C_LIGHT

    params = {
        "physics": {
            "hydro": True,
            "radiation": True,
        },
        "mesh": {
            "geometry": "cartesian",
            "origin": [-0.5, -2.0],
            "resolution": [nx, ny],
            "box_size": [1.0, 4.0],
            "boundary_condition": ["periodic", "driven"],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": nt,
            "save": save,
            "plot_dynamic_range": 1.0e3,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": 5.0 / 3.0},
        },
        "radiation": {
            "speed_of_light": C_LIGHT,
            "radiation_constant": 1.0,
            "gas_constant": 1.0,
            "num_polar": 2,
            "num_azimuthal": 2,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions: uniform gas at rest, which has no opacity and so
    # never sees the radiation, and no radiation
    sim.state["t"] = 0.0
    X, _ = sim.mesh
    sim.state["rho"] = jnp.ones(X.shape)
    sim.state["vx"] = jnp.zeros(X.shape)
    sim.state["vy"] = jnp.zeros(X.shape)
    sim.state["P"] = jnp.ones(X.shape)
    sim.state["I"] = jnp.zeros((1, len(sim.angles["w"]), nx, ny))

    # The beams enter through the bottom boundary, along mu_x = +-0.577 and
    # mu_y = 0.577: the right-moving beam at x = X_BEAM, and its mirror image,
    # the left-moving beam, at x = -X_BEAM. Nothing enters at the top.
    angles = sim.angles
    mu_x, mu_y = angles["mu_x"], angles["mu_y"]
    right = int(np.argmax((mu_x > 0.0) & (mu_y > 0.0)))
    left = int(angles["reflect_x"][right])
    i_right = int(np.argmin(np.abs(np.asarray(X[:, 0]) - X_BEAM)))
    i_left = nx - 1 - i_right
    sim.beam_slope = float(mu_x[right] / mu_y[right])

    I_in = jnp.zeros((1, len(angles["w"]), nx, 1))
    I_in = I_in.at[0, right, i_right].set(I_BEAM)
    I_in = I_in.at[0, left, i_left].set(I_BEAM)

    def driven_boundary(state, axis):
        # the gas flows out freely at the bottom and top
        ghosts = {
            k: (state[k][:, :1], state[k][:, -1:]) for k in ("rho", "vx", "vy", "P")
        }
        ghosts["I"] = (I_in, jnp.zeros_like(I_in))
        return ghosts

    sim.driven_boundary = driven_boundary

    return sim


def beam_paths(slope, y):
    """Where the two beams cross height y, wrapped into the periodic box"""
    shift = slope * (y + 2.0)
    right = X_BEAM + shift
    left = -X_BEAM - shift
    return (right + 0.5) % 1.0 - 0.5, (left + 0.5) % 1.0 - 0.5


def make_plot(sim):
    E = radiation_moments(sim.state["I"], sim.angles, C_LIGHT)["E"][0]

    plt.figure(figsize=(4, 8), dpi=80)
    plt.imshow(
        jnp.maximum(E, 1.0e-6).T,
        cmap="plasma",
        origin="lower",
        extent=[-0.5, 0.5, -2.0, 2.0],
        norm=LogNorm(vmin=0.05, vmax=1.9),
    )
    plt.colorbar(label="radiation energy density")

    # the beams travel in straight lines, wrapping around in x
    y = np.linspace(-2.0, 2.0, 4001)
    for x in beam_paths(sim.beam_slope, y):
        jumps = np.abs(np.diff(x)) > 0.5
        x = np.insert(x, np.where(jumps)[0] + 1, np.nan)
        y_line = np.insert(y, np.where(jumps)[0] + 1, np.nan)
        plt.plot(x, y_line, "w:", lw=0.8, alpha=0.6)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Crossing beams in vacuum")
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

    if args.plot:
        make_plot(sim)

    return sim


if __name__ == "__main__":
    main()
