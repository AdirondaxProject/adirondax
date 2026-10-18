import argparse
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import adirondax as adx
from adirondax.radiation.rt2d import radiation_moments

"""
Simulate the shadow cast by an optically thick cloud (radiation transport)

Two beams of photons enter from the left and cross a box of optically thin gas
that holds a dense, cold cloud. The cloud absorbs the beams and casts a sharp
shadow: an umbra where it blocks both beams, and a penumbra on either side where
it blocks one. Radiation that diffuses or behaves as a fluid (flux-limited
diffusion, M1) cannot hold such a shadow; discrete ordinates can.

Philip Mocz (2026)
"""


# Units: k_B = m_p = 1, so T = P / rho, and
# the speed of light is c = 1.9e5 times the sound speed
C_LIGHT = 1.9e5
A_RAD = 1.0e-7  # the radiation constant: radiation is dynamically negligible
E_BEAM = 1.0e-4  # radiation energy density of the incoming beams
P_GAS = 1.0
GAMMA = 5.0 / 3.0


def cloud_density(x, y):
    """A smooth ellipsoidal cloud, 10x denser than its surroundings"""
    return 1.0 + 9.0 / (1.0 + jnp.exp(10.0 * ((x / 0.1) ** 2 + (y / 0.06) ** 2 - 1.0)))


def opacity(rho, T):
    """Pure absorption, kappa = rho T^-3.5, so the cloud is optically thick"""
    kappa = rho * T**-3.5
    return {"sigma_s": 0.0, "sigma_a": rho * kappa, "sigma_p": rho * kappa}


def set_up_simulation(resolution_multiplier=4, save=True):
    # Define the parameters for the simulation
    nx = 80 * resolution_multiplier
    ny = 48 * resolution_multiplier
    nt = 60 * resolution_multiplier

    # three light-crossing times of the box
    t_stop = 3.0 / C_LIGHT

    params = {
        "physics": {
            "hydro": True,
            "radiation": True,
        },
        "mesh": {
            "geometry": "cartesian",
            "origin": [-0.5, -0.3],
            "resolution": [nx, ny],
            "box_size": [1.0, 0.6],
            "boundary_condition": ["driven", "periodic"],
        },
        "constants": {
            "speed_of_light": C_LIGHT,
            "radiation_constant": A_RAD,
            "boltzmann_constant": 1.0,
            "proton_mass": 1.0,
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": 60,
            "save": save,
            "plot_dynamic_range": 1.0e4,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": GAMMA},
            "riemann_solver": "hllc",
            "slope_limiting": True,
        },
        "radiation": {
            "num_polar": 6,
            "num_azimuthal": 8,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)
    sim.opacity = opacity

    # Set initial conditions: a cloud in pressure equilibrium with its
    # surroundings, and radiation in equilibrium with the gas
    sim.state["t"] = 0.0
    X, Y = sim.mesh
    rho = cloud_density(X, Y)
    sim.state["rho"] = rho
    sim.state["vx"] = jnp.zeros(X.shape)
    sim.state["vy"] = jnp.zeros(X.shape)
    sim.state["P"] = P_GAS * jnp.ones(X.shape)
    sim.state["I"] = sim.blackbody_intensity(P_GAS / rho)

    # The two beams enter through the left boundary, along the directions
    # closest to +-15 degrees from the horizontal; nothing enters on the right
    angles = sim.angles
    mu_x, mu_y = angles["mu_x"], angles["mu_y"]
    up = int(np.argmax(np.where(mu_y > 0.0, mu_x, -1.0)))
    down = int(angles["reflect_y"][up])
    sim.beam_angle = float(np.degrees(np.arctan2(mu_y[up], mu_x[up])))

    I_beam = E_BEAM / (4.0 * np.pi * (angles["w"][up] + angles["w"][down]))
    I_in = jnp.zeros((1, len(angles["w"]), 1, ny))
    I_in = I_in.at[0, [up, down]].set(I_beam)

    def driven_boundary(state, axis):
        # the gas flows out freely at the left and right
        ghosts = {
            k: (state[k][:1, :], state[k][-1:, :]) for k in ("rho", "vx", "vy", "P")
        }
        ghosts["I"] = (I_in, jnp.zeros_like(I_in))
        return ghosts

    sim.driven_boundary = driven_boundary

    return sim


def make_plot(sim):
    # Plot the radiation energy density and the Eddington factor f_xx = P_xx/E
    X, Y = sim.mesh
    angles = sim.angles
    I = sim.state["I"]
    E = radiation_moments(I, angles, C_LIGHT)["E"][0]
    w_xx = jnp.asarray(angles["w"] * angles["mu_x"] ** 2)
    f_xx = 4.0 * jnp.pi * jnp.einsum("n,nij->ij", w_xx, I[0]) / E

    extent = [-0.5, 0.5, -0.3, 0.3]
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), dpi=80)

    ax = axes[0]
    im = ax.imshow(
        jnp.log10(E / E_BEAM).T,
        cmap="inferno",
        origin="lower",
        extent=extent,
        vmin=-4.0,
        vmax=0.5,
    )
    fig.colorbar(im, ax=ax, label="log10 (E_r / E_beam)")

    ax = axes[1]
    im = ax.imshow(
        f_xx.T, cmap="viridis", origin="lower", extent=extent, vmin=0.0, vmax=1.0
    )
    fig.colorbar(im, ax=ax, label="Eddington factor f_xx")

    for ax in axes:
        ax.contour(X, Y, sim.state["rho"], levels=[5.5], colors="w", linewidths=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    axes[0].set_title(
        f"Beams at +-{sim.beam_angle:.0f} degrees, "
        f"after {float(sim.state['t']) * C_LIGHT:.0f} light-crossing times"
    )

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
