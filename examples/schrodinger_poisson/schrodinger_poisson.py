import jax
import jax.numpy as jnp

# TODO: REMOVE THE FOLLOWING LINES
import sys

sys.path.append("../../")

import adirondax as adx
from jaxopt import ScipyMinimize
import time
import matplotlib.pyplot as plt
import matplotlib.image as img

"""
Solve an Inverse-Problem that finds the initial wave function phases that
evolve under the Schrodinger-Poisson equations into a target density field

Philip Mocz (2024)
"""


def setup_simulation():

    # Define the parameters for the simulation
    n = 128
    nt = 100 * int(n / 128)
    t_stop = 0.03
    dt = t_stop / nt

    params = {
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

    # Initialize the simulation
    sim = adx.Simulation(params)

    return sim


def solve_inverse_problem(sim):

    # Load the target density field
    target_data = img.imread("target.png")[:, :, 0]
    rho_target = jnp.flipud(jnp.array(target_data, dtype=float))
    rho_target = 1.0 - 0.5 * (rho_target - 0.5)
    rho_target /= jnp.mean(rho_target)

    assert rho_target.shape[0] == sim.params["mesh"]["resolution"][0]
    assert rho_target.shape[1] == sim.params["mesh"]["resolution"][1]

    # Define the loss function for the optimization
    @jax.jit
    def loss_function(theta, rho_target):
        sim.state["t"] = 0.0
        sim.state["psi"] = jnp.exp(1.0j * theta)
        sim.run()
        psi = sim.state["psi"]
        rho = jnp.abs(psi) ** 2
        return jnp.mean((rho - rho_target) ** 2)

    # Solve the inverse-problem (takes around 5 seconds on my macbook)
    opt = ScipyMinimize(
        method="l-bfgs-b", fun=loss_function, tol=1e-5, options={"disp": True}
    )
    theta = jnp.zeros_like(rho_target)
    t0 = time.time()
    sol = opt.run(theta, rho_target)
    print("Inverse-problem solve time (s): ", time.time() - t0)
    theta = jnp.mod(sol.params, 2.0 * jnp.pi) - jnp.pi
    print("Mean theta:", jnp.mean(theta))

    return theta


def rerun_simulation(sim, theta):

    # Re-run the simulation with the optimal initial conditions
    sim.state["t"] = 0.0
    sim.state["psi"] = jnp.exp(1.0j * theta)
    sim.run()
    print("Final time:", sim.state["t"])

    return sim.state["psi"]


def make_plot(psi, theta):

    # Plot the solution
    plt.figure(figsize=(6, 4), dpi=80)
    grid = plt.GridSpec(1, 2, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    plt.sca(ax1)
    plt.cla()
    plt.imshow(theta, cmap="bwr")
    plt.clim(-jnp.pi, jnp.pi)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.invert_yaxis()
    ax1.set_aspect("equal")
    plt.title(r"${\rm initial\,angle}(\psi)$")
    plt.sca(ax2)
    plt.cla()
    plt.imshow(jnp.log10(jnp.abs(psi) ** 2), cmap="inferno")
    plt.clim(-0.2, 0.2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    plt.title(r"${\rm final\,}\log_{10}(|\psi|^2)$")
    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():

    sim = setup_simulation()
    theta = solve_inverse_problem(sim)
    psi = rerun_simulation(sim, theta)
    make_plot(psi, theta)


if __name__ == "__main__":
    main()
