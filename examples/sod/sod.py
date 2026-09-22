import contextlib
import glob
import io
import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import adirondax as adx

"""
Simulate the Sod shock tube on a 2D cartesian mesh

Philip Mocz (2026)
"""

GAMMA = 1.4
X_DIAPHRAGM = 0.5
RHO_L, VX_L, P_L = 1.0, 0.0, 1.0
RHO_R, VX_R, P_R = 0.125, 0.0, 0.1


def set_up_simulation():
    # Define the parameters for the simulation
    nx = 400
    ny = 40
    nt = 2000
    t_stop = 0.2

    params = {
        "physics": {
            "hydro": True,
        },
        "mesh": {
            "geometry": "cartesian",
            "resolution": [nx, ny],
            "box_size": [1.0, 0.1],
            "boundary_condition": ["reflective", "periodic"],
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
            "eos": {"type": "ideal", "gamma": GAMMA},
            "riemann_solver": "hllc",
            "slope_limiting": True,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions
    sim.state["t"] = 0.0
    X, _ = sim.mesh
    left = X < X_DIAPHRAGM

    sim.state["rho"] = jnp.where(left, RHO_L, RHO_R)
    sim.state["vx"] = jnp.where(left, VX_L, VX_R)
    sim.state["vy"] = jnp.zeros(X.shape)
    sim.state["P"] = jnp.where(left, P_L, P_R)

    return sim


def exact_riemann(x, t, gamma=GAMMA, x0=X_DIAPHRAGM):
    """
    Exact solution of the Riemann problem for an ideal gas
    """

    a_l = np.sqrt(gamma * P_L / RHO_L)
    a_r = np.sqrt(gamma * P_R / RHO_R)
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)

    def f_and_df(p, rho_k, p_k, a_k):
        """Pressure function for one side, and its derivative"""
        if p > p_k:  # shock
            A = 2.0 / ((gamma + 1.0) * rho_k)
            B = (gamma - 1.0) / (gamma + 1.0) * p_k
            sq = np.sqrt(A / (B + p))
            return (p - p_k) * sq, sq * (1.0 - 0.5 * (p - p_k) / (B + p))
        # rarefaction
        return (
            2.0 * a_k / (gamma - 1.0) * ((p / p_k) ** g1 - 1.0),
            (p / p_k) ** (-g2) / (rho_k * a_k),
        )

    # two-rarefaction initial guess, then Newton
    du = VX_R - VX_L
    p = max(
        1.0e-12,
        ((a_l + a_r - 0.5 * (gamma - 1.0) * du) / (a_l / P_L**g1 + a_r / P_R**g1))
        ** (1.0 / g1),
    )
    for _ in range(50):
        fl, dfl = f_and_df(p, RHO_L, P_L, a_l)
        fr, dfr = f_and_df(p, RHO_R, P_R, a_r)
        dp = (fl + fr + du) / (dfl + dfr)
        p = max(1.0e-12, p - dp)
        if abs(dp) < 1.0e-14 * p:
            break
    p_star = p
    fl, _ = f_and_df(p_star, RHO_L, P_L, a_l)
    fr, _ = f_and_df(p_star, RHO_R, P_R, a_r)
    u_star = 0.5 * (VX_L + VX_R) + 0.5 * (fr - fl)

    # sample the self-similar solution at s = (x - x0)/t
    x = np.asarray(x)
    if t <= 0.0:
        left = x < x0
        return (
            np.where(left, RHO_L, RHO_R),
            np.where(left, VX_L, VX_R),
            np.where(left, P_L, P_R),
            p_star,
            u_star,
        )
    s = (x - x0) / t
    rho = np.empty_like(s)
    vx = np.empty_like(s)
    pr = np.empty_like(s)

    for side in ("L", "R"):
        if side == "L":
            sel = s <= u_star
            sgn, rho_k, u_k, p_k, a_k = -1.0, RHO_L, VX_L, P_L, a_l
        else:
            sel = s > u_star
            sgn, rho_k, u_k, p_k, a_k = 1.0, RHO_R, VX_R, P_R, a_r
        if not np.any(sel):
            continue
        sk = s[sel]

        if p_star > p_k:
            # shock: uniform star state up to the shock, ambient beyond
            rho_star = rho_k * (
                (p_star / p_k + (gamma - 1.0) / (gamma + 1.0))
                / ((gamma - 1.0) / (gamma + 1.0) * p_star / p_k + 1.0)
            )
            s_shock = u_k + sgn * a_k * np.sqrt(g2 * p_star / p_k + g1)
            inside = sk * sgn < s_shock * sgn
            rho[sel] = np.where(inside, rho_star, rho_k)
            vx[sel] = np.where(inside, u_star, u_k)
            pr[sel] = np.where(inside, p_star, p_k)
        else:
            # rarefaction fan between its head and tail
            rho_star = rho_k * (p_star / p_k) ** (1.0 / gamma)
            a_star = a_k * (p_star / p_k) ** g1
            s_head = u_k + sgn * a_k
            s_tail = u_star + sgn * a_star
            fan = 2.0 / (gamma + 1.0) - sgn * (gamma - 1.0) / ((gamma + 1.0) * a_k) * (
                u_k - sk
            )
            rho_fan = rho_k * fan ** (2.0 / (gamma - 1.0))
            u_fan = 2.0 / (gamma + 1.0) * (sgn * -a_k + (gamma - 1.0) / 2.0 * u_k + sk)
            p_fan = p_k * fan ** (1.0 / g1)

            # measured along sgn, the fan lies between its head and its tail
            in_fan = (sk * sgn < s_head * sgn) & (sk * sgn > s_tail * sgn)
            in_star = sk * sgn <= s_tail * sgn
            rho[sel] = np.where(in_star, rho_star, np.where(in_fan, rho_fan, rho_k))
            vx[sel] = np.where(in_star, u_star, np.where(in_fan, u_fan, u_k))
            pr[sel] = np.where(in_star, p_star, np.where(in_fan, p_fan, p_k))

    return rho, vx, pr, p_star, u_star


def draw_profiles(sim, axes):
    """
    Draw rho, vx and P against x on the three given axes.
    """
    X, _ = sim.mesh
    x = np.asarray(X).ravel()
    t = float(sim.state["t"])

    x_exact = np.linspace(0.0, 1.0, 2000)
    rho_e, vx_e, p_e, _, _ = exact_riemann(x_exact, t)

    fields = [
        ("rho", np.asarray(sim.state["rho"]).ravel(), rho_e, (0.0, 1.09)),
        ("vx", np.asarray(sim.state["vx"]).ravel(), vx_e, (-0.05, 1.05)),
        ("P", np.asarray(sim.state["P"]).ravel(), p_e, (0.0, 1.09)),
    ]

    for ax, (name, sim_vals, exact_vals, ylim) in zip(axes, fields):
        ax.scatter(
            x,
            sim_vals,
            s=1.0,
            color="tab:blue",
            alpha=0.3,
            linewidths=0,
            label="simulation",
        )
        ax.plot(x_exact, exact_vals, "k-", lw=1.2, label="exact")
        ax.set_ylabel(name)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower left", framealpha=1.0, markerscale=8)
    axes[0].set_title(f"Sod shock tube at t = {t:.3f}")
    axes[-1].set_xlabel("x")
    axes[-1].set_xlim(0.0, 1.0)


def make_plot(sim, filename="output.png", show=True):
    _, axes = plt.subplots(3, 1, figsize=(6, 8), dpi=80, sharex=True)
    draw_profiles(sim, axes)
    plt.tight_layout()
    plt.savefig(filename, dpi=240 if show else 80)
    if show:
        plt.show()
    plt.close()


def run_simulation(sim):
    """
    Evolve the simulation and create plots of the profile
    """
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

    # step forward in equal chunks, with the library's own checkpointing off
    sim.params["output"]["save"] = False
    sim.params["time"]["num_timesteps"] = nt_total // num_checkpoints
    sim.params["time"]["span"] = t_total / num_checkpoints

    make_plot(sim, os.path.join(checkpoint_dir, "sod000.png"), show=False)
    steps = 0
    for i in range(1, num_checkpoints + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run()
        steps += int(sim.steps_taken)
        make_plot(sim, os.path.join(checkpoint_dir, f"sod{i:03d}.png"), show=False)

    sim.params["output"]["save"] = True
    sim.params["time"]["num_timesteps"] = nt_total
    sim.params["time"]["span"] = t_total
    sim.state["steps_taken"] = steps


def main():
    sim = set_up_simulation()

    # Evolve the system
    t0 = time.time()
    run_simulation(sim)
    print("Run time (s): ", time.time() - t0)
    print("Steps taken:", sim.steps_taken)

    make_plot(sim)


if __name__ == "__main__":
    main()
