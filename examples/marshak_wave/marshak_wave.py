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
from scipy.integrate import solve_ivp
from scipy.sparse import diags

import adirondax as adx
from adirondax.constants import code_constants
from adirondax.hydro.geometry import get_geometry
from adirondax.radiation.fld2d import radiation_fld2d_step

# the steep opacity makes the implicit radiation solve stiff, which needs double
# precision
jax.config.update("jax_enable_x64", True)

"""
Simulate a cylindrical Marshak wave (radiation diffusion)

A hot inner region at radius r0, driven to a temperature that ramps from 1.2 to
2 HeV, launches a radiation heat wave outwards into cold, opaque foam. The
smaller r0, the more the wave slows as it spreads out. This is the cylindrical
test problem of Smith, Bursch & McClarren (2025), in C6H12 foam, which we check
against a numerical solution of their diffusion equation and against their
semi-analytic model of the heat front.

The problem is static: the foam only exchanges energy with the radiation.

Philip Mocz (2026)
"""


# Code units: cm, ns, g and HeV (100 eV)
UNITS = {"length": 1.0, "time": 1.0e-9, "mass": 1.0, "temperature": 1.160452e6}
ERG = 1.0e-18  # the code unit of energy is g cm^2 / ns^2 = 1e18 erg

# C6H12 foam (Table I of the paper): the diffusion equation is
#   d(T^beta)/dt = C r^-1 d/dr (r d/dr T^(4+alpha)),  beta = eps (4 + alpha) = 1
C, EPS, ALPHA, RHO = 9.998e-4, 0.1433, 2.98, 0.05
GAMMA = 5.0 / 3.0

# The paper's equation holds the radiation energy to be negligible, and the
# radiation and matter to share one temperature. We give the foam a specific
# heat of 100 MJ/(g HeV), ten times that of a fully ionized CH plasma, so that
# the radiation holds well under 1% of the energy, and a Planck opacity a
# hundred times the Rosseland. (With the CH value, the radiation's own heat
# capacity slows the wave by a few percent.) The Rosseland opacity is then set
# by C: sigma = s T^-alpha.
SPECIFIC_HEAT = 1.0e14 * ERG  # per g per HeV
PLANCK_OVER_ROSSELAND = 100.0
T_COLD = 0.01  # HeV, the initial temperature of the foam

R0S = [1000.0, 1.0, 0.1, 0.01]  # cm, the inner radii
L = 0.6  # cm, the depth of foam simulated
N_DRIVE = 4  # cells inside r0 held at the drive temperature (2 if 4 do not fit)
T_END = 3.0  # ns
DT = 1.0e-3  # ns
N_OUT = 100
T_FRONT = 0.1  # HeV: the heat front is where the temperature falls below this


def drive_temperature(t):
    """1.2 HeV until 1 ns, ramping to 2 HeV at 2 ns (Figure 2 of the paper)"""
    return jnp.where(t < 1.0, 1.2, jnp.where(t < 2.0, 1.2 + 0.8 * (t - 1.0), 2.0))


def set_up_simulation(r0, resolution_multiplier=2):
    # Define the parameters for the simulation
    n = 150 * resolution_multiplier
    dr = L / n
    # as many drive cells as fit inside r0 (an even number, as the resolution
    # must be even)
    n_drive = N_DRIVE if r0 >= (N_DRIVE - 0.5) * dr else 2
    nR = n + n_drive

    # the gas constant k_B / (mu m_p) that gives the specific heat
    c = code_constants(UNITS)
    mu = c["boltzmann_constant"] / (c["proton_mass"] * SPECIFIC_HEAT * (GAMMA - 1.0))

    params = {
        "physics": {
            "radiation": True,
        },
        "mesh": {
            "geometry": "cylindrical",
            # the last of the drive cells is centered on r0
            "origin": [r0 - (n_drive - 0.5) * dr, 0.0],
            "resolution": [nR, 2],
            "box_size": [nR * dr, 2.0 * dr],
            "boundary_condition": ["reflective", "periodic"],
        },
        "units": UNITS,
        "time": {
            "span": T_END,
            "num_timesteps": round(T_END / DT),
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": GAMMA, "mean_molecular_weight": mu},
        },
        "radiation": {
            "method": "diffusion",
            "flux_limiter": "none",
            "face_average": "arithmetic",
            "tol": 1.0e-8,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)
    sim.r0 = r0
    sim.n_drive = n_drive

    # the Rosseland opacity that gives the paper's C (in the diffusion limit,
    # c / (3 sigma) grad(a T^4) = C rho c_v grad(T^(4 + alpha)))
    a = sim.constants["radiation_constant"]
    s = 4.0 * sim.constants["speed_of_light"] * a
    s /= 3.0 * (4.0 + ALPHA) * RHO * SPECIFIC_HEAT * C

    # the drive cells are a reservoir: of huge heat capacity, and tightly
    # coupled to the radiation
    drive = np.zeros((nR, 2), bool)
    drive[:n_drive] = True
    sim.drive = jnp.asarray(drive)

    def opacity(rho, T):
        sigma = s * jnp.maximum(T, T_COLD) ** -ALPHA
        sigma_p = jnp.where(sim.drive, 1.0e8, PLANCK_OVER_ROSSELAND * sigma)
        return {"sigma_s": 0.0, "sigma_a": sigma, "sigma_p": sigma_p}

    sim.opacity = opacity

    # Set initial conditions: cold foam, in equilibrium with the radiation
    R, _ = sim.mesh
    rho = jnp.where(sim.drive, 1.0e6, RHO)
    sim.state["t"] = 0.0
    sim.state["rho"] = rho
    sim.state["vx"] = jnp.zeros(R.shape)
    sim.state["vy"] = jnp.zeros(R.shape)
    sim.state["P"] = rho * sim.gas_constant * T_COLD
    sim.state["Er"] = sim.blackbody_energy(T_COLD * jnp.ones(R.shape))

    return sim


def temperature(sim, P):
    return P / (sim.state["rho"] * sim.gas_constant)


def run(sim):
    """
    Evolve the simulation, holding the drive cells at the drive temperature,
    and record the temperature profile at N_OUT times
    """

    rad = sim.params["radiation"]
    c = sim.constants
    geom = get_geometry(
        "cylindrical", sim.box_size, sim.resolution, r_min=sim.origin[0]
    )
    rho, drive = sim.state["rho"], sim.drive
    W0 = {k: sim.state[k] for k in ("rho", "vx", "vy")}
    gas_constant = sim.gas_constant
    a = c["radiation_constant"]

    def impose(P, E, t):
        T = drive_temperature(t)
        P = jnp.where(drive, rho * gas_constant * T, P)
        E = jnp.where(drive[None], a * T**4, E)
        return P, E

    def step(carry, _):
        P, E, t = carry
        W, E = radiation_fld2d_step(
            {**W0, "P": P},
            E,
            geom,
            DT,
            opacity=sim.opacity(rho, P / (rho * gas_constant)),
            group_edges=(),
            c=c["speed_of_light"],
            a_r=a,
            gas_constant=gas_constant,
            gamma=GAMMA,
            bc_x="reflective",
            bc_y="periodic",
            flux_limiter=rad["flux_limiter"],
            face_average=rad["face_average"],
            tol=rad["tol"],
            static=True,
        )
        P, E = impose(W["P"], E, t + DT)
        return (P, E, t + DT), None

    steps_per_output = round(T_END / DT / N_OUT)

    @jax.jit
    def advance(carry):
        return jax.lax.scan(step, carry, None, length=steps_per_output)[0]

    P, E = impose(sim.state["P"], sim.state["Er"], 0.0)
    carry = (P, E, 0.0)
    times, profiles = [0.0], [np.asarray(temperature(sim, P)[:, 0])]
    for _ in range(N_OUT):
        carry = advance(carry)
        times.append(float(carry[2]))
        profiles.append(np.asarray(temperature(sim, carry[0])[:, 0]))
    sim.state["P"], sim.state["Er"], sim.state["t"] = carry

    R, _ = sim.mesh
    return {
        "r": np.asarray(R[sim.n_drive :, 0]),
        "times": np.array(times),
        "T": np.array(profiles)[:, sim.n_drive :],
    }


def front(r, T):
    """The radius beyond which the temperature is below T_FRONT"""
    hot = np.where(T > T_FRONT)[0]
    return r[hot.max()] if len(hot) else r[0]


def reference_diffusion(r0, times, n=600):
    """
    The paper's diffusion equation, solved by finite volumes and backward
    differentiation (as in their Section III)
    """
    beta = EPS * (4.0 + ALPHA)
    edges = r0 + np.linspace(0.0, L, n + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    dr = L / n
    vol = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)

    def rhs(t, T):
        T = np.maximum(T, 1.0e-3)
        w = T ** (4.0 + ALPHA)
        flux = np.zeros(n + 1)
        flux[1:-1] = -C * (w[1:] - w[:-1]) / dr
        flux[0] = (
            -C * (w[0] - float(drive_temperature(t)) ** (4.0 + ALPHA)) / (0.5 * dr)
        )
        dTb = -(edges[1:] * flux[1:] - edges[:-1] * flux[:-1]) / vol
        return dTb / (beta * T ** (beta - 1.0))

    sol = solve_ivp(
        rhs,
        (0.0, T_END),
        1.0e-3 * np.ones(n),
        method="BDF",
        t_eval=times,
        jac_sparsity=diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(n, n)),
        rtol=1.0e-6,
        atol=1.0e-8,
        max_step=0.01,
    )
    return r, sol.y.T


def reference_model(r0, times):
    """
    The paper's semi-analytic heat front, Eq. (37) in cylindrical geometry
    (d = 1), integrated for (r_f - r0)^2
    """
    d = 1.0

    def H(t):
        return float(drive_temperature(t)) ** (4.0 + ALPHA)

    def dH_dt(t):
        ramp = 0.8 if 1.0 <= t < 2.0 else 0.0
        return (4.0 + ALPHA) * float(drive_temperature(t)) ** (3.0 + ALPHA) * ramp

    def rhs(t, g):
        gam = np.sqrt(max(g[0], 0.0))
        x = gam / r0
        log_term = np.log1p(x) / x if x > 1.0e-12 else 1.0 - 0.5 * x
        bracket = (
            0.5 * EPS * (1.0 - gam**2 / (C * H(t) ** (2.0 - EPS)) * dH_dt(t))
            + (1.0 - EPS) * d * (log_term - 1.0)
            + 1.0
        )
        return [2.0 * C * H(t) ** (1.0 - EPS) * bracket / (1.0 - EPS)]

    sol = solve_ivp(rhs, (0.0, T_END), [0.0], t_eval=times, max_step=0.005, rtol=1e-8)
    return np.sqrt(np.maximum(sol.y[0], 0.0))


def write_frames(result, r0, checkpoint_dir="checkpoints", n_pixels=320):
    """Write frames of the temperature across the cylinder (no text)"""

    os.makedirs(checkpoint_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(checkpoint_dir, "*.png")):
        os.remove(stale)
    extent = r0 + L
    xs = np.linspace(-extent, extent, n_pixels)
    rr = np.hypot(*np.meshgrid(xs, xs, indexing="ij"))
    for i, T in enumerate(result["T"]):
        image = np.interp(
            rr,
            result["r"],
            T,
            left=float(drive_temperature(result["times"][i])),
            right=T_COLD,
        )
        image = np.where(rr < r0, float(drive_temperature(result["times"][i])), image)
        fig = plt.figure(figsize=(n_pixels / 100, n_pixels / 100), dpi=100)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.imshow(image.T, cmap="inferno", origin="lower", vmin=0.0, vmax=2.0)
        ax.set_axis_off()
        fig.savefig(os.path.join(checkpoint_dir, f"T{i:03d}.png"), dpi=100)
        plt.close(fig)


def make_plot(results):
    # Plot the heat front of each inner radius, and the temperature profiles
    # of the smallest, against the paper's diffusion equation and model
    _, axes = plt.subplots(2, 1, figsize=(7, 8), dpi=80)

    ax = axes[0]
    for k, (r0, res) in enumerate(results.items()):
        ax.plot(
            res["times"], res["front"] - r0, "-", color=f"C{k}", label=f"r0 = {r0:g} cm"
        )
        ax.plot(res["times"], res["front_ref"] - r0, "--", color=f"C{k}")
        ax.plot(res["times"], res["front_model"], ":", color=f"C{k}")
    ax.set_xlabel("t [ns]")
    ax.set_ylabel("r_f - r0 [cm]")
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, None)
    ax.set_title("solid adirondax   dashed diffusion equation   dotted model")
    ax.legend(loc="upper left", framealpha=1.0)

    ax = axes[1]
    r0 = min(results)
    res = results[r0]
    for k, t in enumerate((1.0, 2.0, 3.0)):
        i = np.argmin(np.abs(res["times"] - t))
        ax.plot(
            res["r"] - r0, res["T"][i], "-", color=f"C{k + 4}", label=f"t = {t:g} ns"
        )
        ax.plot(res["r_ref"] - r0, res["T_ref"][i], "--", color=f"C{k + 4}")
    ax.set_xlabel("r - r0 [cm]")
    ax.set_ylabel("T [HeV]")
    ax.set_xlim(0.0, 0.4)
    ax.set_ylim(0.0, 2.1)
    ax.set_title(
        f"temperature, r0 = {r0:g} cm   (solid adirondax, dashed diffusion equation)"
    )
    ax.legend(loc="upper right", framealpha=1.0)

    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=2, help="resolution multiplier")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write frames",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make the summary plot",
    )
    args = parser.parse_args()

    results = {}
    t0 = time.time()
    for r0 in R0S:
        with contextlib.redirect_stdout(io.StringIO()):
            sim = set_up_simulation(r0, args.res)

        # Evolve the system
        res = run(sim)
        res["front"] = np.array([front(res["r"], T) for T in res["T"]])

        # the references
        res["r_ref"], res["T_ref"] = reference_diffusion(r0, res["times"])
        res["front_ref"] = np.array([front(res["r_ref"], T) for T in res["T_ref"]])
        res["front_model"] = reference_model(r0, res["times"])
        results[r0] = res

        at = [np.argmin(np.abs(res["times"] - t)) for t in (1.0, 2.0, 3.0)]
        print(
            f"r0 = {r0:g} cm: r_f - r0 at 1, 2, 3 ns = "
            + ", ".join(f"{res['front'][i] - r0:.3f}" for i in at)
            + " cm (diffusion equation "
            + ", ".join(f"{res['front_ref'][i] - r0:.3f}" for i in at)
            + "; model "
            + ", ".join(f"{res['front_model'][i]:.3f}" for i in at)
            + ")"
        )
    print("Run time (s): ", time.time() - t0)

    if args.save:
        write_frames(results[0.1], 0.1)

    if args.plot:
        make_plot(results)

    sim.results = results
    return sim


if __name__ == "__main__":
    main()
