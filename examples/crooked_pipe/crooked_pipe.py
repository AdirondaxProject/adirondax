import argparse
import glob
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import adirondax as adx
from adirondax.constants import code_constants
from adirondax.hydro.geometry import get_geometry
from adirondax.radiation.fld2d import radiation_fld2d_step
from adirondax.radiation.rt2d import radiation_rt2d_step

# the thick walls make the implicit radiation solve stiff, which needs double
# precision
jax.config.update("jax_enable_x64", True)

"""
Simulate the crooked pipe (radiation transport and diffusion)

A thin, winding pipe runs through an optically thick wall. Radiation from a
hot source at its mouth flows down the pipe, around its bends, and slowly
heats the walls. It is a standard test of radiation transport through
optically thin channels in thick material, posed by Graziani & LeBlanc (2000),
in axisymmetric (R,z) geometry.

The problem is static: the material only exchanges energy with the radiation.

Philip Mocz (2026)
"""


# Code units: cm, shake (1e-8 s), g and keV, so that energies are in GJ; times
# are reported in seconds
UNITS = {"length": 1.0, "time": 1.0e-8, "mass": 1.0, "temperature": 1.160452e7}
SECOND = 1.0 / UNITS["time"]

GAMMA = 5.0 / 3.0
CV = 0.1  # specific heat, GJ / (g keV)
RHO_THIN, SIGMA_THIN = 0.01, 0.2  # g / cm^3, 1 / cm
RHO_THICK, SIGMA_THICK = 10.0, 200.0
T_INIT = 0.05  # keV
T_SOURCE = 0.3  # keV, at the mouth of the pipe (R < 0.5 at z = 0)
R_PIPE = 0.5

# where the temperature is recorded: (R, z), as in Graziani & LeBlanc (2000)
TRACERS = [(0.0, 0.25), (0.0, 2.75), (1.25, 3.5), (0.0, 4.25), (0.0, 6.75)]

# The timestep starts at 1e-11 s and grows by 10% a step, as in Brantley et al.
# (2019), up to a maximum, until the end time
T_END = 1.0e-5  # s
DT_FIRST = 1.0e-11  # s
DT_GROWTH = 1.1
DT_MAX = 1.0e-8  # s


def in_pipe(R, z):
    """The thin pipe: along the axis, up and over the inner block, and back"""
    return (
        ((z < 2.5) & (R < 0.5))
        | ((z >= 2.5) & (z < 3.0) & (R < 1.5))
        | ((z >= 3.0) & (z < 4.0) & (R >= 1.0) & (R < 1.5))
        | ((z >= 4.0) & (z < 4.5) & (R < 1.5))
        | ((z >= 4.5) & (R < 0.5))
    )


def set_up_simulation(
    resolution_multiplier=2, method="diffusion", num_polar=4, num_azimuthal=4
):
    # Define the parameters for the simulation
    nR = 20 * resolution_multiplier
    nz = 70 * resolution_multiplier

    # the gas constant k_B / (mu m_p) that gives the specific heat CV
    c = code_constants(UNITS)
    mu = c["boltzmann_constant"] / (c["proton_mass"] * CV * (GAMMA - 1.0))

    params = {
        "physics": {
            "radiation": True,
        },
        "mesh": {
            "geometry": "cylindrical",
            "resolution": [nR, nz],
            "box_size": [2.0, 7.0],
            "boundary_condition": ["axis", "driven"],
        },
        "units": UNITS,
        "time": {
            "span": 1.0,
            "num_timesteps": 1,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": GAMMA, "mean_molecular_weight": mu},
        },
        "radiation": {
            "method": method,
            "num_polar": num_polar,
            "num_azimuthal": num_azimuthal,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # the opacity of each material is fixed
    def opacity(rho, T):
        sigma = jnp.where(rho > 1.0, SIGMA_THICK, SIGMA_THIN)
        return {"sigma_s": 0.0, "sigma_a": sigma, "sigma_p": sigma}

    sim.opacity = opacity

    # Set initial conditions: everything at T_INIT, in equilibrium with the
    # radiation
    sim.state["t"] = 0.0
    R, z = sim.mesh
    rho = jnp.where(in_pipe(R, z), RHO_THIN, RHO_THICK)
    sim.state["rho"] = rho
    sim.state["vx"] = jnp.zeros(R.shape)
    sim.state["vy"] = jnp.zeros(R.shape)
    sim.state["P"] = rho * sim.gas_constant * T_INIT
    T_init = T_INIT * jnp.ones(R.shape)
    if method == "diffusion":
        sim.state["Er"] = sim.blackbody_energy(T_init)
    else:
        sim.state["I"] = sim.blackbody_intensity(T_init)

    # The source fills the mouth of the pipe at z = 0; everything else on the
    # ends of the pipe is a vacuum (the outer wall at R = 2 reflects, but it
    # is too deep in the thick material to matter)
    mouth = (R[:, :1] < R_PIPE) * jnp.ones((1, 1, 1))
    if method == "diffusion":
        E_in = sim.blackbody_energy(T_SOURCE * jnp.ones((nR, 1))) * mouth
        ghosts = {"Er": (E_in, jnp.zeros_like(E_in))}
    else:
        I_in = sim.blackbody_intensity(T_SOURCE * jnp.ones((nR, 1))) * mouth[:, None]
        ghosts = {"I": (I_in, jnp.zeros_like(I_in))}
    sim.driven_boundary = lambda state, axis: ghosts

    return sim


def temperature(sim, P):
    return P / (sim.state["rho"] * sim.gas_constant)


def timesteps(dt_max=DT_MAX):
    """The timesteps, in seconds"""
    dts, t, dt = [], 0.0, DT_FIRST
    while t < T_END * (1.0 - 1e-12):
        dt = min(dt, dt_max, T_END - t)
        dts.append(dt)
        t += dt
        dt *= DT_GROWTH
    return np.array(dts)


def make_step(sim):
    """
    A jitted timestep of variable size, from the radiation solvers that
    Simulation uses, for a material held static
    """

    rad = sim.params["radiation"]
    c = sim.constants
    geom = get_geometry("cylindrical", sim.box_size, sim.resolution)
    W0 = {k: sim.state[k] for k in ("rho", "vx", "vy")}
    common = {
        "opacity": sim.opacity(sim.state["rho"], None),
        "group_edges": (),
        "c": c["speed_of_light"],
        "a_r": c["radiation_constant"],
        "gas_constant": sim.gas_constant,
        "gamma": GAMMA,
        "bc_x": "axis",
        "bc_y": "driven",
        "tol": rad["tol"],
        "static": True,
    }
    ghosts = sim.driven_boundary(sim.state, 1)

    @jax.jit
    def step(P, field, dt):
        W = {**W0, "P": P}
        if rad["method"] == "diffusion":
            W, field = radiation_fld2d_step(
                W, field, geom, dt, ghost_y=ghosts["Er"], **common
            )
        else:
            W, field = radiation_rt2d_step(
                W,
                field,
                geom,
                dt,
                angles=sim.angles,
                ghost_y=ghosts["I"],
                **common,
            )
        return W["P"], field

    return step


def run(sim, dt_max=DT_MAX):
    """
    Evolve the simulation, recording the temperature at the tracers at every
    step, and its map at 100 times spaced evenly in log time
    """

    R, z = sim.mesh
    R, z = np.asarray(R[:, 0]), np.asarray(z[0, :])
    cells = [(np.argmin(np.abs(R - r)), np.argmin(np.abs(z - zz))) for r, zz in TRACERS]
    key = "Er" if sim.params["radiation"]["method"] == "diffusion" else "I"
    step = make_step(sim)

    dts = timesteps(dt_max)
    t_steps = np.cumsum(dts)
    map_times = np.geomspace(DT_FIRST, T_END, 100)
    map_steps = set(np.searchsorted(t_steps, map_times * (1.0 - 1e-9)))

    P, field = sim.state["P"], sim.state[key]
    times, history, maps = [0.0], [[T_INIT] * len(TRACERS)], []
    for n, dt in enumerate(dts):
        P, field = step(P, field, dt * SECOND)
        T = np.asarray(temperature(sim, P))
        times.append(t_steps[n])
        history.append([T[i, j] for i, j in cells])
        if n in map_steps:
            maps.append((t_steps[n], T))
    sim.state["P"], sim.state[key] = P, field
    sim.state["t"] = t_steps[-1] * SECOND

    return {"times": np.array(times), "history": np.array(history), "maps": maps}


def plot_map(ax, T, sim, title=None, tracers=True):
    """The temperature in the (z, R) plane, mirrored about the axis"""

    R, z = sim.mesh
    T = np.concatenate((np.flip(T, axis=0), T), axis=0)
    im = ax.imshow(
        T,
        cmap="inferno",
        origin="lower",
        extent=[0.0, 7.0, -2.0, 2.0],
        vmin=T_INIT,
        vmax=T_SOURCE,
    )
    pipe = np.asarray(in_pipe(R, z))
    pipe = np.concatenate((np.flip(pipe, axis=0), pipe), axis=0)
    ax.contour(
        np.linspace(0.0, 7.0, pipe.shape[1]),
        np.linspace(-2.0, 2.0, pipe.shape[0]),
        pipe.astype(float),
        levels=[0.5],
        colors="w",
        linewidths=0.6,
    )
    if tracers:
        for k, (r, zz) in enumerate(TRACERS):
            ax.plot(zz, r, "o", ms=4, color=f"C{k}", mec="w")
    if title is not None:
        ax.set_title(title)
    return im


def load_reference():
    """
    The implicit Monte Carlo solution (Kull) of Brantley et al. (2019): for each
    tracer, its times [s] and temperatures [keV]
    """
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), "imc_reference.txt"))
    return [data[data[:, 0] == k + 1, 1:].T for k in range(len(TRACERS))]


def compare(result, reference):
    """
    Print the temperature of each tracer at the end, and when it gets halfway
    there from T_INIT
    """

    def arrival(t, T, T_level):
        i = np.argmax(T > T_level)
        return t[i] if T[i] > T_level else np.inf

    for k, (t_ref, T_ref) in enumerate(reference):
        T_level = 0.5 * (T_INIT + T_ref[-1])
        t_k = arrival(result["times"], result["history"][:, k], T_level)
        t_ref_k = arrival(t_ref, T_ref, T_level)
        print(
            f"  Pt {k + 1}: T(1e-5 s) = {result['history'][-1, k]:.3f} keV "
            f"(IMC {T_ref[-1]:.3f});  reaches {T_level:.3f} keV at {t_k:.2e} s "
            f"(IMC {t_ref_k:.2e})"
        )


def write_frames(results, sim, checkpoint_dir="checkpoints", width=480, gap=4):
    """
    Write frames of the temperature of each method (transport above diffusion),
    at matching times, filling the frame
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(checkpoint_dir, "*.png")):
        os.remove(stale)

    # each map is 7 cm by 4 cm, the panels separated by a gap of a few pixels
    n = len(results)
    panel = width * 4.0 / 7.0
    height = n * panel + (n - 1) * gap
    dpi = 100
    n_frames = len(next(iter(results.values()))["maps"])
    for i in range(n_frames):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        for j, result in enumerate(results.values()):
            bottom = (n - 1 - j) * (panel + gap) / height
            ax = fig.add_axes([0.0, bottom, 1.0, panel / height])
            plot_map(ax, result["maps"][i][1], sim, tracers=False)
            ax.set_axis_off()
        fig.savefig(os.path.join(checkpoint_dir, f"T{i:03d}.png"), dpi=dpi)
        plt.close(fig)


def make_plot(results, sim):
    # Plot the temperature of each method, and its history at the tracers
    # against the implicit Monte Carlo reference solution
    n = len(results)
    fig, axes = plt.subplots(n + 1, 1, figsize=(7, 3.2 * (n + 1)), dpi=80)

    for ax, (method, result) in zip(axes[:n], results.items()):
        t, T = result["maps"][-1]
        im = plot_map(ax, T, sim, f"{method}, t = {t:.0e} s")
        fig.colorbar(im, ax=ax, label="T [keV]")
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("R [cm]")

    ax = axes[n]
    for k, (t_ref, T_ref) in enumerate(load_reference()):
        ax.loglog(t_ref, T_ref, ":", color="k", lw=1.2, label="IMC" if k == 0 else None)
    for (method, result), ls in zip(results.items(), ("-", "--")):
        for k, (r, zz) in enumerate(TRACERS):
            ax.loglog(
                result["times"][1:],
                result["history"][1:, k],
                ls,
                color=f"C{k}",
                label=f"Pt {k + 1}: (R, z) = ({r}, {zz})" if ls == "-" else None,
            )
    ax.set_xlim(1.0e-11, T_END)
    ax.set_ylim(0.05, 0.4)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4], ["0.05", "0.1", "0.2", "0.3", "0.4"])
    ax.minorticks_off()
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T [keV]")
    ax.set_title(
        "   ".join(
            f"{ls} {method}"
            for (method, _), ls in zip(results.items(), ("solid", "dashed"))
        )
        + "   dotted implicit Monte Carlo"
    )
    ax.legend(loc="upper left", framealpha=1.0, fontsize=8)

    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=2, help="resolution multiplier")
    parser.add_argument(
        "--method",
        default="both",
        choices=["both", "transport", "diffusion"],
        help="radiation method",
    )
    parser.add_argument(
        "--dt-max", type=float, default=DT_MAX, help="largest timestep, in s"
    )
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

    methods = ["transport", "diffusion"] if args.method == "both" else [args.method]
    reference = load_reference()
    results = {}
    for method in methods:
        sim = set_up_simulation(args.res, method)

        # Evolve the system
        t0 = time.time()
        results[method] = run(sim, args.dt_max)
        print(f"Run time, {method} (s): ", time.time() - t0)
        compare(results[method], reference)

    if args.save:
        write_frames(results, sim)

    if args.plot:
        make_plot(results, sim)

    sim.results = results
    return sim


if __name__ == "__main__":
    main()
