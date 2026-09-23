import argparse
import glob
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import spicex

import adirondax as adx
from adirondax import coupled_step
from adirondax.hydro.geometry import get_geometry
from adirondax.hydro.mhd2d import hydro_mhd2d_fluxes, hydro_mhd2d_timestep

jax.config.update("jax_enable_x64", True)

"""
Flux compression: a pulsed power generator driving an imploding liner

Philip Mocz (2026)

This is the cylindrical verification problem of Beresnyak et al. (2022). A
capacitor bank drives current through a coaxial device. The current sits in the
vacuum outside a thin annular liner, whose magnetic pressure pushes the liner
inwards; the flux trapped inside is compressed and pushes back, so the liner
rings rather than simply collapsing.

The simulation is coupled to a circuit model;ed with SpiceX.
"""

# Physical constants and parameters.
MU0 = 4.0e-7 * jnp.pi
L_Z = 0.60  # device length (m)
R_MIN = 0.08  # inner electrode (m)
R_0 = 0.18  # initial liner radius (m)
R_MAX = 0.20  # outer electrode, where the generator feeds in (m)
MASS = 80.0e-9  # liner mass (kg), i.e. 80 ug
RHO_VAC = 1.0e-10  # "vacuum" density (kg/m^3)
C_BANK = 1.0e-6  # capacitance (F)
L_BANK = 700.0e-9  # inductance (H)
V_BANK = 6.0e5  # initial capacitor voltage (V)
I_INIT = 2.0e5  # initial current (A)
GAMMA = 5.0 / 3.0
T_END = 2.0e-6  # two full rings (s)

# The magnetic field is carried in units where mu0 = 1, so a current maps to a
# field as B = sqrt(mu0) I / (2 pi r).
SQRT_MU0 = jnp.sqrt(MU0)

# the evolved fields, in the order run_simulation carries them
STATE_KEYS = ("rho", "vx", "vy", "P", "bx", "by", "vphi", "bphi")


def b_phi(current, radius):
    """Azimuthal field of a line current, in code units"""
    return SQRT_MU0 * current / (2.0 * jnp.pi * radius)


def inductance_outer(r):
    return (MU0 / (2.0 * jnp.pi)) * L_Z * jnp.log(R_MAX / r)


def inductance_inner(r):
    return (MU0 / (2.0 * jnp.pi)) * L_Z * jnp.log(r / R_MIN)


def reference_solution(n_steps=200000, t_end=T_END):
    mu = float(MU0)

    def dl1(r):
        return -(mu / (2.0 * np.pi)) * L_Z / r

    def l1(r):
        return (mu / (2.0 * np.pi)) * L_Z * np.log(R_MAX / r)

    def l2(r):
        return (mu / (2.0 * np.pi)) * L_Z * np.log(r / R_MIN)

    def deriv(y):
        r, v, q, i = y
        i2 = I_INIT * l2(R_0) / l2(r)  # flux conserved inside
        di = (q / C_BANK - i * dl1(r) * v) / (L_BANK + l1(r))
        dv = 0.5 * dl1(r) * (i**2 - i2**2) / MASS
        return np.array([v, dv, -i, di])

    dt = t_end / n_steps
    y = np.array([R_0, 0.0, C_BANK * V_BANK, I_INIT])
    out = np.empty((n_steps + 1, 4))
    out[0] = y
    for k in range(n_steps):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[k + 1] = y
    return np.linspace(0.0, t_end, n_steps + 1), out


def set_up_simulation(resolution_multiplier=4, save=True):
    nR = 50 * resolution_multiplier
    nz = 4
    t_stop = T_END

    params = {
        "physics": {
            "hydro": True,
            "magnetic": True,
            "rotation": True,
        },
        "mesh": {
            "geometry": "cylindrical",
            "resolution": [nR, nz],
            "origin": [R_MIN, 0.0],
            "box_size": [R_MAX - R_MIN, L_Z],
            "boundary_condition": ["driven", "periodic"],
        },
        "time": {"span": t_stop, "num_timesteps": -1},
        "output": {"num_checkpoints": 100, "save": save},
        "hydro": {
            "eos": {"type": "ideal", "gamma": GAMMA},
            "riemann_solver": "hlld",
            "slope_limiting": True,
            "cfl": 0.3,
        },
    }

    sim = adx.Simulation(params)

    # Initial conditions
    sim.state["t"] = jnp.array(0.0)
    R, _ = sim.mesh
    dR = (R_MAX - R_MIN) / nR

    # the liner: all of the mass in a thin shell at R_0
    shell = jnp.abs(R - R_0) < 2.0 * dR
    shell_volume = 2.0 * jnp.pi * R_0 * (4.0 * dR) * L_Z
    rho = RHO_VAC + shell * (MASS / shell_volume)

    field = b_phi(I_INIT, R)
    p_gas = 1.0e-4 * 0.5 * jnp.max(field) ** 2

    didt = V_BANK / (L_BANK + inductance_outer(R_0))
    v_r = jnp.where(R > R_0, -(didt / I_INIT) * R * jnp.log(R / R_0), 0.0)

    sim.state["rho"] = rho
    sim.state["vx"] = v_r
    sim.state["vy"] = jnp.zeros(R.shape)
    sim.state["bx"] = jnp.zeros(R.shape)  # b_R
    sim.state["by"] = jnp.zeros(R.shape)  # b_z
    sim.state["vphi"] = jnp.zeros(R.shape)
    sim.state["bphi"] = field
    sim.state["P"] = p_gas + 0.5 * field**2

    return sim


def build_circuit():
    circuit = spicex.Circuit(n_nodes=3)
    circuit.add_capacitor(1, 0, C_BANK)
    circuit.add_inductor(1, 2, L_BANK)
    circuit.add_voltage_source(0, 2, 0.0)  # the device; driven by the plasma
    return circuit


def port_voltage(state, geom_r_edge):
    vx, bz = state[1], state[7]
    e_z = -vx[-1, :] * bz[-1, :] * SQRT_MU0
    return jnp.mean(e_z) * L_Z


def ghost_values(state, current, didt, r_edge, r_ghost, dR):
    rho, vx, vy, P, _, by, vz, bz = state
    zeros = jnp.zeros_like(rho[0:1, :])

    v_edge = vx[-1:, :]
    v_ghost = v_edge + dR * (v_edge / r_edge - didt / current)
    field_ghost = b_phi(current, r_ghost) * jnp.ones_like(zeros)
    p_gas = P[-1:, :] - 0.5 * bz[-1:, :] ** 2

    return {
        "rho": (rho[0:1, :], RHO_VAC * jnp.ones_like(zeros)),
        "vx": (-vx[0:1, :], v_ghost),
        "vy": (vy[0:1, :], vy[-1:, :]),
        "P": (P[0:1, :], p_gas + 0.5 * field_ghost**2),
        "bx": (zeros, zeros),
        "by": (by[0:1, :], by[-1:, :]),
        "vz": (vz[0:1, :], vz[-1:, :]),
        "bz": (bz[0:1, :], field_ghost),
    }


def run_simulation(sim, t_end=T_END, cfl=0.3, record_every=20):
    """
    Step the plasma and the generator forward together.
    """
    nR, nz = sim.resolution
    dR = (R_MAX - R_MIN) / nR
    geom = get_geometry("cylindrical", sim.box_size, sim.resolution, r_min=R_MIN)
    geom_work = get_geometry(
        "cylindrical", sim.box_size, sim.resolution, num_ghost_x=1, r_min=R_MIN
    )
    r_edge = R_MAX - 0.5 * dR
    r_ghost = R_MAX + 0.5 * dR
    cell_volume = np.asarray(geom["vol"]) * np.ones(sim.resolution)
    radius = np.asarray(sim.mesh[0])

    circuit = build_circuit()
    v_nodes = jnp.array([0.0, V_BANK, 0.0])
    i_bank = jnp.array([I_INIT])
    currents = jnp.array([I_INIT])

    state = tuple(sim.state[k] for k in STATE_KEYS)

    @jax.jit
    def advance(state, v_nodes, i_bank, currents, dt, didt):
        def load_step(port_currents):
            ghost = ghost_values(state, port_currents[0], didt, r_edge, r_ghost, dR)
            new = hydro_mhd2d_fluxes(
                *state[:6],
                GAMMA,
                geom_work,
                dt,
                "hlld",
                True,
                "driven",
                "periodic",
                state[6],
                state[7],
                geom,
                ghost_x=ghost,
            )
            return new, jnp.array([port_voltage(new, r_edge)])

        def circuit_step(voltages):
            v_new, _, i_new, _ = circuit.transient_step(
                v_nodes, i_bank, dt, vsrc_values=voltages
            )
            return (v_new, i_new), i_new

        return coupled_step(load_step, circuit_step, currents)

    @jax.jit
    def timestep(state):
        return cfl * hydro_mhd2d_timestep(
            *state[:6], GAMMA, dR, L_Z / nz, state[6], state[7]
        )

    # Run a block of steps inside one compiled scan. The timestep is set by
    # the Alfven speed in the tenuous pseudo-vacuum, so there are a lot of
    # them; pulling the state back to the host on each one costs far more than
    # the step itself.
    @jax.jit
    def block(carry):
        def one_step(carry, _):
            state, v_nodes, i_bank, currents, t, i_prev, didt = carry
            dt = cfl * hydro_mhd2d_timestep(
                *state[:6], GAMMA, dR, L_Z / nz, state[6], state[7]
            )
            # A block always runs its full length, so the final steps of a run
            # would otherwise be clamped to zero length. The circuit's backward
            # Euler companion models go as C/dt, so the step has to stay
            # strictly positive; the overshoot past t_end is a millionth of one
            # step.
            dt = jnp.clip(jnp.minimum(dt, t_end - t), 1.0e-6 * dt, None)
            state, (v_nodes, i_bank), currents, voltages = advance(
                state, v_nodes, i_bank, currents, dt, didt
            )
            i_now = currents[0]
            carry = (
                state,
                v_nodes,
                i_bank,
                currents,
                t + dt,
                i_now,
                (i_now - i_prev) / dt,
            )
            return carry, voltages[0]

        carry, voltages = jax.lax.scan(one_step, carry, None, length=record_every)
        return carry, voltages[-1]

    carry = (state, v_nodes, i_bank, currents, jnp.array(0.0), I_INIT, 0.0)
    history = []
    profiles = []
    while float(carry[4]) < t_end * (1.0 - 1e-12):
        carry, voltage = block(carry)
        state = carry[0]
        mass = np.asarray(state[0]) * cell_volume
        total = mass.sum()
        profiles.append(np.asarray(state[0]).mean(axis=1))
        history.append(
            (
                float(carry[4]),
                float((radius * mass).sum() / total),
                float((np.asarray(state[1]) * mass).sum() / total),
                float(carry[1][1]),
                float(carry[3][0]),
                float(voltage),
            )
        )
    state = carry[0]

    return state, np.array(history), np.array(profiles)


def plot_density(profile, radius, filename, n_pixels=256):
    extent = R_MAX
    xs = np.linspace(-extent, extent, n_pixels)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)

    logrho = np.interp(rr, radius, np.log10(profile), left=np.nan, right=np.nan)
    logrho = np.where((rr >= radius[0]) & (rr <= radius[-1]), logrho, np.log10(RHO_VAC))

    plt.clf()
    ax = plt.gca()
    ax.imshow(
        logrho.T,
        cmap="inferno",
        origin="lower",
        extent=[-extent, extent, -extent, extent],
        vmin=np.log10(RHO_VAC),
        vmax=np.log10(RHO_VAC) + 6.0,
    )
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def write_frames(profiles, radius, checkpoint_dir="checkpoints", n_frames=101):
    """Write evenly spaced frames of the imploding liner."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(checkpoint_dir, "*.png")):
        os.remove(stale)
    picks = np.linspace(0, len(profiles) - 1, n_frames).round().astype(int)
    for i, k in enumerate(picks):
        plot_density(
            profiles[k], radius, os.path.join(checkpoint_dir, f"rho{i:03d}.png")
        )


def make_plot(history):
    """Reproduce figure 4 of Beresnyak et al. (2022)."""
    t, r, v, v_cap, current, v_dev = history.T
    t_ref, y = reference_solution(n_steps=40000, t_end=float(t[-1]))

    _, axes = plt.subplots(2, 1, figsize=(7, 8), dpi=80, sharex=True)

    ax = axes[0]
    ax.plot(t_ref * 1e6, y[:, 2] / C_BANK / 1e3, "k-", lw=1.0, label="exact")
    ax.plot(t_ref * 1e6, y[:, 3] / 1e3, "k-", lw=1.0)
    ax.plot(t * 1e6, v_cap / 1e3, "o", ms=3, color="tab:blue", label="capacitor")
    ax.plot(t * 1e6, current / 1e3, "o", ms=3, color="tab:green", label="current")
    ax.plot(t * 1e6, v_dev / 1e3, "o", ms=3, color="tab:red", label="device")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylabel("voltage [kV],  current [kA]")
    ax.set_title("Flux compression: generator driving an imploding liner")
    ax.legend(loc="lower left", ncol=2, framealpha=1.0)

    ax = axes[1]
    ax.plot(t_ref * 1e6, y[:, 0] * 100, "k-", lw=1.0, label="exact")
    ax.plot(t_ref * 1e6, y[:, 1] / 1e4, "k-", lw=1.0)
    ax.plot(t * 1e6, r * 100, "o", ms=3, color="tab:blue", label="radius [cm]")
    ax.plot(t * 1e6, v / 1e4, "o", ms=3, color="tab:green", label="velocity [cm/us]")
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("time [us]")
    ax.set_ylabel("liner radius [cm],  velocity [cm/us]")
    ax.legend(loc="upper right", framealpha=1.0)

    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=4, help="resolution multiplier")
    parser.add_argument(
        "--t-stop", type=float, default=T_END * 1e6, help="stop time in us"
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write density frames",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make the summary plot",
    )
    args = parser.parse_args()

    sim = set_up_simulation(args.res, save=False)

    t0 = time.time()
    _, history, profiles = run_simulation(sim, t_end=args.t_stop * 1e-6)
    print("Run time (s): ", time.time() - t0)

    t, r, _, v_cap, current, _ = history.T
    _, y = reference_solution(n_steps=40000, t_end=float(t[-1]))
    print(
        f"peak current   : {current.max() / 1e3:7.1f} kA   (exact {y[:, 3].max() / 1e3:7.1f})"
    )
    print(
        f"min radius     : {r.min() * 100:7.2f} cm   (exact {y[:, 0].min() * 100:7.2f})"
    )
    print(
        f"final capacitor: {v_cap[-1] / 1e3:7.1f} kV   "
        f"(exact {y[-1, 2] / C_BANK / 1e3:7.1f})"
    )

    if args.save:
        write_frames(profiles, np.asarray(sim.mesh[0])[:, 0])

    if args.plot:
        make_plot(history)

    return history


if __name__ == "__main__":
    main()
