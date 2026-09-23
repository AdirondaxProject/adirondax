"""
Tests for coupling the simulation domain to external circuits.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adirondax import coupled_step
from adirondax.hydro.geometry import get_geometry
from adirondax.hydro.mhd2d import hydro_mhd2d_fluxes

# A series RLC bank driving a load
R, L, C, L_LOAD = 0.5, 7.0e-7, 1.0e-6, 3.0e-7
V0, DT, N_STEPS = 6.0e5, 2.0e-9, 500


def _circuit_step(q_prev, i_prev):
    """Backward Euler on the bank, with the load presented as a voltage."""

    def step(voltages):
        i = (L * i_prev / DT + q_prev / C - voltages[0]) / (L / DT + DT / C + R)
        return (q_prev - DT * i, i), jnp.array([i])

    return step


def _load_step(i_prev):
    """The load is an inductor: V = L_load dI/dt."""

    def step(currents):
        i = currents[0]
        return (i,), jnp.array([L_LOAD * (i - i_prev) / DT])

    return step


def _reference():
    """The same circuit with the load folded in as extra inductance."""
    L_total = L + L_LOAD
    q, i = C * V0, 0.0
    history = []
    for _ in range(N_STEPS):
        i = (L_total * i / DT + q / C) / (L_total / DT + DT / C + R)
        q = q - DT * i
        history.append(i)
    return np.array(history)


def test_coupled_step_matches_the_tightly_coupled_solution():
    reference = _reference()

    q, i = C * V0, 0.0
    currents = jnp.array([0.0])
    history = []
    for _ in range(N_STEPS):
        _, (q, i), currents, _ = coupled_step(
            _load_step(i), _circuit_step(q, i), currents
        )
        i = float(currents[0])
        history.append(i)
    history = np.array(history)

    scale = np.max(np.abs(reference))
    assert scale > 1.0e5  # the bank really did deliver a large current
    assert np.max(np.abs(history - reference)) / scale < 1.0e-12


def test_coupled_step_handles_several_ports():
    gains = jnp.array([2.0, -3.0])  # load impedance per port
    offsets = jnp.array([1.5, 0.5])

    def load_step(currents):
        return currents, gains * currents

    def circuit_step(voltages):
        # a passive admittance per port, plus a drive
        currents = offsets - 0.25 * voltages
        return voltages, currents

    _, _, currents, voltages = coupled_step(load_step, circuit_step, jnp.zeros(2))

    # both relations must hold at once
    np.testing.assert_allclose(voltages, gains * currents, rtol=1e-12)
    np.testing.assert_allclose(currents, offsets - 0.25 * voltages, rtol=1e-12)


def _driven_run(bphi_drive, n_steps):
    nR, nz, LR, Lz, r_min = 32, 8, 0.12, 0.6, 0.08
    geom = get_geometry("cylindrical", [LR, Lz], [nR, nz], r_min=r_min)
    geom_work = get_geometry(
        "cylindrical", [LR, Lz], [nR, nz], num_ghost_x=1, r_min=r_min
    )
    one = jnp.ones((nR, nz))
    edge = jnp.ones((1, nz))
    p_gas, b_z = 1.0, 0.5
    p_tot = p_gas + 0.5 * b_z**2

    rho, vx, vy, P = one, 0 * one, 0 * one, p_tot * one
    bx, by, vz, bz = 0 * one, b_z * one, 0 * one, 0 * one
    for _ in range(n_steps):
        ghost = {
            "rho": (edge, edge),
            "vx": (0 * edge, 0 * edge),
            "vy": (0 * edge, 0 * edge),
            # gas pressure held fixed, so an imposed B_phi raises the total
            "P": (p_tot * edge, (p_tot + 0.5 * bphi_drive**2) * edge),
            "bx": (0 * edge, 0 * edge),
            "by": (b_z * edge, b_z * edge),
            "vz": (0 * edge, 0 * edge),
            "bz": (0 * edge, bphi_drive * edge),
        }
        rho, vx, vy, P, bx, by, vz, bz = hydro_mhd2d_fluxes(
            rho,
            vx,
            vy,
            P,
            bx,
            by,
            5.0 / 3.0,
            geom_work,
            2.0e-4,
            "hlld",
            True,
            "driven",
            "periodic",
            vz,
            bz,
            geom,
            ghost_x=ghost,
        )
    return rho, vx, bz


def test_driven_boundary_is_quiet_when_it_matches_the_interior():
    rho, vx, bz = _driven_run(0.0, 40)
    assert jnp.max(jnp.abs(rho - 1.0)) < 1e-12
    assert jnp.max(jnp.abs(vx)) < 1e-12
    assert jnp.max(jnp.abs(bz)) == 0.0


def test_driven_boundary_acts_as_a_magnetic_piston():
    rho, vx, bz = _driven_run(0.4, 40)

    assert jnp.min(vx) < -1e-3  # pushed inwards
    assert jnp.max(jnp.abs(rho - 1.0)) > 1e-3
    # the field penetrates from the driven edge and has not crossed the domain
    assert jnp.abs(bz[-1, 0]) > 1e-2
    assert jnp.max(jnp.abs(bz[: bz.shape[0] // 2, :])) < 1e-9
