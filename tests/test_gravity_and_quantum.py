"""
Tests for the gravitational kick and the Schrodinger field.
"""

import contextlib
import io

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import adirondax as adx
from adirondax.constants import constants
from adirondax.hydro.euler2d import hydro_euler2d_accelerate
from adirondax.hydro.geometry import get_geometry


def make_sim(**params):
    with contextlib.redirect_stdout(io.StringIO()):
        return adx.Simulation(params)


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical"])
def test_kick_leaves_the_pressure_unchanged(geometry):
    # a strong kick, a*dt of a few times the velocity: the work done on the gas
    # goes into its kinetic energy alone
    n = 8
    geom = get_geometry(geometry, [1.0, 1.0], [n, n])
    rng = np.random.default_rng(0)
    W = {
        "rho": jnp.asarray(1.0 + rng.random((n, n))),
        "vx": jnp.asarray(rng.normal(size=(n, n))),
        "vy": jnp.asarray(rng.normal(size=(n, n))),
        "P": jnp.asarray(1.0 + rng.random((n, n))),
    }
    if geometry == "cylindrical":
        W["vz"] = jnp.asarray(rng.normal(size=(n, n)))
    ax = jnp.asarray(rng.normal(size=(n, n)))
    ay = jnp.asarray(rng.normal(size=(n, n)))
    dt = 3.0

    W_new = hydro_euler2d_accelerate(W, ax, ay, 5.0 / 3.0, geom, dt)

    np.testing.assert_allclose(W_new["P"], W["P"], rtol=1e-12)
    np.testing.assert_allclose(W_new["vx"], W["vx"] + dt * ax, rtol=1e-12)
    np.testing.assert_allclose(W_new["vy"], W["vy"] + dt * ay, rtol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("num_timesteps", [10, -1])
def test_quantum_field_keeps_its_precision(dtype, num_timesteps):
    # a mass with m/hbar = 1 in the (cgs) code units, so that the span takes
    # some tens of the timesteps the kinetic term allows
    hbar = constants["planck_constant"] / (2.0 * jnp.pi)
    m_eV = constants["speed_of_light"] ** 2 * hbar / constants["electron_volt"]
    sim = make_sim(
        physics={"quantum": True, "gravity": True},
        mesh={"resolution": [16, 16]},
        time={"span": 1.0e-2, "num_timesteps": num_timesteps},
        quantum={"m_eV": m_eV},
    )
    assert sim.m_per_hbar == pytest.approx(1.0)
    X, Y = sim.mesh
    psi = jnp.sqrt(1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * X)) * jnp.exp(
        1.0j * jnp.cos(2.0 * jnp.pi * Y)
    )
    sim.state["t"] = 0.0
    sim.state["psi"] = psi.astype(dtype)
    sim.run()

    assert sim.state["psi"].dtype == dtype
    # the evolution is unitary
    mass = jnp.sum(jnp.abs(psi) ** 2)
    rtol = 1e-5 if dtype == jnp.complex64 else 1e-12
    assert float(jnp.sum(jnp.abs(sim.state["psi"]) ** 2)) == pytest.approx(
        float(mass), rel=rtol
    )
