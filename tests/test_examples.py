"""
Smoke tests for the scripts in examples/.
"""

import jax.numpy as jnp
import pytest

from adirondax.hydro.common2d import get_div
from adirondax.utils import run_example_main

rel_tol = 1e-3

ARGV = ["--res", "1", "--no-save", "--no-plot"]


def run(name):
    return run_example_main(f"examples/{name}/{name}.py", argv=ARGV)


def test_gresho():
    sim = run("gresho")
    assert sim.resolution == [32, 32]
    assert sim.state["t"] > 0.0

    # the vortex is set up symmetrically in x and y, and the scheme must not
    # break that symmetry
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(
        float(jnp.mean(jnp.abs(sim.state["vy"]))), rel=1e-5
    )

    assert jnp.mean(sim.state["rho"]) == pytest.approx(1.0, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(0.0817774, rel=rel_tol)
    assert jnp.mean(sim.state["P"]) == pytest.approx(5.737929, rel=rel_tol)


def test_kelvin_helmholtz():
    sim = run("kelvin_helmholtz")
    assert sim.resolution == [32, 32]
    assert sim.state["t"] > 0.0

    # mass is conserved exactly by the finite volume scheme
    assert jnp.mean(sim.state["rho"]) == pytest.approx(1.5, rel=rel_tol)
    assert jnp.max(sim.state["rho"]) == pytest.approx(2.083289, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vy"])) == pytest.approx(0.00497184, rel=rel_tol)
    assert jnp.mean(sim.state["P"]) == pytest.approx(2.525596, rel=rel_tol)


def test_orszag_tang():
    sim = run("orszag_tang")
    assert sim.resolution == [32, 32]
    assert sim.state["t"] > 0.0

    # constrained transport keeps the field divergence-free to round-off
    dx = sim.box_size[0] / sim.resolution[0]
    dy = sim.box_size[1] / sim.resolution[1]
    div_B = get_div(sim.state["bx"], sim.state["by"], dx, dy)
    b_rms = jnp.sqrt(jnp.mean(sim.state["bx"] ** 2 + sim.state["by"] ** 2))
    assert jnp.max(jnp.abs(div_B)) * dx / b_rms < 1.0e-4

    assert jnp.mean(sim.state["rho"]) == pytest.approx(0.2210484, rel=rel_tol)
    assert jnp.mean(sim.state["P"]) == pytest.approx(0.2219067, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["bx"])) == pytest.approx(0.1804803, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["by"])) == pytest.approx(0.2074233, rel=rel_tol)


def test_rayleigh_taylor():
    sim = run("rayleigh_taylor")
    assert sim.resolution == [16, 48]
    assert sim.state["t"] > 0.0

    # reflecting walls: no mass leaves the box
    assert jnp.mean(sim.state["rho"]) == pytest.approx(1.5, rel=rel_tol)
    assert jnp.max(sim.state["rho"]) == pytest.approx(2.078529, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vy"])) == pytest.approx(0.03609437, rel=rel_tol)
    assert jnp.mean(sim.state["P"]) == pytest.approx(2.482582, rel=rel_tol)


def test_sedov():
    sim = run("sedov")
    assert sim.resolution == [32, 64]
    assert sim.state["t"] > 0.0

    # the blast is still well inside the box, so the ambient medium is intact
    assert jnp.max(sim.state["rho"]) == pytest.approx(1.822437, rel=rel_tol)
    assert jnp.mean(sim.state["rho"]) == pytest.approx(0.9667935, rel=rel_tol)
    assert jnp.mean(sim.state["P"]) == pytest.approx(0.222086, rel=rel_tol)


def test_sod():
    sim = run("sod")
    assert sim.resolution == [100, 10]
    assert sim.state["t"] > 0.0

    # the problem is one-dimensional and must stay that way
    assert jnp.max(jnp.abs(sim.state["vy"])) < 1.0e-5
    assert jnp.max(jnp.ptp(sim.state["rho"], axis=1)) < 1.0e-6

    # no new extremum is created: rho stays within the two initial states
    assert jnp.max(sim.state["rho"]) == pytest.approx(1.0, rel=rel_tol)
    assert jnp.min(sim.state["rho"]) == pytest.approx(0.125, rel=rel_tol)

    assert jnp.mean(sim.state["rho"]) == pytest.approx(0.5624968, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(0.4433709, rel=rel_tol)
    assert jnp.mean(sim.state["P"]) == pytest.approx(0.5215686, rel=rel_tol)
