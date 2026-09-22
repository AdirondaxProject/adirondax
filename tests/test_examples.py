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


def test_brio_wu():
    sim = run("brio_wu")
    assert sim.resolution == [100, 10]
    assert sim.state["t"] > 0.0

    # constrained transport keeps the field divergence-free to round-off
    dx = sim.box_size[0] / sim.resolution[0]
    dy = sim.box_size[1] / sim.resolution[1]
    div_B = get_div(sim.state["bx"], sim.state["by"], dx, dy)
    b_rms = jnp.sqrt(jnp.mean(sim.state["bx"] ** 2 + sim.state["by"] ** 2))
    assert jnp.max(jnp.abs(div_B)) * dx / b_rms < 1.0e-4

    # the outflow boundaries must carry the uniform normal field through
    # untouched: a reflecting wall would drive Bx to zero there
    assert jnp.max(jnp.abs(sim.state["bx"] - 0.75)) < 1.0e-4

    # the problem is one-dimensional and must stay that way
    assert jnp.max(jnp.ptp(sim.state["rho"], axis=1)) < 1.0e-4

    # no new extremum in rho, and the far states are still untouched
    assert jnp.max(sim.state["rho"]) == pytest.approx(1.0, rel=rel_tol)
    assert sim.state["rho"][-1, 0] == pytest.approx(0.125, rel=rel_tol)

    assert jnp.mean(sim.state["rho"]) == pytest.approx(0.5624992, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(0.209511, rel=rel_tol)
    assert jnp.min(sim.state["vy"]) == pytest.approx(-1.620257, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["by"])) == pytest.approx(0.8496271, rel=rel_tol)


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


def test_alfven_wave():
    sim = run("alfven_wave")
    assert sim.resolution == [8, 32]
    assert sim.state["t"] > 0.0

    R, z = sim.mesh

    # after one full period the wave is back where it started
    exact = 0.01 * jnp.sin(2.0 * jnp.pi * z) * R
    assert jnp.mean(jnp.abs(sim.state["bphi"] - exact)) < 5.0e-4

    # the Alfven relation v_phi = -B_phi / sqrt(rho) is maintained
    assert jnp.max(jnp.abs(sim.state["vphi"] + sim.state["bphi"])) < 1.0e-4

    # the hoop stress and the centrifugal force cancel, so the plasma stays put
    # radially; what is left is the O(amplitude^2) magnetic pressure imbalance
    assert jnp.max(jnp.abs(sim.state["vx"])) < 5.0e-3

    # the poloidal field is untouched
    assert jnp.max(jnp.abs(sim.state["bx"])) < 1.0e-4
    assert jnp.max(jnp.abs(sim.state["by"] - 1.0)) < 5.0e-3

    # constrained transport keeps the field divergence-free
    dx = sim.box_size[0] / sim.resolution[0]
    dy = sim.box_size[1] / sim.resolution[1]
    div_B = get_div(sim.state["bx"], sim.state["by"], dx, dy)
    assert jnp.max(jnp.abs(div_B)) * dx < 1.0e-6
