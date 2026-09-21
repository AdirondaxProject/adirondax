"""
Tests for axisymmetric cylindrical (R,z) geometry.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import adirondax as adx


def make_params(nR=32, nz=32, LR=1.0, Lz=1.0, nt=20, t_stop=0.1, rotation=False):
    return {
        "physics": {"hydro": True, "rotation": rotation},
        "mesh": {
            "geometry": "cylindrical",
            "resolution": [nR, nz],
            "box_size": [LR, Lz],
            "boundary_condition": ["axis", "reflective"],
        },
        "time": {"span": t_stop, "num_timesteps": nt},
        "hydro": {"eos": {"gamma": 5.0 / 3.0}, "slope_limiting": True},
    }


def cell_volumes(sim):
    """2*pi * integral R dR dz over each cell"""
    LR, Lz = sim.box_size
    nR, nz = sim.resolution
    dR, dz = LR / nR, Lz / nz
    edges = dR * jnp.arange(nR + 1)
    vol = (jnp.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * dz)[:, None]
    return jnp.broadcast_to(vol, (nR, nz))


def test_uniform_state_is_well_balanced():
    sim = adx.Simulation(make_params(nt=50, t_stop=0.5))
    sim.state["t"] = 0.0
    R, _ = sim.mesh
    sim.state["rho"] = jnp.ones_like(R)
    sim.state["vx"] = jnp.zeros_like(R)
    sim.state["vy"] = jnp.zeros_like(R)
    sim.state["P"] = jnp.ones_like(R)
    sim.run()

    assert jnp.max(jnp.abs(sim.state["rho"] - 1.0)) < 1e-12
    assert jnp.max(jnp.abs(sim.state["P"] - 1.0)) < 1e-12
    assert jnp.max(jnp.abs(sim.state["vx"])) < 1e-12
    assert jnp.max(jnp.abs(sim.state["vy"])) < 1e-12


def test_shock_tube_along_z_matches_cartesian():
    def run(geometry):
        params = make_params(nR=16, nz=64, nt=40, t_stop=0.1)
        params["mesh"]["geometry"] = geometry
        params["mesh"]["boundary_condition"] = [
            "axis" if geometry == "cylindrical" else "reflective",
            "reflective",
        ]
        sim = adx.Simulation(params)
        sim.state["t"] = 0.0
        R, z = sim.mesh
        sim.state["rho"] = jnp.where(z < 0.5, 1.0, 0.125)
        sim.state["vx"] = jnp.zeros_like(R)
        sim.state["vy"] = jnp.zeros_like(R)
        sim.state["P"] = jnp.where(z < 0.5, 1.0, 0.1)
        sim.run()
        return sim.state

    cyl = run("cylindrical")
    car = run("cartesian")

    for key in ["rho", "vy", "P"]:
        np.testing.assert_allclose(cyl[key], car[key], rtol=1e-10, atol=1e-10)
    # no radial motion is generated
    assert jnp.max(jnp.abs(cyl["vx"])) < 1e-12


def test_mass_and_energy_conservation():
    sim = adx.Simulation(make_params(nR=32, nz=32, nt=60, t_stop=0.15))
    sim.state["t"] = 0.0
    R, z = sim.mesh
    r_sph = jnp.sqrt(R**2 + (z - 0.5) ** 2)
    sim.state["rho"] = jnp.ones_like(R)
    sim.state["vx"] = jnp.zeros_like(R)
    sim.state["vy"] = jnp.zeros_like(R)
    sim.state["P"] = jnp.where(r_sph < 0.15, 10.0, 0.1)

    vol = cell_volumes(sim)
    gamma = sim.params["hydro"]["eos"]["gamma"]

    def totals(st):
        mass = jnp.sum(st["rho"] * vol)
        en = jnp.sum(
            (
                st["P"] / (gamma - 1.0)
                + 0.5 * st["rho"] * (st["vx"] ** 2 + st["vy"] ** 2)
            )
            * vol
        )
        return mass, en

    m0, e0 = totals(sim.state)
    sim.run()
    m1, e1 = totals(sim.state)

    assert abs(m1 - m0) / m0 < 1e-12
    assert abs(e1 - e0) / e0 < 1e-12


def test_angular_momentum_conservation():
    sim = adx.Simulation(make_params(nR=32, nz=32, nt=50, t_stop=0.15, rotation=True))
    sim.state["t"] = 0.0
    R, z = sim.mesh
    sim.state["rho"] = jnp.ones_like(R)
    sim.state["vx"] = jnp.zeros_like(R)
    sim.state["vy"] = jnp.zeros_like(R)
    sim.state["P"] = jnp.ones_like(R) + 5.0 * jnp.exp(
        -((R / 0.2) ** 2 + ((z - 0.5) / 0.2) ** 2)
    )
    sim.state["vphi"] = 0.5 * R * jnp.exp(-(((z - 0.5) / 0.3) ** 2))

    vol = cell_volumes(sim)
    R_c = R  # arithmetic center is fine for a convergence-level check

    def total_L(st):
        return jnp.sum(st["rho"] * R_c * st["vphi"] * vol)

    L0 = total_L(sim.state)
    sim.run()
    L1 = total_L(sim.state)

    assert abs(L1 - L0) / abs(L0) < 1e-3


def test_rotating_equilibrium_is_steady():
    omega = 1.0
    sim = adx.Simulation(make_params(nR=48, nz=8, nt=40, t_stop=0.2, rotation=True))
    sim.state["t"] = 0.0
    R, _ = sim.mesh
    # rho = 1, vphi = omega*R  =>  dP/dR = omega^2 R  =>  P = P0 + omega^2 R^2/2
    sim.state["rho"] = jnp.ones_like(R)
    sim.state["vx"] = jnp.zeros_like(R)
    sim.state["vy"] = jnp.zeros_like(R)
    sim.state["vphi"] = omega * R
    sim.state["P"] = 10.0 + 0.5 * omega**2 * R**2
    sim.run()

    # the radial velocity stays small: the equilibrium is maintained rather
    # than drifting under an inconsistent source term
    cs = jnp.sqrt(5.0 / 3.0 * 10.0)
    assert jnp.max(jnp.abs(sim.state["vx"])) / cs < 2e-3


def test_sedov_blast_stays_spherical():
    nR, nz = 64, 128
    sim = adx.Simulation(make_params(nR=nR, nz=nz, LR=0.5, Lz=1.0, nt=-1, t_stop=0.05))
    sim.state["t"] = 0.0
    R, z = sim.mesh
    r_sph = jnp.sqrt(R**2 + (z - 0.5) ** 2)
    sim.state["rho"] = jnp.ones_like(R)
    sim.state["vx"] = jnp.zeros_like(R)
    sim.state["vy"] = jnp.zeros_like(R)
    sim.state["P"] = jnp.where(r_sph < 0.05, 100.0, 0.1)
    sim.run()

    rho = sim.state["rho"]
    assert jnp.isfinite(rho).all()

    jmid = nz // 2
    profile_mid = rho[:, jmid]  # outward along R at z = 0.5
    profile_axis = rho[0, jmid : jmid + nR]  # outward along z at R -> 0

    # dR == dz, so both profiles sample radius (i + 0.5) * dR. Compare in L1:
    # at a captured shock the two profiles can be offset by a fraction of a
    # cell, which makes a pointwise norm meaningless no matter the resolution.
    assert float(jnp.mean(jnp.abs(profile_mid - profile_axis))) < 0.02

    # the shock is well developed and sits in the same cell in both directions
    def front(profile):
        return int(nR - 1 - jnp.argmax((profile > 1.05)[::-1]))

    assert jnp.max(profile_mid) > 1.8  # strong shock present
    assert abs(front(profile_mid) - front(profile_axis)) <= 1


def test_gradient_is_finite():
    def loss(p0):
        sim = adx.Simulation(make_params(nR=16, nz=16, nt=5, t_stop=0.02))
        sim.state["t"] = 0.0
        R, _ = sim.mesh
        sim.state["rho"] = jnp.ones_like(R)
        sim.state["vx"] = jnp.zeros_like(R)
        sim.state["vy"] = jnp.zeros_like(R)
        sim.state["P"] = p0 * (1.0 + jnp.exp(-((R / 0.2) ** 2)))
        sim.state = sim._evolve(sim.state)
        return jnp.sum(sim.state["rho"] ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)

    fd = (loss(1.0 + 1e-4) - loss(1.0 - 1e-4)) / 2e-4
    assert abs(g - fd) / max(abs(fd), 1e-8) < 1e-2


# --------------------------------------------------------------------------
# Parameter validation
# --------------------------------------------------------------------------
def test_validation_errors():
    p = make_params()
    p["mesh"]["boundary_condition"] = ["reflective", "reflective"]
    with pytest.raises(ValueError, match="requires boundary_condition"):
        adx.Simulation(p)

    p = make_params()
    p["mesh"]["geometry"] = "cartesian"
    with pytest.raises(ValueError, match="requires cylindrical"):
        adx.Simulation(p)

    p = make_params(rotation=True)
    p["mesh"]["geometry"] = "cartesian"
    p["mesh"]["boundary_condition"] = ["reflective", "reflective"]
    with pytest.raises(ValueError, match="rotation"):
        adx.Simulation(p)

    p = make_params()
    p["physics"]["gravity"] = True
    with pytest.raises(NotImplementedError, match="gravity"):
        adx.Simulation(p)
