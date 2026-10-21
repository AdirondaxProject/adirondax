"""
Tests for the implicit, multigroup, flux-limited radiation diffusion.
"""

import contextlib
import io

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

import adirondax as adx
from adirondax.hydro.geometry import get_geometry
from adirondax.radiation.fld2d import LIMITERS, radiation_fld2d_step
from adirondax.radiation.planck import group_emission

GAMMA = 5.0 / 3.0


def gas(n_x, n_y, T=1.0, vx=0.0, rho=1.0):
    """Uniform gas of temperature T (with gas_constant = 1)"""

    ones = jnp.ones((n_x, n_y))
    return {"rho": rho * ones, "vx": vx * ones, "vy": 0.0 * ones, "P": rho * T * ones}


def make_step(geom, dt, opacity, group_edges=(), c=100.0, a_r=1.0, **kwargs):
    """A jitted diffusion step with gas_constant = 1"""

    def step(W, E):
        return radiation_fld2d_step(
            W,
            E,
            geom,
            dt,
            opacity=opacity,
            group_edges=group_edges,
            c=c,
            a_r=a_r,
            gas_constant=1.0,
            gamma=GAMMA,
            **kwargs,
        )

    return jax.jit(step)


def test_grey_thermal_equilibrium():
    n = 8
    geom = get_geometry("cartesian", [1.0, 1.0], [n, n])
    W, E = gas(n, n), 100.0 * jnp.ones((1, n, n))
    step = make_step(geom, 1e-4, {"sigma_s": 0.0, "sigma_a": 100.0, "sigma_p": 100.0})

    for _ in range(40):
        W, E = step(W, E)

    T_eq = brentq(lambda T: 1.5 * T + T**4 - 101.5, 1.0, 10.0)
    assert jnp.max(jnp.abs(W["P"] - T_eq)) < 1e-6
    assert jnp.max(jnp.abs(E - T_eq**4)) < 1e-4
    assert abs(float(jnp.mean(1.5 * W["P"] + E[0])) - 101.5) < 1e-12


def test_multigroup_thermal_equilibrium():
    """Section 4.1 of Jiang (2022): three groups of different opacities"""

    n = 4
    geom = get_geometry("cartesian", [1.0, 1.0], [n, n])
    edges = (4.0, 8.0)
    sigma = jnp.array([100.0, 200.0, 300.0])[:, None, None]
    W = gas(n, n)
    E = jnp.array([10.0, 20.0, 30.0])[:, None, None] * jnp.ones((3, n, n))
    opacity = {"sigma_s": 0.0, "sigma_a": sigma, "sigma_p": sigma}
    step = make_step(geom, 1e-3, opacity, edges, c=10.0)

    for _ in range(50):
        W, E = step(W, E)

    T = float(jnp.mean(W["P"]))
    E_eq = 4.0 * jnp.pi * group_emission(jnp.array(T), edges, 1.0)
    assert abs(T - 2.752) < 1e-3
    assert jnp.allclose(jnp.mean(E, axis=(1, 2)), E_eq, rtol=1e-8)


@pytest.mark.parametrize(
    "geometry, bc_x", [("cartesian", "outflow"), ("cylindrical", "axis")]
)
def test_identical_groups_match_grey(geometry, bc_x):
    """
    With the same opacities in every group, the sum over groups is grey (with
    plain diffusion: a flux limiter depends on the shape of each group's own
    energy density)
    """

    nx, ny = 8, 8
    geom = get_geometry(geometry, [1.0, 1.0], [nx, ny])
    R, z = np.meshgrid(
        (np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny, indexing="ij"
    )
    W = gas(nx, ny, vx=0.3)
    W["P"] = 1.0 + 2.0 * jnp.exp(-((R - 0.3) ** 2 + (z - 0.5) ** 2) / 0.05)
    opacity = {"sigma_s": 2.0, "sigma_a": 5.0, "sigma_p": 7.0}
    kwargs = {"bc_x": bc_x, "bc_y": "outflow", "tol": 1e-12, "flux_limiter": "none"}
    edges = (0.5, 2.0)
    E3 = 4.0 * jnp.pi * group_emission(jnp.ones((nx, ny)), edges, 1.0)

    W1, E1 = make_step(geom, 0.05, opacity, c=10.0, **kwargs)(
        W, jnp.sum(E3, axis=0)[None]
    )
    W3, E3 = make_step(geom, 0.05, opacity, edges, c=10.0, **kwargs)(W, E3)

    assert jnp.allclose(jnp.sum(E3, axis=0), E1[0], rtol=1e-8)
    assert jnp.allclose(W3["P"], W1["P"], rtol=1e-8)
    assert jnp.allclose(W3["vx"], W1["vx"], rtol=1e-8)


@pytest.mark.parametrize("limiter", LIMITERS)
def test_static_diffusion(limiter):
    """A gaussian pulse diffuses through a scattering medium as it should"""

    nx, ny = 256, 2
    geom = get_geometry("cartesian", [2.0, 2.0 * ny / nx], [nx, ny])
    x = -1.0 + (jnp.arange(nx) + 0.5) * 2.0 / nx
    E0 = jnp.where(jnp.abs(x) < 0.5, jnp.exp(-40.0 * x**2), jnp.exp(-10.0))
    E = E0[None, :, None] * jnp.ones((1, nx, ny))
    c, sigma, dt = 1e3, 4e4, 0.01
    opacity = {"sigma_s": sigma, "sigma_a": 0.0, "sigma_p": 0.0}
    step = make_step(geom, dt, opacity, c=c, bc_x="outflow", flux_limiter=limiter)

    W = gas(nx, ny)
    for _ in range(60):
        _, E = step(W, E)

    s = 1.0 + 160.0 * c / (3.0 * sigma) * 60 * dt
    E_exact = jnp.exp(-40.0 * x**2 / s) / jnp.sqrt(s)
    assert jnp.sum(jnp.abs(E[0, :, 0] - E_exact)) / jnp.sum(E_exact) < 0.005


def test_flux_limiter_keeps_the_radiation_slower_than_light():
    """
    In optically thin gas the bulk of the radiation advances at most at the
    speed of light with a flux limiter, while plain diffusion runs far ahead.
    (Backward Euler smears the front ahead of itself, so it is located where
    the energy density falls to a tenth of its peak.)
    """

    nx, ny = 200, 2
    geom = get_geometry("cartesian", [1.0, ny / nx], [nx, ny])
    x = (jnp.arange(nx) + 0.5) / nx
    E0 = jnp.where(x < 0.1, 1.0, 1e-8)[None, :, None] * jnp.ones((1, nx, ny))
    c, dt, n_steps = 1.0, 0.01, 30
    opacity = {"sigma_s": 1e-3, "sigma_a": 0.0, "sigma_p": 0.0}

    def front(limiter):
        step = make_step(geom, dt, opacity, c=c, bc_x="outflow", flux_limiter=limiter)
        E, W = E0, gas(nx, ny)
        for _ in range(n_steps):
            _, E = step(W, E)
        return float(jnp.max(jnp.where(E[0, :, 0] > 0.1 * jnp.max(E), x, 0.0)))

    light = 0.1 + c * dt * n_steps
    assert front("levermore_pomraning") < light + 0.15
    assert front("none") > light + 0.4


def test_heat_front_advances_into_cold_opaque_material():
    """
    A heat front whose opacity falls steeply as it heats (sigma ~ T^-3)
    advances into cold material with the power-law (and arithmetic) face
    average, while the harmonic one lets the opacity of the cold side choke it
    """

    nx, ny = 64, 2
    geom = get_geometry("cartesian", [1.0, ny / nx], [nx, ny])
    x = (jnp.arange(nx) + 0.5) / nx
    T0 = jnp.where(x < 0.1, 1.0, 0.01)[:, None] * jnp.ones((nx, ny))
    W, E = gas(nx, ny), (T0**4)[None]
    W["P"] = T0

    def depth(face_average):
        opacity = {"sigma_s": 0.0}
        step = jax.jit(
            lambda W, E: radiation_fld2d_step(
                W,
                E,
                geom,
                1e-3,
                opacity={
                    **opacity,
                    "sigma_a": 10.0 * jnp.maximum(W["P"], 0.01) ** -3,
                    "sigma_p": 10.0 * jnp.maximum(W["P"], 0.01) ** -3,
                },
                group_edges=(),
                c=100.0,
                a_r=1.0,
                gas_constant=1.0,
                gamma=GAMMA,
                bc_x="reflective",
                flux_limiter="none",
                face_average=face_average,
            )
        )
        W_, E_ = W, E
        for _ in range(50):
            W_, E_ = step(W_, E_)
        return float(jnp.max(jnp.where(W_["P"][:, 0] > 0.3, x, 0.0)))

    assert depth("power_law") > 0.2
    assert depth("arithmetic") > 0.2
    assert depth("harmonic") < 0.12


def test_marshak_source_fills_an_empty_box():
    """A box driven by the same radiation from both ends fills up with it"""

    nx, ny = 16, 4
    geom = get_geometry("cartesian", [1.0, 1.0], [nx, ny])
    E_src = 2.0 * jnp.ones((1, 1, ny))
    opacity = {"sigma_s": 3.0, "sigma_a": 0.0, "sigma_p": 0.0}
    step = make_step(geom, 1.0, opacity, c=10.0, bc_x="driven", ghost_x=(E_src, E_src))

    W, E = gas(nx, ny), jnp.zeros((1, nx, ny))
    for _ in range(50):
        _, E = step(W, E)

    assert jnp.max(jnp.abs(E - 2.0)) < 1e-6


def test_moving_gas_conserves_energy_and_momentum():
    nx, ny = 16, 16
    geom = get_geometry("cartesian", [1.0, 1.0], [nx, ny])
    rng = np.random.default_rng(0)
    W = gas(nx, ny)
    W["vx"] = 0.5 * jnp.asarray(rng.standard_normal((nx, ny)))
    W["vy"] = 0.5 * jnp.asarray(rng.standard_normal((nx, ny)))
    W["P"] = 1.0 + 0.5 * jnp.asarray(rng.random((nx, ny)))
    E = 1.0 + jnp.asarray(rng.random((2, nx, ny)))
    opacity = {"sigma_s": 1.0, "sigma_a": jnp.array([5.0, 0.5])[:, None, None]}
    opacity["sigma_p"] = opacity["sigma_a"]
    step = make_step(geom, 0.01, opacity, (1.0,), bc_x="reflective", bc_y="periodic")

    def total_energy(W, E):
        kinetic = 0.5 * W["rho"] * (W["vx"] ** 2 + W["vy"] ** 2)
        return float(jnp.sum(W["P"] / (GAMMA - 1.0) + kinetic + jnp.sum(E, axis=0)))

    E0 = total_energy(W, E)
    for _ in range(10):
        W, E = step(W, E)

    assert abs(total_energy(W, E) - E0) < 1e-12 * E0


def test_uniform_radiation_in_a_uniform_flow_is_unchanged():
    nx, ny = 8, 8
    geom = get_geometry("cartesian", [1.0, 1.0], [nx, ny])
    T = 1.0
    W = gas(nx, ny, T=T, vx=0.3)
    E = T**4 * jnp.ones((1, nx, ny))
    opacity = {"sigma_s": 1.0, "sigma_a": 10.0, "sigma_p": 10.0}
    W_new, E_new = make_step(geom, 0.01, opacity, a_r=1.0)(W, E)

    assert jnp.max(jnp.abs(E_new - E)) < 1e-12
    assert jnp.max(jnp.abs(W_new["vx"] - 0.3)) < 1e-12
    assert jnp.max(jnp.abs(W_new["P"] - T)) < 1e-12


def test_radiation_pushes_optically_thick_gas_with_its_pressure_gradient():
    """In the diffusion limit the force on the gas is -grad(E/3)"""

    nx, ny = 64, 2
    geom = get_geometry("cartesian", [1.0, ny / nx], [nx, ny])
    x = (jnp.arange(nx) + 0.5) / nx
    E = (1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * x))[None, :, None] * jnp.ones((1, nx, ny))
    W = gas(nx, ny, rho=1e3)
    dt = 1e-6
    opacity = {"sigma_s": 1e5, "sigma_a": 0.0, "sigma_p": 0.0}
    W_new, _ = make_step(geom, dt, opacity, c=1.0)(W, E)

    acceleration = (W_new["vx"][:, 0] - W["vx"][:, 0]) / dt
    expected = -(1.0 / 3.0) * 0.1 * 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x) / 1e3
    assert jnp.max(jnp.abs(acceleration - expected)) < 0.01 * jnp.max(jnp.abs(expected))


@pytest.mark.parametrize(
    "geometry, bc_x, group_edges",
    [("cylindrical", "axis", (1.0, 3.0)), ("cartesian", "periodic", ())],
)
def test_gradients_match_finite_differences(geometry, bc_x, group_edges):
    nx, ny = 6, 4
    geom = get_geometry(geometry, [1.0, 1.0], [nx, ny])
    R = (jnp.arange(nx) + 0.5) / nx
    num_groups = len(group_edges) + 1
    sigma = 3.0 * jnp.arange(1, num_groups + 1)[:, None, None]
    opacity = {"sigma_s": 1.0, "sigma_a": sigma, "sigma_p": 2.0 * sigma}
    step = make_step(
        geom, 0.05, opacity, group_edges, c=10.0, bc_x=bc_x, bc_y="outflow", tol=1e-12
    )

    @jax.jit
    def loss(amplitude):
        W = gas(nx, ny, vx=0.1)
        W["P"] = W["P"] + amplitude * jnp.exp(-10.0 * (R[:, None] - 0.5) ** 2)
        E = jnp.ones((num_groups, nx, ny)) / num_groups
        for _ in range(2):
            W, E = step(W, E)
        return jnp.sum(W["P"] ** 2) + 10.0 * jnp.sum(E[:, 0] ** 2) + jnp.sum(W["vx"])

    a, h = 0.5, 1e-6
    fd = (loss(a + h) - loss(a - h)) / (2.0 * h)
    assert abs(jax.grad(loss)(a) - fd) < 1e-6 * abs(fd)
    assert abs(jax.jacfwd(loss)(a) - fd) < 1e-6 * abs(fd)


def make_sim(geometry, bcs, hydro=True, magnetic=False):
    params = {
        "physics": {"hydro": hydro, "radiation": True, "magnetic": magnetic},
        "mesh": {
            "geometry": geometry,
            "resolution": [16, 16],
            "box_size": [1.0, 1.0],
            "boundary_condition": bcs,
        },
        "constants": {
            "speed_of_light": 100.0,
            "radiation_constant": 1.0,
            "boltzmann_constant": 1.0,
            "proton_mass": 1.0,
        },
        "time": {"span": 0.1, "num_timesteps": 10},
        "hydro": {"eos": {"gamma": GAMMA}, "slope_limiting": True},
        "radiation": {
            "method": "diffusion",
            "group_edges": [3.0],
            "opacity": {"scattering": 1.0, "absorption": [3.0, 0.3]},
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        sim = adx.Simulation(params)

    R, z = sim.mesh
    sim.state["t"] = 0.0
    sim.state["rho"] = jnp.ones_like(R)
    sim.state["vx"] = jnp.zeros_like(R)
    sim.state["vy"] = jnp.zeros_like(R)
    sim.state["P"] = 1.0 + 4.0 * jnp.exp(-(R**2 + (z - 0.5) ** 2) / 0.02)
    if magnetic:
        sim.state["bx"] = jnp.zeros_like(R)
        sim.state["by"] = 0.1 * jnp.ones_like(R)
    sim.state["Er"] = sim.blackbody_energy(jnp.ones_like(R))
    return sim


@pytest.mark.parametrize(
    "geometry, bcs, hydro",
    [
        ("cylindrical", ["axis", "reflective"], True),
        ("cartesian", ["reflective", "periodic"], True),
        ("cylindrical", ["axis", "reflective"], False),
    ],
)
def test_simulation_conserves_energy(geometry, bcs, hydro):
    sim = make_sim(geometry, bcs, hydro=hydro)
    vol = get_geometry(geometry, sim.box_size, sim.resolution)["vol"]

    def total_energy(s):
        E_gas = s["P"] / (GAMMA - 1.0) + 0.5 * s["rho"] * (s["vx"] ** 2 + s["vy"] ** 2)
        return float(jnp.sum((E_gas + jnp.sum(s["Er"], axis=0)) * vol))

    E0 = total_energy(sim.state)
    E_rad0 = float(jnp.sum(sim.state["Er"] * vol))
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()

    assert abs(total_energy(sim.state) - E0) < 1e-12 * E0
    assert float(jnp.sum(sim.state["Er"] * vol)) > 1.01 * E_rad0
    if not hydro:
        # a static material does not move
        assert jnp.all(sim.state["vx"] == 0.0)
        assert jnp.all(sim.state["rho"] == 1.0)


def test_simulation_runs_with_mhd():
    sim = make_sim("cylindrical", ["axis", "wall"], magnetic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()

    for k in ("rho", "P", "Er", "bx", "by"):
        assert jnp.all(jnp.isfinite(sim.state[k]))


def test_diffusion_parameters_are_validated():
    def params(**rad):
        return {"physics": {"hydro": True, "radiation": True}, "radiation": rad}

    for bad in [
        {"method": "monte_carlo"},
        {"flux_limiter": "minerbo"},
        {"face_average": "geometric"},
    ]:
        with pytest.raises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            adx.Simulation(params(**bad))

    # a static material needs a fixed number of timesteps
    with pytest.raises(ValueError), contextlib.redirect_stdout(io.StringIO()):
        adx.Simulation({"physics": {"radiation": True}})
