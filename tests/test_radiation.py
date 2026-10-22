"""
Tests for the implicit, multigroup, discrete ordinates radiation transport.
"""

import contextlib
import io

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

import adirondax as adx
from adirondax.hydro.geometry import get_geometry
from adirondax.radiation.angles import get_angles
from adirondax.radiation.planck import group_emission, planck_fraction
from adirondax.radiation.rt2d import radiation_moments, radiation_rt2d_step

GAMMA = 5.0 / 3.0


def gas(n_x, n_y, T=1.0, vx=0.0, rho=1.0):
    """Uniform gas of temperature T (with gas_constant = 1)"""

    ones = jnp.ones((n_x, n_y))
    return {
        "rho": rho * ones,
        "vx": vx * ones,
        "vy": 0.0 * ones,
        "P": rho * T * ones,
    }


def make_step(geom, dt, angles, opacity, group_edges=(), c=1.0, a_r=1.0, **kwargs):
    """A jitted radiation step with gas_constant = 1"""

    def step(W, I):
        return radiation_rt2d_step(
            W,
            I,
            geom,
            dt,
            angles=angles,
            opacity=opacity,
            group_edges=group_edges,
            c=c,
            a_r=a_r,
            gas_constant=1.0,
            gamma=GAMMA,
            **kwargs,
        )

    return jax.jit(step)


def isotropic(E, num_dirs, num_groups=1):
    """Isotropic intensities of energy density E in each group"""

    E = jnp.asarray(E)
    return jnp.broadcast_to(
        E[..., None, :, :] / (4.0 * jnp.pi), (num_groups, num_dirs) + E.shape[-2:]
    )


@pytest.mark.parametrize("x", [1e-4, 0.1, 1.0, 1.999, 2.0, 2.001, 5.0, 30.0])
def test_planck_fraction_matches_quadrature(x):
    exact = quad(lambda t: t**3 / np.expm1(t), 0.0, x, epsabs=1e-14, epsrel=1e-13)
    exact = 15.0 / np.pi**4 * exact[0]

    assert abs(float(planck_fraction(x)) - exact) < 1e-13


def test_group_emission_sums_to_grey():
    T = jnp.array([[0.5, 1.0], [3.0, 10.0]])
    eps = group_emission(T, (0.3, 1.0, 4.0, 9.0), 2.0)

    assert eps.shape == (5, 2, 2)
    assert jnp.allclose(jnp.sum(eps, axis=0), 2.0 * T**4 / (4.0 * jnp.pi), rtol=1e-13)


def test_group_emission_is_accurate_in_the_wien_tail():
    # nearly all the emission lies below these groups, so differencing the
    # fractions below their edges would cancel down to round-off
    eps = group_emission(jnp.array(1.0), (10.0, 30.0, 40.0), 4.0 * jnp.pi)
    for f, (lo, hi) in enumerate([(30.0, 40.0), (40.0, np.inf)], start=2):
        exact = quad(
            lambda t: t**3 * np.exp(-t) / -np.expm1(-t),
            lo,
            hi,
            epsabs=0.0,
            epsrel=1e-13,
        )
        exact = 15.0 / np.pi**4 * exact[0]
        assert float(eps[f]) == pytest.approx(exact, rel=1e-10, abs=0.0)


def test_group_emission_is_finite_at_zero_temperature():
    def total(T):
        return jnp.sum(group_emission(T, (1.0, 3.0), 1.0))

    eps = group_emission(jnp.array([0.0, 1.0e-3]), (1.0, 3.0), 1.0)
    assert jnp.all(jnp.isfinite(eps))
    assert jnp.all(eps[:, 0] == 0.0)
    assert jnp.isfinite(jax.grad(total)(0.0))
    assert jnp.isfinite(jax.grad(total)(1.0e-3))


@pytest.mark.parametrize("num_polar, num_azimuthal", [(2, 2), (4, 6), (6, 8)])
def test_angles_integrate_moments_exactly(num_polar, num_azimuthal):
    a = get_angles(num_polar, num_azimuthal)
    w = a["w"]

    assert abs(w.sum() - 1.0) < 1e-14
    for mu in (a["mu_x"], a["mu_y"]):
        assert abs((w * mu).sum()) < 1e-14
        assert abs((w * mu**2).sum() - 1.0 / 3.0) < 1e-14
    assert np.allclose(a["mu_x"] ** 2 + a["mu_y"] ** 2 + a["mu_z"] ** 2, 1.0)

    # mirror images flip one component and are involutions
    assert np.allclose(a["mu_x"][a["reflect_x"]], -a["mu_x"])
    assert np.allclose(a["mu_y"][a["reflect_x"]], a["mu_y"])
    assert np.allclose(a["mu_y"][a["reflect_y"]], -a["mu_y"])
    assert np.array_equal(a["reflect_x"][a["reflect_x"]], np.arange(len(w)))

    # the psi-flux vanishes at psi = 0 and pi, and differences to cos(psi)
    d = (a["psi_face_hi"] - a["psi_face_lo"]) / a["d_psi"]
    assert np.allclose(d * a["sin_zeta"], a["mu_x"])


def test_odd_angle_counts_are_rejected():
    with pytest.raises(ValueError):
        get_angles(3, 4)
    with pytest.raises(ValueError):
        get_angles(4, 5)


def test_grey_thermal_equilibrium():
    """Gas and radiation relax to the equilibrium set by total energy"""

    n = 8
    angles = get_angles(2, 2)
    geom = get_geometry("cartesian", [1.0, 1.0], [n, n])
    W, I = gas(n, n, T=1.0), isotropic(100.0 * jnp.ones((n, n)), 4)
    opacity = {"sigma_s": 0.0, "sigma_a": 100.0, "sigma_p": 100.0}
    step = make_step(geom, 1e-4, angles, opacity, c=100.0)

    E_tot = 1.5 + 100.0
    for _ in range(40):
        W, I = step(W, I)

    T_eq = brentq(lambda T: 1.5 * T + T**4 - E_tot, 1.0, 10.0)
    E_r = radiation_moments(I, angles, 100.0)["E"][0]
    assert jnp.max(jnp.abs(W["P"] - T_eq)) < 1e-6
    assert jnp.max(jnp.abs(E_r - T_eq**4)) < 1e-4
    assert abs(float(jnp.mean(1.5 * W["P"] + E_r)) - E_tot) < 1e-12


def test_multigroup_thermal_equilibrium():
    """three groups of different opacities"""

    n = 4
    angles = get_angles(2, 2)
    geom = get_geometry("cartesian", [1.0, 1.0], [n, n])
    edges = (4.0, 8.0)
    sigma = jnp.array([100.0, 200.0, 300.0])[:, None, None]
    E0 = jnp.array([10.0, 20.0, 30.0])[:, None, None] * jnp.ones((3, n, n))
    W, I = gas(n, n, T=1.0), isotropic(E0, 4, 3)
    step = make_step(
        geom,
        1e-3,
        angles,
        {"sigma_s": 0.0, "sigma_a": sigma, "sigma_p": sigma},
        edges,
        c=10.0,
    )

    for _ in range(50):
        W, I = step(W, I)

    T = float(jnp.mean(W["P"]))
    E_r = jnp.mean(radiation_moments(I, angles, 10.0)["E"], axis=(1, 2))
    assert abs(T - 2.752) < 1e-3
    assert jnp.allclose(
        E_r, 4.0 * jnp.pi * group_emission(jnp.array(T), edges, 1.0), rtol=1e-6
    )


@pytest.mark.parametrize(
    "geometry, bc_x", [("cartesian", "outflow"), ("cylindrical", "axis")]
)
def test_identical_groups_match_grey(geometry, bc_x):
    """With the same opacities in every group, the sum over groups is grey"""

    nx, ny = 8, 8
    angles = get_angles(2, 4)
    geom = get_geometry(geometry, [1.0, 1.0], [nx, ny])
    R, z = np.meshgrid(
        (np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny, indexing="ij"
    )
    W = gas(nx, ny, vx=0.3)
    W["P"] = 1.0 + 2.0 * jnp.exp(-((R - 0.3) ** 2 + (z - 0.5) ** 2) / 0.05)
    opacity = {"sigma_s": 2.0, "sigma_a": 5.0, "sigma_p": 7.0}
    kwargs = {"bc_x": bc_x, "bc_y": "outflow", "tol": 1e-12}

    E0 = jnp.ones((nx, ny))
    W1, I1 = make_step(geom, 0.05, angles, opacity, c=10.0, **kwargs)(
        W, isotropic(E0, 8)
    )
    edges = (0.5, 2.0)
    W3, I3 = make_step(geom, 0.05, angles, opacity, edges, c=10.0, **kwargs)(
        W,
        isotropic(group_emission(jnp.ones((nx, ny)), edges, 1.0) * 4.0 * jnp.pi, 8, 3),
    )

    assert jnp.allclose(jnp.sum(I3, axis=0), I1[0], rtol=1e-8, atol=1e-12)
    assert jnp.allclose(W3["P"], W1["P"], rtol=1e-8)
    assert jnp.allclose(W3["vx"], W1["vx"], rtol=1e-8)


@pytest.mark.parametrize("sigma", [0.0, 10.0])
@pytest.mark.parametrize("bc_y", ["reflective", "periodic"])
def test_isotropic_field_is_balanced_in_cylindrical_geometry(sigma, bc_y):
    """The angular redistribution cancels the radial flux of a uniform field"""

    nR, nz = 16, 8
    angles = get_angles(4, 4)
    geom = get_geometry("cylindrical", [1.0, 1.0], [nR, nz])
    W, I = gas(nR, nz), isotropic(jnp.ones((nR, nz)), 16)
    opacity = {"sigma_s": 0.5 * sigma, "sigma_a": sigma, "sigma_p": sigma}
    step = make_step(geom, 0.1, angles, opacity, c=10.0, bc_x="axis", bc_y=bc_y)

    for _ in range(3):
        W, I = step(W, I)

    assert jnp.max(jnp.abs(4.0 * jnp.pi * I - 1.0)) < 1e-12
    assert jnp.max(jnp.abs(W["P"] - 1.0)) < 1e-12
    assert jnp.max(jnp.abs(W["vx"])) < 1e-12


@pytest.mark.parametrize("solver", ["gmres", "jacobi"])
def test_static_diffusion(solver):
    """a gaussian pulse diffuses through a scattering medium"""

    nx, ny = 256, 2
    angles = get_angles(2, 2)
    geom = get_geometry("cartesian", [2.0, 2.0 * ny / nx], [nx, ny])
    x = -1.0 + (jnp.arange(nx) + 0.5) * 2.0 / nx
    E0 = jnp.where(jnp.abs(x) < 0.5, jnp.exp(-40.0 * x**2), jnp.exp(-10.0))
    W, I = gas(nx, ny), isotropic(E0[:, None] * jnp.ones((nx, ny)), 4)
    c, sigma, dt = 1e3, 4e4, 0.01
    opacity = {"sigma_s": sigma, "sigma_a": 0.0, "sigma_p": 0.0}
    step = make_step(
        geom, dt, angles, opacity, c=c, bc_x="outflow", solver=solver, max_iter=100000
    )

    # the gas is held fixed, else the radiation pressure accelerates it
    for _ in range(60):
        _, I = step(W, I)

    s = 1.0 + 160.0 * c / (3.0 * sigma) * 60 * dt
    E_exact = jnp.exp(-40.0 * x**2 / s) / jnp.sqrt(s)
    E = radiation_moments(I, angles, c)["E"][0, :, 0]
    assert jnp.sum(jnp.abs(E - E_exact)) / jnp.sum(E_exact) < 0.05


@pytest.mark.parametrize("solver", ["gmres", "jacobi"])
def test_beam_in_vacuum_follows_its_direction(solver):
    nx, ny = 64, 32
    angles = get_angles(2, 4)
    num_dirs = len(angles["w"])
    geom = get_geometry("cartesian", [2.0, 1.0], [nx, ny])
    n_beam = int(np.argmax(np.where(angles["mu_y"] > 0.0, angles["mu_x"], -1.0)))
    i_beam = 8
    lo = jnp.zeros((1, num_dirs, nx, 1)).at[0, n_beam, i_beam, 0].set(1.0)
    ghost_y = (lo, jnp.zeros((1, num_dirs, nx, 1)))
    opacity = {"sigma_s": 0.0, "sigma_a": 0.0, "sigma_p": 0.0}
    step = make_step(
        geom,
        10.0,
        angles,
        opacity,
        bc_x="outflow",
        bc_y="driven",
        ghost_y=ghost_y,
        solver=solver,
        tol=1e-10,
        max_iter=100000,
    )

    _, I = step(gas(nx, ny), jnp.zeros((1, num_dirs, nx, ny)))

    E = np.array(radiation_moments(I, angles, 1.0)["E"][0])
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) / ny
    centroid = (E * x[:, None]).sum(0) / E.sum(0)
    expected = x[i_beam] + y * angles["mu_x"][n_beam] / angles["mu_y"][n_beam]
    assert np.max(np.abs(centroid - expected)) < 2.0 * (2.0 / nx)
    assert np.all(E[: i_beam - 2] < 1e-8)


def test_moving_gas_is_isotropic_in_its_rest_frame():
    """uniform gas moving at 3% of the speed of light"""

    n, c, v = 4, 100.0, 3.0
    angles = get_angles(4, 4)
    geom = get_geometry("cartesian", [1.0, 1.0], [n, n])
    W, I = gas(n, n, vx=v), isotropic(jnp.ones((n, n)), 16)
    step = make_step(
        geom, 0.01, angles, {"sigma_s": 0.0, "sigma_a": 1.0, "sigma_p": 1.0}, c=c
    )

    def totals(W, I):
        m = radiation_moments(I, angles, c)
        energy = W["P"] / (GAMMA - 1.0) + 0.5 * W["rho"] * W["vx"] ** 2 + m["E"][0]
        momentum = W["rho"] * W["vx"] + m["Fx"][0] / c**2
        return jnp.mean(energy), jnp.mean(momentum)

    E0, p0 = totals(W, I)
    for _ in range(1000):
        W, I = step(W, I)
    E1, p1 = totals(W, I)

    assert abs(E1 - E0) < 1e-12 * E0
    assert abs(p1 - p0) < 1e-12 * p0

    v = float(jnp.mean(W["vx"]))
    Gam = (1.0 - angles["mu_x"] * v / c) / np.sqrt(1.0 - (v / c) ** 2)
    I0 = np.array(I[0, :, 0, 0]) * Gam**4
    assert I0.std() / I0.mean() < 1e-6

    m = radiation_moments(I, angles, c)
    P_xx = (
        4.0
        * jnp.pi
        * jnp.sum(
            jnp.asarray(angles["w"] * angles["mu_x"] ** 2)[:, None, None] * I[0], axis=0
        )
    )
    F_first_order = v * (m["E"][0] + P_xx)
    assert jnp.allclose(m["Fx"][0], F_first_order, rtol=2e-3)


@pytest.mark.parametrize(
    "geometry, bc_x, solver, group_edges",
    [
        ("cylindrical", "axis", "gmres", (1.0, 3.0)),
        ("cartesian", "periodic", "jacobi", ()),
    ],
)
def test_gradients_match_finite_differences(geometry, bc_x, solver, group_edges):
    nx, ny = 6, 4
    angles = get_angles(2, 2)
    geom = get_geometry(geometry, [1.0, 1.0], [nx, ny])
    R = (jnp.arange(nx) + 0.5) / nx
    num_groups = len(group_edges) + 1
    sigma = 3.0 * jnp.arange(1, num_groups + 1)[:, None, None]
    opacity = {"sigma_s": 1.0, "sigma_a": sigma, "sigma_p": 2.0 * sigma}
    step = make_step(
        geom,
        0.05,
        angles,
        opacity,
        group_edges,
        c=10.0,
        bc_x=bc_x,
        bc_y="outflow",
        solver=solver,
        tol=1e-12,
    )

    @jax.jit
    def loss(amplitude):
        W = gas(nx, ny, vx=0.1)
        W["P"] = W["P"] + amplitude * jnp.exp(-10.0 * (R[:, None] - 0.5) ** 2)
        I = jnp.ones((num_groups, 4, nx, ny)) / (4.0 * jnp.pi * num_groups)
        for _ in range(2):
            W, I = step(W, I)
        return jnp.sum(W["P"] ** 2) + 10.0 * jnp.sum(I[:, 0] ** 2) + jnp.sum(W["vx"])

    a, h = 0.5, 1e-6
    fd = (loss(a + h) - loss(a - h)) / (2.0 * h)
    assert abs(jax.grad(loss)(a) - fd) < 1e-7 * abs(fd)
    assert abs(jax.jacfwd(loss)(a) - fd) < 1e-7 * abs(fd)


def make_sim(geometry, bcs, magnetic=False):
    params = {
        "physics": {"hydro": True, "radiation": True, "magnetic": magnetic},
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
            "num_polar": 2,
            "num_azimuthal": 4,
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
    sim.state["I"] = sim.blackbody_intensity(jnp.ones_like(R))
    return sim


@pytest.mark.parametrize(
    "geometry, bcs",
    [
        ("cylindrical", ["axis", "reflective"]),
        ("cartesian", ["reflective", "periodic"]),
    ],
)
def test_simulation_conserves_energy(geometry, bcs):
    sim = make_sim(geometry, bcs)
    vol = get_geometry(geometry, sim.box_size, sim.resolution)["vol"]

    def total_energy(s):
        E_gas = s["P"] / (GAMMA - 1.0) + 0.5 * s["rho"] * (s["vx"] ** 2 + s["vy"] ** 2)
        E_rad = jnp.sum(radiation_moments(s["I"], sim.angles, 100.0)["E"], axis=0)
        return float(jnp.sum((E_gas + E_rad) * vol))

    def radiation_energy(s):
        return float(jnp.sum(radiation_moments(s["I"], sim.angles, 100.0)["E"] * vol))

    E0, E_rad0 = total_energy(sim.state), radiation_energy(sim.state)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()

    assert abs(total_energy(sim.state) - E0) < 1e-12 * E0
    # the hot spot has heated the radiation
    assert radiation_energy(sim.state) > 1.01 * E_rad0


def test_simulation_runs_with_mhd():
    sim = make_sim("cylindrical", ["axis", "wall"], magnetic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()

    for k in ("rho", "P", "I", "bx", "by"):
        assert jnp.all(jnp.isfinite(sim.state[k]))


def test_radiation_parameters_are_validated():
    def params(**rad):
        return {
            "physics": {"hydro": True, "radiation": True},
            "radiation": rad,
        }

    for bad in [
        {"num_polar": 3},
        {"group_edges": [2.0, 1.0]},
        {"group_edges": [1.0], "opacity": {"absorption": [1.0, 2.0, 3.0]}},
        {"solver": "sor"},
    ]:
        with pytest.raises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            adx.Simulation(params(**bad))

    with pytest.raises(ValueError), contextlib.redirect_stdout(io.StringIO()):
        adx.Simulation({"physics": {"radiation": True}})


@pytest.mark.parametrize("rotation", [False, True])
def test_magnetized_gas_in_equilibrium_with_radiation_stays_put(rotation):
    """The radiation sees the gas pressure, not the total pressure with B^2/2"""

    params = {
        "physics": {
            "hydro": True,
            "radiation": True,
            "magnetic": True,
            "rotation": rotation,
        },
        "mesh": {
            "geometry": "cylindrical",
            "resolution": [8, 8],
            "box_size": [1.0, 1.0],
            "boundary_condition": ["axis", "periodic"],
        },
        "constants": {
            "speed_of_light": 100.0,
            "radiation_constant": 1.0,
            "boltzmann_constant": 1.0,
            "proton_mass": 1.0,
        },
        "time": {"span": 0.05, "num_timesteps": 5},
        "hydro": {"eos": {"gamma": GAMMA}},
        "radiation": {
            "num_polar": 2,
            "num_azimuthal": 2,
            "opacity": {"absorption": 10.0},
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        sim = adx.Simulation(params)

    # gas at T = 1 threaded by a uniform axial field, so the total pressure is
    # 1 + 0.5 (and the azimuthal field and velocity, if any, are zero)
    R, _ = sim.mesh
    sim.state["t"] = 0.0
    sim.state["rho"] = jnp.ones_like(R)
    sim.state["vx"] = jnp.zeros_like(R)
    sim.state["vy"] = jnp.zeros_like(R)
    sim.state["bx"] = jnp.zeros_like(R)
    sim.state["by"] = jnp.ones_like(R)
    sim.state["P"] = 1.5 * jnp.ones_like(R)
    if rotation:
        sim.state["vphi"] = jnp.zeros_like(R)
        sim.state["bphi"] = jnp.zeros_like(R)
    sim.state["I"] = sim.blackbody_intensity(jnp.ones_like(R))
    I0 = sim.state["I"]
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()

    assert jnp.max(jnp.abs(sim.state["P"] - 1.5)) < 1e-10
    assert jnp.max(jnp.abs(sim.state["I"] - I0)) < 1e-10 * jnp.max(I0)
