"""
Tests for the half-step predictors for primitive fields.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from adirondax.hydro import euler2d, mhd2d

GAMMA = 5.0 / 3.0


def fields(x, y, magnetic, out_of_plane, cylindrical):
    """Primitive fields at a point; B comes from a potential, so div(B) = 0"""

    f = {
        "rho": 1.0 + 0.2 * jnp.sin(1.1 * x + 0.4 * y),
        "vx": 0.3 * jnp.sin(1.3 * x + 0.7 * y),
        "vy": 0.2 * jnp.cos(0.9 * x - 1.1 * y),
        "Pg": 1.5 + 0.3 * jnp.sin(x * y),
    }
    if out_of_plane:
        f["vz"] = 0.4 * jnp.cos(0.5 * x + 0.8 * y)

    if magnetic:

        def A(x, y):
            return 0.3 * jnp.sin(x) * jnp.cos(y) + 0.2 * x

        if cylindrical:
            f["Bx"] = -jax.grad(A, 1)(x, y)
            f["By"] = jax.grad(lambda x, y: x * A(x, y))(x, y) / x
        else:
            f["Bx"] = jax.grad(A, 1)(x, y)
            f["By"] = -jax.grad(A, 0)(x, y)
        if out_of_plane:
            f["Bz"] = 0.5 * jnp.cos(x - 0.6 * y)
    return f


def exact_rates(x, y, magnetic, out_of_plane, cylindrical):
    """d/dt of the primitive fields, from the conservative equations"""

    def F(x, y):
        return fields(x, y, magnetic, out_of_plane, cylindrical)

    def get(f, k):
        return f.get(k, 0.0)

    def derived(x, y):
        f = F(x, y)
        rho, vx, vy, vz = f["rho"], f["vx"], f["vy"], get(f, "vz")
        Bx, By, Bz = get(f, "Bx"), get(f, "By"), get(f, "Bz")
        B_sq = Bx**2 + By**2 + Bz**2
        v_sq = vx**2 + vy**2 + vz**2
        Pt = f["Pg"] + 0.5 * B_sq
        E = f["Pg"] / (GAMMA - 1.0) + 0.5 * rho * v_sq + 0.5 * B_sq
        vB = vx * Bx + vy * By + vz * Bz
        h = x if cylindrical else 1.0
        return {
            "mass": (rho * vx * h, rho * vy),
            "momx": ((rho * vx * vx + Pt - Bx * Bx) * h, rho * vy * vx - By * Bx),
            "momy": ((rho * vx * vy - Bx * By) * h, rho * vy * vy + Pt - By * By),
            "momz": ((rho * vx * vz - Bx * Bz) * h * h, rho * vy * vz - By * Bz),
            "energy": (((E + Pt) * vx - vB * Bx) * h, (E + Pt) * vy - vB * By),
            "Bx": (0.0, vx * By - vy * Bx),
            "By": (-(vx * By - vy * Bx) * h, 0.0),
            "Bz": (-(vx * Bz - vz * Bx), -(vy * Bz - vz * By)),
            "hoop": (rho * vz**2 + Pt - Bz**2) / x if cylindrical else 0.0,
        }

    h = x if cylindrical else 1.0
    hz = h * h

    def div(k):
        dx = jax.grad(lambda x, y: derived(x, y)[k][0], 0)(x, y)
        dy = jax.grad(lambda x, y: derived(x, y)[k][1], 1)(x, y)
        scale = hz if k == "momz" else h
        return dx / (scale if k not in ("Bx", "Bz") else 1.0) + dy

    f = F(x, y)
    rho, vx, vy, vz = f["rho"], f["vx"], f["vy"], get(f, "vz")
    Bx, By, Bz = get(f, "Bx"), get(f, "By"), get(f, "Bz")

    d_rho = -div("mass")
    d_mom = {
        "vx": -div("momx") + derived(x, y)["hoop"],
        "vy": -div("momy"),
        "vz": -div("momz"),
    }
    v = {"vx": vx, "vy": vy, "vz": vz}
    rates = {"rho": d_rho}
    for k, v_k in v.items():
        rates[k] = (d_mom[k] - v_k * d_rho) / rho
    rates["Bx"] = div("Bx")
    rates["By"] = div("By")
    rates["Bz"] = div("Bz")

    d_E = -div("energy")
    d_kin = sum(rho * v[k] * rates[k] + 0.5 * v[k] ** 2 * d_rho for k in v)
    d_mag = Bx * rates["Bx"] + By * rates["By"] + Bz * rates["Bz"]
    d_Pg = (GAMMA - 1.0) * (d_E - d_kin - d_mag)
    rates["P"] = d_Pg + d_mag if magnetic else d_Pg

    return rates


CASES = [
    ("euler", False, False),
    ("euler", True, False),
    ("euler", True, True),
    ("mhd", False, False),
    ("mhd", False, True),
    ("mhd", True, False),
    ("mhd", True, True),
]


@pytest.mark.parametrize("solver, cylindrical, out_of_plane", CASES)
def test_half_step_matches_the_conservative_equations(
    solver, cylindrical, out_of_plane
):
    magnetic = solver == "mhd"
    x, y = jnp.meshgrid(
        jnp.linspace(0.5, 2.0, 7), jnp.linspace(-1.0, 1.0, 5), indexing="ij"
    )
    x, y = x.ravel(), y.ravel()

    def primitive(x, y):
        f = fields(x, y, magnetic, out_of_plane, cylindrical)
        f["P"] = f.pop("Pg")
        if magnetic:
            f["P"] = f["P"] + 0.5 * (f["Bx"] ** 2 + f["By"] ** 2 + f.get("Bz", 0) ** 2)
        return f

    def with_RBz(x, y):
        f = primitive(x, y)
        if magnetic and cylindrical and out_of_plane:
            f["RBz"] = x * f["Bz"]
        return f

    W = jax.vmap(primitive)(x, y)
    W_dx = jax.vmap(jax.jacfwd(with_RBz, 0))(x, y)
    W_dy = jax.vmap(jax.jacfwd(with_RBz, 1))(x, y)
    geom = {"is_cylindrical": cylindrical, "r": x if cylindrical else None}

    dt = 1.0e-3
    module = mhd2d if magnetic else euler2d
    Wp = module.half_step(W, W_dx, W_dy, GAMMA, geom, dt)
    rates = jax.vmap(
        lambda x, y: exact_rates(x, y, magnetic, out_of_plane, cylindrical)
    )(x, y)

    for k in W:
        np.testing.assert_allclose(
            (Wp[k] - W[k]) / (0.5 * dt), rates[k], atol=1e-10, err_msg=k
        )
