"""
Tests for the Riemann solvers.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from adirondax.hydro import euler2d, mhd2d

GAMMA = 5.0 / 3.0


def random_state(rng, magnetic, Bx=None):
    W = {
        "rho": 0.5 + rng.random(),
        "vx": rng.normal(),
        "vy": rng.normal(),
        "vz": rng.normal(),
        "P": 0.5 + rng.random(),
    }
    if magnetic:
        W.update(
            Bx=rng.normal() if Bx is None else Bx, By=rng.normal(), Bz=rng.normal()
        )
        # the solvers take the total pressure
        W["P"] = W["P"] + 0.5 * (W["Bx"] ** 2 + W["By"] ** 2 + W["Bz"] ** 2)
    return {k: jnp.asarray(v) for k, v in W.items()}


def euler_flux(W):
    """The physical flux, the conserved state and the signal speed |vx| + a"""
    rho, vx, vy, vz, P = (W[k] for k in ("rho", "vx", "vy", "vz", "P"))
    en = P / (GAMMA - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    U = {"mass": rho, "momx": rho * vx, "momy": rho * vy, "momz": rho * vz}
    U["energy"] = en
    F = {
        "mass": rho * vx,
        "momx": rho * vx**2 + P,
        "momy": rho * vx * vy,
        "momz": rho * vx * vz,
        "energy": (en + P) * vx,
    }
    return F, U, jnp.abs(vx) + jnp.sqrt(GAMMA * P / rho)


def mhd_flux(W):
    """The physical flux, the conserved state and the signal speed |vx| + c_f"""
    rho, vx, vy, vz, P = (W[k] for k in ("rho", "vx", "vy", "vz", "P"))
    Bx, By, Bz = W["Bx"], W["By"], W["Bz"]
    B_sq = Bx**2 + By**2 + Bz**2
    en = (P - 0.5 * B_sq) / (GAMMA - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    en = en + 0.5 * B_sq
    v_dot_B = vx * Bx + vy * By + vz * Bz
    U = {"mass": rho, "momx": rho * vx, "momy": rho * vy, "momz": rho * vz}
    U.update(energy=en, By=By, Bz=Bz)
    F = {
        "mass": rho * vx,
        "momx": rho * vx**2 + P - Bx**2,
        "momy": rho * vx * vy - Bx * By,
        "momz": rho * vx * vz - Bx * Bz,
        "energy": (en + P) * vx - Bx * v_dot_B,
        "By": By * vx - Bx * vy,
        "Bz": Bz * vx - Bx * vz,
    }
    # the fast magnetosonic speed along x
    a_sq = GAMMA * (P - 0.5 * B_sq) / rho
    b_sq, bx_sq = B_sq / rho, Bx**2 / rho
    cf_sq = 0.5 * (a_sq + b_sq + jnp.sqrt((a_sq + b_sq) ** 2 - 4.0 * a_sq * bx_sq))
    return F, U, jnp.abs(vx) + jnp.sqrt(cf_sq)


@pytest.mark.parametrize("magnetic", [False, True])
def test_llf_is_the_rusanov_flux(magnetic):
    rng = np.random.default_rng(1)
    solver, physical = (mhd2d, mhd_flux) if magnetic else (euler2d, euler_flux)
    for _ in range(10):
        # the normal field is continuous across the face
        WL = random_state(rng, magnetic)
        WR = random_state(rng, magnetic, Bx=WL.get("Bx"))
        F_L, U_L, C_L = physical(WL)
        F_R, U_R, C_R = physical(WR)
        C = jnp.maximum(C_L, C_R)

        flux = solver.get_flux_llf(WL, WR, GAMMA)

        for k, f in flux.items():
            expected = 0.5 * (F_L[k] + F_R[k]) - 0.5 * C * (U_R[k] - U_L[k])
            assert float(f) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


def degenerate_state(**changes):
    # gamma = 2 and P_gas = 0.5 give gamma P = 1 < Bx^2 = 4 with no transverse
    # field, so the fast and Alfven speeds coincide, and the star states of
    # Miyoshi & Kusano (2005) divide 0 by 0
    W = {"rho": 1.0, "vx": 0.0, "vy": 0.0, "P_gas": 0.5, "Bx": 2.0, "By": 0.0}
    W.update(changes)
    W["P"] = W.pop("P_gas") + 0.5 * (W["Bx"] ** 2 + W["By"] ** 2)
    return {k: jnp.asarray(v) for k, v in W.items()}


@pytest.mark.parametrize("name", ["rho", "vx", "vy", "P_gas", "Bx", "By"])
def test_hlld_has_finite_derivatives_where_it_is_degenerate(name):
    W = degenerate_state()

    def total_flux(x):
        flux = mhd2d.get_flux_hlld(degenerate_state(**{name: x}), W, 2.0)
        return sum(flux.values())

    x0 = {"rho": 1.0, "vx": 0.0, "vy": 0.0, "P_gas": 0.5, "Bx": 2.0, "By": 0.0}[name]
    assert jnp.isfinite(jax.grad(total_flux)(x0))
    assert jnp.isfinite(jax.jit(jax.grad(total_flux))(x0))


def test_hlld_is_consistent_where_it_is_degenerate():
    W = degenerate_state()
    flux = mhd2d.get_flux_hlld(W, W, 2.0)

    # rho vx^2 + P_tot - Bx^2, with everything else at rest
    assert float(flux["momx"]) == pytest.approx(0.5 + 2.0 - 4.0, abs=1e-14)
    for k in ("mass", "momy", "energy", "By"):
        assert float(flux[k]) == pytest.approx(0.0, abs=1e-14)
