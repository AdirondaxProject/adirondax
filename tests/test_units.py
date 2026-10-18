"""
Tests for the code units and the physical constants derived from them.
"""

import contextlib
import io

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import adirondax as adx
from adirondax.constants import code_constants, constants

CGS = {"mass": 1.0, "length": 1.0, "time": 1.0, "temperature": 1.0}
SI = {"mass": 1.0e3, "length": 1.0e2, "time": 1.0, "temperature": 1.0}


def make_sim(**params):
    with contextlib.redirect_stdout(io.StringIO()):
        return adx.Simulation(params)


def test_cgs_units_leave_the_constants_unchanged():
    assert code_constants(CGS) == constants


@pytest.mark.parametrize(
    "name, si_value",
    [
        ("speed_of_light", 2.998e8),  # m s^-1
        ("gravitational_constant", 6.6743e-11),  # m^3 kg^-1 s^-2
        ("planck_constant", 6.6261e-34),  # J s
        ("boltzmann_constant", 1.380649e-23),  # J K^-1
        ("radiation_constant", 7.5657e-16),  # J m^-3 K^-4
        ("electron_volt", 1.602176634e-19),  # J
        ("proton_mass", 1.6726e-27),  # kg
        ("stefan_boltzmann_constant", 5.6704e-8),  # W m^-2 K^-4
        ("gas_constant", 8.3145),  # J mol^-1 K^-1
        ("rydberg_constant", 1.09737316e7),  # m^-1
        ("avogadro_number", 6.02214076e23),
    ],
)
def test_si_units_give_the_si_constants(name, si_value):
    assert code_constants(SI)[name] == pytest.approx(si_value, rel=1e-12)


def test_constants_can_be_set_directly():
    c = code_constants(CGS, {"speed_of_light": 100.0, "proton_mass": None})

    assert c["speed_of_light"] == 100.0
    assert c["proton_mass"] == constants["proton_mass"]


def test_simulation_constants_follow_the_parameters():
    sim = make_sim(
        units={"length": 1.0e2, "mass": 1.0e3},
        constants={"radiation_constant": 2.0},
        hydro={"eos": {"mean_molecular_weight": 0.5}},
    )

    assert sim.constants["speed_of_light"] == pytest.approx(2.998e8)
    assert sim.constants["radiation_constant"] == 2.0
    assert sim.gas_constant == pytest.approx(1.380649e-23 / (0.5 * 1.6726e-27))


def test_gravity_uses_the_code_units():
    def potential(G):
        sim = make_sim(
            physics={"hydro": True, "gravity": True},
            constants={"gravitational_constant": G},
        )
        X, _ = sim.mesh
        sim.state["rho"] = 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * X)
        return sim.potential

    assert jnp.allclose(potential(2.0), 2.0 * potential(1.0))


def test_axion_mass_sets_m_per_hbar():
    # m/hbar = m_eV eV / (c^2 hbar), in s cm^-2 for cgs units
    hbar = 6.6261e-27 / (2.0 * jnp.pi)
    m_per_hbar = 1.0e-22 * 1.602176634e-12 / 2.998e10**2 / hbar

    assert make_sim().m_per_hbar == pytest.approx(m_per_hbar, rel=1e-12)
    assert make_sim(quantum={"m_eV": 2.0e-22}).m_per_hbar == pytest.approx(
        2.0 * m_per_hbar, rel=1e-12
    )

    # in code units of kpc and Gyr, m/hbar scales as time / length^2
    kpc, Gyr = 3.0857e21, 3.15576e16
    sim = make_sim(units={"length": kpc, "time": Gyr})
    assert sim.m_per_hbar == pytest.approx(m_per_hbar * kpc**2 / Gyr, rel=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        {"units": {"length": 0.0}},
        {"units": {"temperature": -1.0}},
        {"constants": {"speed_of_light": -1.0}},
        {"hydro": {"eos": {"mean_molecular_weight": 0.0}}},
        {"quantum": {"m_eV": -1.0}},
    ],
)
def test_invalid_units_are_rejected(params):
    with pytest.raises(ValueError):
        make_sim(**params)
