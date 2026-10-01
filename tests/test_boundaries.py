"""
Tests for the boundary conditions, parameter validation and the time loop.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import adirondax as adx
from adirondax.hydro.boundary import add_ghost_cells, set_ghost_gradients

SWAP_XY = {"vx": "vy", "vy": "vx", "bx": "by", "by": "bx"}


def make_sim(bc, magnetic=False, res=(16, 24), box=(1.0, 1.5), **physics):
    params = {
        "physics": {"hydro": True, "magnetic": magnetic, **physics},
        "mesh": {
            "resolution": list(res),
            "box_size": list(box),
            "boundary_condition": bc,
        },
        "time": {"span": 0.05, "num_timesteps": 20},
        "hydro": {
            "riemann_solver": "hlld" if magnetic else "llf",
            "slope_limiting": True,
        },
    }
    return adx.Simulation(params)


def initial_state(sim):
    X, Y = sim.mesh
    Lx, Ly = sim.box_size
    u, v = X / Lx, 2.0 * jnp.pi * Y / Ly
    state = {
        "t": 0.0,
        "rho": 1.0 + 0.3 * jnp.exp(-((u - 0.4) ** 2 + (Y / Ly - 0.6) ** 2) / 0.02),
        "vx": 0.2 * jnp.sin(v) * u * (1.0 - u) + 0.05,
        "vy": 0.1 * jnp.cos(2.0 * jnp.pi * u) * u,
        "P": 1.0 + 0.5 * jnp.exp(-((u - 0.5) ** 2 + (Y / Ly - 0.3) ** 2) / 0.01),
    }
    if sim.params["physics"]["magnetic"]:
        state["bx"] = 0.3 + 0.1 * jnp.sin(v)
        state["by"] = 0.2 + 0.1 * jnp.cos(2.0 * jnp.pi * u) * u
    if sim.params["physics"]["rotation"]:
        state["vz"] = 0.1 * u * (1.0 - u) * jnp.sin(v)
        state["bz"] = 0.2 * u * jnp.cos(v)
    return state


def transpose(state):
    return {
        SWAP_XY.get(k, k): (f.T if jnp.ndim(f) == 2 else f) for k, f in state.items()
    }


def copy_edges(state, axis):
    return {
        k: (lax.index_in_dim(f, 0, axis), lax.index_in_dim(f, -1, axis))
        for k, f in state.items()
        if jnp.ndim(f) == 2
    }


@pytest.mark.parametrize(
    "magnetic, bc",
    [(True, "wall"), (True, "outflow"), (False, "reflective"), (False, "outflow")],
)
def test_boundary_along_y_mirrors_boundary_along_x(magnetic, bc):
    sim_x = make_sim([bc, "periodic"], magnetic=magnetic, rotation=magnetic)
    sim_y = make_sim(
        ["periodic", bc],
        magnetic=magnetic,
        rotation=magnetic,
        res=sim_x.resolution[::-1],
        box=sim_x.box_size[::-1],
    )
    sim_x.state.update(initial_state(sim_x))
    sim_y.state.update(transpose(initial_state(sim_x)))
    sim_x.run()
    sim_y.run()

    expected = transpose(sim_x.state)
    for key in expected:
        np.testing.assert_allclose(sim_y.state[key], expected[key], atol=1e-5)


@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_driven_boundary_that_copies_the_edge_is_outflow(magnetic, axis):
    def run(bc):
        bcs = ["periodic", "periodic"]
        bcs[axis] = bc
        sim = make_sim(bcs, magnetic=magnetic, rotation=magnetic)
        sim.state.update(initial_state(sim))
        sim.driven_boundary = copy_edges
        sim.run()
        return sim.state

    outflow = run("outflow")
    driven = run("driven")
    for key in outflow:
        np.testing.assert_array_equal(driven[key], outflow[key])


def test_driven_boundary_needs_a_callback():
    sim = make_sim(["driven", "periodic"])
    sim.state.update(initial_state(sim))
    with pytest.raises(ValueError, match="driven_boundary"):
        sim.run()


def test_axis_boundary_is_a_wall_at_the_outer_edge():
    rng = np.random.default_rng(0)
    W = {k: jnp.asarray(rng.normal(size=(6, 4))) for k in ["rho", "vx", "vz", "bz"]}
    W_d = {k: jnp.asarray(rng.normal(size=(8, 4))) for k in ["rho", "vx", "vz", "Bz"]}

    axis = add_ghost_cells(W, 0, "axis")
    wall = add_ghost_cells(W, 0, "wall")
    axis_d = set_ghost_gradients(W_d, 0, "axis")
    wall_d = set_ghost_gradients(W_d, 0, "wall")

    for k in W:
        np.testing.assert_array_equal(axis[k][-1], wall[k][-1])
    for k in W_d:
        np.testing.assert_array_equal(axis_d[k][-1], wall_d[k][-1])

    # the out-of-plane components reverse sense through R=0
    np.testing.assert_array_equal(axis["vz"][0], -W["vz"][0])
    np.testing.assert_array_equal(axis_d["vz"][0], W_d["vz"][1])


def test_resolution_must_be_even():
    with pytest.raises(ValueError, match="even"):
        make_sim(["periodic", "periodic"], res=(16, 25))


@pytest.mark.parametrize("physics", ["gravity", "external_potential"])
def test_magnetic_with_a_potential_is_rejected_up_front(physics):
    with pytest.raises(NotImplementedError, match=physics):
        make_sim(["periodic", "periodic"], magnetic=True, **{physics: True})


def test_repeated_runs_reuse_the_compiled_loop():
    sim = make_sim(["outflow", "periodic"])
    for _ in range(3):
        sim.state.update(initial_state(sim))
        sim.run()
    advance = sim._advance_cache[1]
    assert advance._cache_size() == 1

    sim.params["time"]["span"] = 0.1
    sim.state.update(initial_state(sim))
    sim.run()
    assert sim._advance_cache[1] is not advance
