import jax
import jax.numpy as jnp
import adirondax as adx
from jaxopt import ScipyMinimize
import matplotlib.image as img


def solve_inverse_problem():

    target_data = img.imread("examples/schrodinger_poisson/target.png")[:, :, 0]
    rho_target = jnp.flipud(jnp.array(target_data, dtype=float))
    rho_target = 1.0 - 0.5 * (rho_target - 0.5)
    rho_target /= jnp.mean(rho_target)

    n = rho_target.shape[0]
    nt = 100 * int(n / 128)
    t_stop = 0.03
    dt = t_stop / nt
    params = {
        "n": n,
        "t_stop": t_stop,
        "dt": dt,
        "nt": nt,
    }

    sim = adx.Simulation(params)

    @jax.jit
    def loss_function(theta, rho_target):
        psi = jnp.exp(1.0j * theta)
        psi = sim.evolve(psi, dt, nt)
        rho = jnp.abs(psi) ** 2
        return jnp.mean((rho - rho_target) ** 2)

    opt = ScipyMinimize(method="l-bfgs-b", fun=loss_function, tol=1e-5)
    theta = jnp.zeros_like(rho_target)
    sol = opt.run(theta, rho_target)
    theta = jnp.mod(sol.params, 2.0 * jnp.pi) - jnp.pi

    return jnp.mean(theta)


def test_solve_inverse_problem():
    assert abs(solve_inverse_problem() - 0.019558249) < 1e-5
