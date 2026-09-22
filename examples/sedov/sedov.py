import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import adirondax as adx
from adirondax.hydro.geometry import get_geometry

"""
Simulate a Sedov-Taylor blast wave on an axisymmetric cylindrical (R,z) mesh

The explosion is spherical, so the solution must remain spherical even though
the mesh is not: this is a direct check of the geometric source term and of the
treatment of the R=0 axis.

Philip Mocz (2026)
"""


# Initial condition: a hot sphere of radius R_BLAST in a uniform ambient medium
RHO_AMBIENT = 1.0
P_AMBIENT = 0.1
P_BLAST = 100.0
R_BLAST = 0.05
GAMMA = 5.0 / 3.0


def set_up_simulation():
    # Define the parameters for the simulation
    nR = 256
    nz = 512
    nt = 800
    t_stop = 0.05

    params = {
        "physics": {
            "hydro": True,
        },
        "mesh": {
            "geometry": "cylindrical",
            "resolution": [nR, nz],
            "box_size": [0.5, 1.0],
            "boundary_condition": ["axis", "reflective"],
        },
        "time": {
            "span": t_stop,
            "num_timesteps": nt,
        },
        "output": {
            "num_checkpoints": 100,
            "save": True,
            "plot_dynamic_range": 10.0,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": GAMMA},
            "slope_limiting": True,
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions: a hot sphere at the origin of the (R,z) plane
    sim.state["t"] = 0.0
    R, z = sim.mesh
    r_sph = jnp.sqrt(R**2 + (z - 0.5) ** 2)

    sim.state["rho"] = RHO_AMBIENT * jnp.ones(R.shape)
    sim.state["vx"] = jnp.zeros(R.shape)
    sim.state["vy"] = jnp.zeros(R.shape)
    sim.state["P"] = jnp.where(r_sph < R_BLAST, P_BLAST, P_AMBIENT)

    vol = get_geometry("cylindrical", sim.box_size, sim.resolution)["vol"]
    sim.blast_energy = float(
        jnp.sum((sim.state["P"] - P_AMBIENT) / (GAMMA - 1.0) * vol)
    )

    return sim


_SELFSIMILAR_CACHE = {}


def sedov_selfsimilar(gamma, nu=3, lam_min=1.0e-4, n=20001):
    """
    Self-similar Sedov-Taylor solution.
    """
    key = (gamma, nu, lam_min, n)
    if key in _SELFSIMILAR_CACHE:
        return _SELFSIMILAR_CACHE[key]

    alpha = 2.0 / (nu + 2.0)

    # strong-shock (Rankine-Hugoniot) values at lam = 1
    y = np.array(
        [
            2.0 * alpha / (gamma + 1.0),  # V
            (gamma + 1.0) / (gamma - 1.0),  # G
            2.0 * alpha**2 / (gamma + 1.0),  # P
        ]
    )

    def dydx(y):
        V, G, P = y
        D = V - alpha
        dV = (
            P * (2.0 - 2.0 * V - nu * gamma * V + 2.0 * D) + D * G * V * (V - 1.0)
        ) / (gamma * P - G * D * D)
        dP = -G * V * (V - 1.0) - 2.0 * P - G * D * dV
        dG = -G * (nu * V + dV) / D
        return np.array([dV, dG, dP])

    x = np.linspace(0.0, np.log(lam_min), n)
    h = x[1] - x[0]
    Y = np.empty((n, 3))
    Y[0] = y
    for i in range(n - 1):
        k1 = dydx(y)
        k2 = dydx(y + 0.5 * h * k1)
        k3 = dydx(y + 0.5 * h * k2)
        k4 = dydx(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        Y[i + 1] = y

    lam = np.exp(x)[::-1]
    V, G, P = Y[::-1, 0], Y[::-1, 1], Y[::-1, 2]

    # energy normalization: 1 = S_nu * xi0^(nu+2) * integral
    S_nu = {1: 2.0, 2: 2.0 * np.pi, 3: 4.0 * np.pi}[nu]
    integrand = lam ** (nu + 1) * (0.5 * G * V**2 + P / (gamma - 1.0))
    xi0 = (1.0 / (S_nu * np.trapezoid(integrand, lam))) ** (1.0 / (nu + 2.0))

    _SELFSIMILAR_CACHE[key] = (lam, G, xi0)
    return lam, G, xi0


def sedov_density(r, t, energy, rho0, gamma, nu=3):
    """Analytic density profile of a Sedov-Taylor blast, and the shock radius"""

    lam, G, xi0 = sedov_selfsimilar(gamma, nu)
    r_shock = xi0 * (energy * t**2 / rho0) ** (1.0 / (nu + 2.0))
    x = np.asarray(r) / r_shock
    rho = np.where(x <= 1.0, rho0 * np.interp(x, lam, G, left=0.0), rho0)
    return rho, r_shock


def make_plot(sim):
    # Radial density profile
    R, z = sim.mesh
    r_sph = np.asarray(jnp.sqrt(R**2 + (z - 0.5) ** 2)).ravel()
    rho = np.asarray(sim.state["rho"]).ravel()
    t = float(sim.state["t"])

    r_exact = np.linspace(1.0e-4, 1.6 * float(np.max(r_sph)), 2000)
    rho_exact, r_shock = sedov_density(r_exact, t, sim.blast_energy, RHO_AMBIENT, GAMMA)

    plt.figure(figsize=(6, 4), dpi=80)
    plt.scatter(
        r_sph, rho, s=1.0, color="tab:blue", alpha=0.3, linewidths=0, label="simulation"
    )
    plt.plot(r_exact, rho_exact, "k-", lw=1.5, label="Sedov-Taylor")
    plt.axvline(r_shock, color="tab:red", ls=":", lw=1.0, label="analytic shock")

    plt.xlim(0.0, 2.0 * r_shock)
    plt.ylim(0.0, 1.1 * (GAMMA + 1.0) / (GAMMA - 1.0))
    plt.xlabel("r")
    plt.ylabel("rho")
    plt.title(f"Sedov-Taylor blast wave at t = {t:.3f}")
    plt.legend(loc="upper left", framealpha=1.0, markerscale=8)
    plt.tight_layout()
    plt.savefig("output.png", dpi=240)
    plt.show()


def main():
    sim = set_up_simulation()

    # Evolve the system
    t0 = time.time()
    sim.run()
    print("Run time (s): ", time.time() - t0)
    print("Steps taken:", sim.steps_taken)

    make_plot(sim)


if __name__ == "__main__":
    main()
