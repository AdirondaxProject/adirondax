import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres

from .planck import group_emission

# Pure functions for implicit, multigroup, discrete ordinates radiation transport
# in 2D (x,y) and axisymmetric (R,z) geometry
#
# Follows Jiang (2021, ApJS 253, 49) and its multigroup extension Jiang (2022,
# arXiv:2209.06240). The specific intensities I[f, n, i, j] of each frequency
# group f and direction n are evolved in the lab frame, in units of energy
# density per steradian, so that the radiation energy density is
# E_r = 4 pi sum_n w_n I_n. The transport and the gas-radiation coupling are
# solved together, implicitly, with backward Euler; the coupling is evaluated in
# the fluid rest frame via a Lorentz transformation. The gas receives whatever
# energy and momentum the radiation loses to the coupling, so the total of both
# is conserved to round-off.
#
# The gas density and velocity are held fixed while the radiation is solved for,
# which assumes the inertia of the radiation, E_r/c^2, is small next to that of
# the gas, rho (as in any nonrelativistic flow); otherwise the momentum handed
# back to the gas overshoots, and grows from step to step.
#
# Not yet included: the Doppler shift of photons between frequency groups (the
# M_nu mapping of Jiang 2022, exact for one group), and Compton scattering.

_MIRRORED = ("reflective", "wall", "axis")


def _dir(angles, name):
    """A per-direction quantity, shaped to broadcast against (group, dir, x, y)"""

    return jnp.asarray(angles[name])[None, :, None, None]


def _pad_intensity(I, axis, bc, angles, ghost=None):
    """
    Pad the intensities with one ghost cell on each side of a spatial axis.

    'outflow' is a vacuum: outgoing directions are copied outwards and nothing
    comes back in. 'reflective', 'wall' and 'axis' mirror the directions.
    'driven' takes the incoming intensities from ghost = (lo, hi), and is a
    vacuum if ghost is None.
    """

    ax = axis + 2
    if bc == "periodic":
        lo = lax.index_in_dim(I, -1, ax)
        hi = lax.index_in_dim(I, 0, ax)
        return jnp.concatenate((lo, I, hi), axis=ax)

    lo = lax.index_in_dim(I, 0, ax)
    hi = lax.index_in_dim(I, -1, ax)
    if bc in _MIRRORED:
        mirror = angles["reflect_x"] if axis == 0 else angles["reflect_y"]
        lo, hi = lo[:, mirror], hi[:, mirror]
    elif bc in ("outflow", "driven"):
        mu = _dir(angles, "mu_x" if axis == 0 else "mu_y")
        lo_in, hi_in = (0.0, 0.0) if ghost is None else ghost
        lo = jnp.where(mu < 0.0, lo, lo_in)
        hi = jnp.where(mu > 0.0, hi, hi_in)
    else:
        raise ValueError(f"unknown boundary condition: '{bc}'")

    return jnp.concatenate((lo, I, hi), axis=ax)


def _pad_field(f, axis, bc, odd=False):
    """Pad a cell-centered field (..., x, y) with one ghost cell on each side"""

    ax = f.ndim - 2 + axis
    if bc == "periodic":
        lo = lax.index_in_dim(f, -1, ax)
        hi = lax.index_in_dim(f, 0, ax)
    else:
        lo = lax.index_in_dim(f, 0, ax)
        hi = lax.index_in_dim(f, -1, ax)
        if odd and bc in _MIRRORED:
            lo, hi = -lo, -hi

    return jnp.concatenate((lo, f, hi), axis=ax)


def _face_avg(f, axis):
    """Average a padded field (..., x, y) onto the faces between its cells"""

    ax = f.ndim - 2 + axis
    n = f.shape[ax]
    return 0.5 * (
        lax.slice_in_dim(f, 0, n - 1, axis=ax) + lax.slice_in_dim(f, 1, n, axis=ax)
    )


def _face_coefficients(sigma_t, v, mu, c, dx, tau_factor, axis, bc):
    """
    Coefficients of the modified HLLE flux of Jiang (2021, eq. A3) on the faces
    along an axis, F = a_L I_L + a_R I_R, which is upwind where a cell is
    optically thin and turns into centered differencing plus a small diffusion
    where it is optically thick, so that it recovers the diffusion limit.

    Also returns f*v on the faces, the part of the flow velocity whose
    advection of photons is taken out of the implicit flux and done explicitly
    (Jiang 2021, eq. 18).
    """

    # the face is as opaque as the more transparent of its two cells, so that
    # the flux stays upwind (and positive) between optically thin gas and an
    # optically thick body, and turns diffusive only inside the body
    sigma_t = _pad_field(sigma_t, axis, bc)
    ax = sigma_t.ndim - 2 + axis
    n = sigma_t.shape[ax]
    sigma_face = jnp.minimum(
        lax.slice_in_dim(sigma_t, 0, n - 1, axis=ax),
        lax.slice_in_dim(sigma_t, 1, n, axis=ax),
    )[:, None]
    v_face = _face_avg(_pad_field(v, axis, bc, odd=True), axis)[None, None]

    # optical depth per cell, guarded against tau = 0 (with a margin that keeps
    # the square roots below differentiable in single precision)
    tau_sq = jnp.maximum((tau_factor * sigma_face * dx) ** 2, 1.0e-12)
    f_adv = -jnp.expm1(-tau_sq)

    # the signal speeds (eqs. A4, A5): 'up' is the speed on the upwind side,
    # c|mu| when optically thin and c|mu|/tau when thick, and 'down' the speed on
    # the other side, which goes to zero when optically thin
    up = c * jnp.abs(mu) * jnp.sqrt(-jnp.expm1(-tau_sq) / tau_sq)
    down = c * jnp.abs(mu) * jnp.sqrt(-jnp.expm1(-(tau_sq**2)) / tau_sq)
    s_plus = jnp.where(mu > 0.0, up, down)
    s_minus = jnp.where(mu > 0.0, -down, -up)

    u = c * mu - f_adv * v_face
    denom = s_plus - s_minus
    a_L = (s_plus * u - s_plus * s_minus) / denom
    a_R = (s_plus * s_minus - s_minus * u) / denom

    return a_L, a_R, f_adv * v_face


def _divergence(F_x, F_y, geom):
    """Volume-averaged divergence of the face fluxes F_x (x+1 faces) and F_y (y+1)"""

    vol = geom["vol"]
    A_x = _face_areas_x(geom)
    A_y = geom["area_y"]
    AF_x = A_x * F_x
    AF_y = A_y * F_y
    return (
        AF_x[..., 1:, :] - AF_x[..., :-1, :] + AF_y[..., :, 1:] - AF_y[..., :, :-1]
    ) / vol


def _face_areas_x(geom):
    """Areas of all the x-faces, from the lower face of the first cell up"""

    if not geom["is_cylindrical"]:
        return jnp.asarray(geom["area_x"])
    area_lo = geom["area_x"] - geom["d_area_x"]
    return jnp.concatenate((area_lo[:1], geom["area_x"]), axis=0)


def _outflow_diagonal(a_L, a_R, A_lo, A_hi, mu, axis, bc):
    """
    The flux through an 'outflow' or 'driven' boundary that the edge cell sends
    itself, via the outgoing intensities its ghost cell copies. Keeping it on
    the diagonal matters where the edge is optically thick, where the copy is
    large and lagging it makes the Jacobi iteration diverge.
    """

    if bc not in ("outflow", "driven"):
        return 0.0

    ax = axis + 2
    lo = -A_lo * lax.index_in_dim(a_L, 0, ax) * (mu < 0.0)
    hi = A_hi * lax.index_in_dim(a_R, -1, ax) * (mu > 0.0)
    middle = jnp.zeros_like(lax.slice_in_dim(a_L, 2, -1, axis=ax))
    return jnp.concatenate((lo, middle, hi), axis=ax)


def _minmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _advect_explicit(I, fv, axis, bc, angles, ghost):
    """
    Upwind flux of I with the face velocity fv along an axis, from a
    slope-limited linear reconstruction (ghost cells get flat slopes)
    """

    Ip = _pad_intensity(I, axis, bc, angles, ghost)
    ax = axis + 2
    n = Ip.shape[ax]

    def sl(a, b):
        return lax.slice_in_dim(Ip, a, n + b, axis=ax)

    slope = _minmod(sl(1, -1) - sl(0, -2), sl(2, 0) - sl(1, -1))
    zero = jnp.zeros_like(lax.index_in_dim(slope, 0, ax))
    slope = jnp.concatenate((zero, slope, zero), axis=ax)

    left = sl(0, -1) + 0.5 * lax.slice_in_dim(slope, 0, n - 1, axis=ax)
    right = sl(1, 0) - 0.5 * lax.slice_in_dim(slope, 1, n, axis=ax)
    return fv * jnp.where(fv > 0.0, left, right)


def _solve_temperature(phi, T_lo, T_hi, T_guess, num_iter=50):
    """
    Root of the increasing function phi(T) in [T_lo, T_hi], cell by cell, by
    Newton's method falling back to bisection whenever a step leaves the
    bracket
    """

    def body(_, carry):
        T, lo, hi = carry
        f, df = jax.jvp(phi, (T,), (jnp.ones_like(T),))
        lo = jnp.where(f < 0.0, T, lo)
        hi = jnp.where(f < 0.0, hi, T)
        T_new = T - f / df
        bad = ~jnp.isfinite(T_new) | (T_new < lo) | (T_new > hi)
        return jnp.where(bad, 0.5 * (lo + hi), T_new), lo, hi

    T = jnp.clip(T_guess, T_lo, T_hi)
    T, _, _ = lax.fori_loop(0, num_iter, body, (T, T_lo, T_hi))
    return T


def _local_solve(I_c, g1, coef):
    """
    Solve the implicit equations of every cell for the intensities of all its
    groups and directions, and the gas temperature, with the intensities of the
    neighbouring cells (and of the other directions, in the angular transport)
    held fixed in I_c (Jiang 2022, eqs. 28-33).

    In the fluid frame, with the frame factor Gam = gamma (1 - n.v/c) and
    k = c dt, each direction obeys
      (g1 + k Gam sigma_t) I0 = Gam^4 I_c + k Gam [sigma_t J0 + sigma_p (eps - J0)]
    which ties every I0 to the mean intensity J0 of its group; summing over
    directions gives J0 (1 - A1) = A2 eps + A3, and the gas energy equation
    leaves a scalar equation for T.
    """

    Gam, w_co = coef["Gam"], coef["w_co"]
    sigma_t, sigma_p = coef["sigma_t"], coef["sigma_p"]
    k = coef["k"]

    D = g1 + k * Gam * sigma_t[:, None]
    wG_D = jnp.sum(w_co * Gam / D, axis=1)
    A1 = wG_D * k * (sigma_t - sigma_p)
    A2 = wG_D * k * sigma_p
    A3 = jnp.sum(w_co * Gam**4 * I_c / D, axis=1)
    one_A1 = 1.0 - A1

    # the gas energy equation, phi(T) = 0
    p = {
        "A2": A2,
        "A3": A3,
        "one_A1": one_A1,
        "sigma_p": sigma_p,
        "k": k,
        "C_v": coef["C_v"],
        "T_m": coef["T_m"],
    }

    def phi_of(T, p):
        eps = group_emission(T, coef["group_edges"], coef["a_r"])
        exchange = (p["one_A1"] - p["A2"]) * eps - p["A3"]
        exchange = jnp.sum(p["sigma_p"] * exchange / p["one_A1"], axis=0)
        return p["C_v"] * (T - p["T_m"]) + 4.0 * jnp.pi * p["k"] * exchange

    # The emission rises with T, so the root lies below the temperature the gas
    # would reach by absorbing everything and emitting nothing. The bracketed
    # Newton iteration carries no derivatives; one more Newton step from its
    # root gives the derivatives of the root by the implicit function theorem.
    ps = jax.tree.map(lax.stop_gradient, p)
    absorbed = 4.0 * jnp.pi * ps["k"] * ps["sigma_p"] * ps["A3"] / ps["one_A1"]
    absorbed = jnp.maximum(jnp.sum(absorbed, axis=0), 0.0)
    T_hi = ps["T_m"] + absorbed / ps["C_v"]
    T_lo = 1.0e-6 * ps["T_m"]
    T = _solve_temperature(lambda T: phi_of(T, ps), T_lo, T_hi, ps["T_m"])

    f, df = jax.jvp(lambda T: phi_of(T, p), (T,), (jnp.ones_like(T),))
    T = T - f / lax.stop_gradient(df)

    eps = group_emission(T, coef["group_edges"], coef["a_r"])
    J0 = (A2 * eps + A3) / one_A1
    source = k * (sigma_t * J0 + sigma_p * (eps - J0))
    I0 = (Gam**4 * I_c + Gam * source[:, None]) / D

    return I0 / Gam**4, T


def _norm1(x):
    return jnp.sum(jnp.abs(x))


def _converged(update, x, x0, tol):
    """
    Whether the last update of the intensities is small next to their change
    over the step, so that a field changing slowly is still followed (rather
    than accepted as is), down to round-off
    """

    floor = 100.0 * jnp.finfo(x.dtype).eps * _norm1(x)
    return _norm1(update) <= tol * _norm1(x - x0) + floor


def _jacobi(f, x0, tol, max_iter):
    """The Jacobi-like iteration of Jiang (2021): x <- G(x) = x - f(x)"""

    def cond(carry):
        x, dx, it = carry
        return ~_converged(dx, x, x0, tol) & (it < max_iter)

    def body(carry):
        x, _, it = carry
        dx = -f(x)
        return x + dx, dx, it + 1

    x, _, _ = lax.while_loop(cond, body, body((x0, None, 0)))
    return x


def _newton_krylov(f, x0, tol, max_iter, restart, max_restarts):
    """
    Newton's method on f(x) = x - G(x), each step solved by GMRES. The cell-by-cell
    solve inside G preconditions the system, and the nonlinearity is local (the
    gas temperature), so a step or two usually suffices.
    """

    def cond(carry):
        x, r, it = carry
        return ~_converged(r, x, x0, tol) & (it < max_iter)

    def body(carry):
        x, r, it = carry
        _, f_lin = jax.linearize(f, x)
        atol = 100.0 * jnp.finfo(x.dtype).eps * jnp.linalg.norm(x)
        dx, _ = gmres(
            f_lin, -r, tol=tol, atol=atol, restart=restart, maxiter=max_restarts
        )
        x = x + dx
        return x, f(x), it + 1

    x, _, _ = lax.while_loop(cond, body, body((x0, f(x0), 0)))
    return x


def _tangent_solve(g, y, tol, restart, max_restarts):
    """Solve g(x) = y for the linearized system, in a way that can be transposed"""

    def solve(matvec, b):
        return gmres(matvec, b, tol=tol, restart=restart, maxiter=max_restarts)[0]

    return lax.custom_linear_solve(g, y, solve, transpose_solve=solve)


def radiation_moments(I, angles, c):
    """
    The lab-frame energy density 'E' and in-plane flux 'Fx', 'Fy' of each
    frequency group, each of shape (num_groups, nx, ny)
    """

    w = _dir(angles, "w")
    return {
        "E": 4.0 * jnp.pi * jnp.sum(w * I, axis=1),
        "Fx": 4.0 * jnp.pi * c * jnp.sum(w * _dir(angles, "mu_x") * I, axis=1),
        "Fy": 4.0 * jnp.pi * c * jnp.sum(w * _dir(angles, "mu_y") * I, axis=1),
    }


def radiation_rt2d_step(
    W,
    I,
    geom,
    dt,
    *,
    angles,
    opacity,
    group_edges,
    c,
    a_r,
    gas_constant,
    gamma,
    bc_x="periodic",
    bc_y="periodic",
    ghost_x=None,
    ghost_y=None,
    tau_factor=5.0,
    solver="gmres",
    tol=1.0e-4,
    max_iter=1000,
    restart=30,
    max_restarts=100,
):
    """
    Advance the radiation field implicitly by dt, and exchange energy and
    momentum with the gas.

    Parameters
    ----------
    W: dict
      Primitive gas fields; uses 'rho', 'vx', 'vy', 'P' and updates 'vx', 'vy'
      and 'P'. The gas temperature is T = P / (rho * gas_constant).
    I: jax.Array
      Lab-frame specific intensities, shape (num_groups, num_dirs, nx, ny).
    geom: dict
      Mesh metric factors on the bare grid (no ghost cells), from get_geometry.
    dt: float
      Timestep.
    angles: dict
      The angular grid, from get_angles.
    opacity: dict
      Absorption coefficients rho*kappa, held fixed over the step:
      'sigma_s' (scattering), 'sigma_a' (Rosseland mean absorption) and
      'sigma_p' (Planck mean absorption), each broadcastable to
      (num_groups, nx, ny).
    group_edges: sequence
      Interior frequency group boundaries h nu / k_B, in temperature units.
      Empty for grey radiation.
    c, a_r: float
      Speed of light and radiation constant.
    gas_constant: float
      k_B / (mean molecular weight * m_p).
    gamma: float
      Adiabatic index.
    bc_x, bc_y: str
      Boundary conditions. A 'driven' boundary takes the intensities entering
      the domain from ghost_x/ghost_y = (lo, hi), each of shape
      (num_groups, num_dirs, 1, ny) or (num_groups, num_dirs, nx, 1).
    tau_factor: float
      Scales the optical depth per cell that switches the face fluxes from
      upwind to diffusive (alpha of Jiang 2021, eq. A5).
    solver: str
      'gmres' (Newton-Krylov) or 'jacobi' (the iteration of Jiang 2021).
    tol: float
      Tolerance of the iteration: it stops once its last update of the
      intensities, summed over the grid, is this fraction of their change over
      the step.
    max_iter: int
      Maximum number of Jacobi or Newton iterations.
    restart: int
      GMRES restart length.
    max_restarts: int
      Maximum number of GMRES restarts per linear solve.

    Returns
    -------
    W: dict
      The gas fields after the energy and momentum exchange.
    I: jax.Array
      The intensities at the end of the step.
    """

    num_groups = len(group_edges) + 1
    if I.shape[0] != num_groups:
        raise ValueError(
            f"intensities have {I.shape[0]} groups but group_edges gives {num_groups}"
        )
    shape = (num_groups,) + I.shape[2:]

    rho, vx, vy, P = W["rho"], W["vx"], W["vy"], W["P"]
    sigma_s = jnp.broadcast_to(opacity["sigma_s"], shape)
    sigma_a = jnp.broadcast_to(opacity["sigma_a"], shape)
    sigma_p = jnp.broadcast_to(opacity["sigma_p"], shape)
    sigma_t = sigma_s + sigma_a

    mu_x, mu_y = _dir(angles, "mu_x"), _dir(angles, "mu_y")

    # implicit flux coefficients, and the explicit advection with f*v
    a_Lx, a_Rx, fv_x = _face_coefficients(
        sigma_t, vx, mu_x, c, geom["dx"], tau_factor, 0, bc_x
    )
    a_Ly, a_Ry, fv_y = _face_coefficients(
        sigma_t, vy, mu_y, c, geom["dy"], tau_factor, 1, bc_y
    )

    def transport(I):
        Ip = _pad_intensity(I, 0, bc_x, angles, ghost_x)
        F_x = a_Lx * Ip[..., :-1, :] + a_Rx * Ip[..., 1:, :]
        Ip = _pad_intensity(I, 1, bc_y, angles, ghost_y)
        F_y = a_Ly * Ip[..., :-1] + a_Ry * Ip[..., 1:]
        return dt * _divergence(F_x, F_y, geom)

    # the part of the transport that acts on each cell's own intensities
    nx = I.shape[2]
    A_x = jnp.broadcast_to(_face_areas_x(geom), (nx + 1, 1))
    A_y, vol = geom["area_y"], geom["vol"]
    diag = (
        A_x[1:] * a_Lx[..., 1:, :]
        - A_x[:-1] * a_Rx[..., :-1, :]
        + A_y * (a_Ly[..., 1:] - a_Ry[..., :-1])
        + _outflow_diagonal(a_Lx, a_Rx, A_x[:1], A_x[-1:], mu_x, 0, bc_x)
        + _outflow_diagonal(a_Ly, a_Ry, A_y, A_y, mu_y, 1, bc_y)
    )
    diag = dt * diag / vol

    # in axisymmetry the local basis turns as a ray crosses the mesh, which
    # moves photons from psi towards 0 (the radially outward direction) at the
    # rate c sin(zeta) sin(psi) / R; it is upwinded in psi, with 1/R taken as
    # d(area_x)/vol to balance the radial flux of an isotropic field exactly
    if geom["is_cylindrical"]:
        n_z, n_p = angles["num_polar"], angles["num_azimuthal"]
        rate = (
            c
            * dt
            * _dir(angles, "sin_zeta")
            * (geom["d_area_x"] / vol)
            / angles["d_psi"]
        )
        rate_hi = rate * _dir(angles, "psi_face_hi")
        rate_lo = rate * _dir(angles, "psi_face_lo")

        def angular(I):
            Ik = I.reshape(I.shape[:1] + (n_z, n_p) + I.shape[2:])
            I_next = jnp.roll(Ik, -1, axis=2).reshape(I.shape)
            return rate_lo * I - rate_hi * I_next

        diag = diag + rate_lo
    else:

        def angular(I):
            return 0.0

    g1 = 1.0 + diag

    def off_diagonal(I):
        return transport(I) + angular(I) - diag * I

    # the part of the right-hand side known from the start of the step
    F_x = _advect_explicit(I, fv_x, 0, bc_x, angles, ghost_x)
    F_y = _advect_explicit(I, fv_y, 1, bc_y, angles, ghost_y)
    rhs = I - dt * _divergence(F_x, F_y, geom)

    # frame transformation, with the fluid-frame quadrature weights normalized
    # (Jiang 2021, eq. 12)
    beta_x, beta_y = vx / c, vy / c
    lorentz = 1.0 / jnp.sqrt(1.0 - beta_x**2 - beta_y**2)
    Gam = lorentz * (1.0 - mu_x[0] * beta_x - mu_y[0] * beta_y)
    w_co = _dir(angles, "w")[0] / Gam**2
    w_co = w_co / jnp.sum(w_co, axis=0)

    coef = {
        "Gam": Gam,
        "w_co": w_co,
        "sigma_t": sigma_t,
        "sigma_p": sigma_p,
        "k": c * dt,
        "T_m": P / (rho * gas_constant),
        "C_v": rho * gas_constant / (gamma - 1.0),
        "group_edges": tuple(group_edges),
        "a_r": a_r,
    }

    def residual(I):
        I_new, _ = _local_solve(rhs - off_diagonal(I), g1, coef)
        return I - I_new

    if solver == "gmres":

        def solve(f, x0):
            return _newton_krylov(f, x0, tol, max_iter, restart, max_restarts)

    elif solver == "jacobi":

        def solve(f, x0):
            return _jacobi(f, x0, tol, max_iter)

    else:
        raise ValueError(f"unknown radiation solver: '{solver}'")

    I_new = lax.custom_root(
        residual,
        I,
        solve,
        lambda g, y: _tangent_solve(g, y, tol, restart, max_restarts),
    )

    # what the radiation lost to the gas is its change less what was transported
    collision = I_new + transport(I_new) + angular(I_new) - rhs
    w = _dir(angles, "w")
    dE = -4.0 * jnp.pi * jnp.sum(w * collision, axis=(0, 1))
    dpx = -(4.0 * jnp.pi / c) * jnp.sum(w * mu_x * collision, axis=(0, 1))
    dpy = -(4.0 * jnp.pi / c) * jnp.sum(w * mu_y * collision, axis=(0, 1))

    vx_new = vx + dpx / rho
    vy_new = vy + dpy / rho
    d_kinetic = 0.5 * rho * (vx_new**2 + vy_new**2 - vx**2 - vy**2)
    P_new = P + (gamma - 1.0) * (dE - d_kinetic)

    return {**W, "vx": vx_new, "vy": vy_new, "P": P_new}, I_new
