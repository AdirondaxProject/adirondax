import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import cg, gmres

from .planck import group_emission
from .rt2d import _MIRRORED, _converged, _face_areas_x, _solve_temperature

# Pure functions for implicit, multigroup, flux-limited radiation diffusion in 2D
# (x,y) and axisymmetric (R,z) geometry
#
# Evolves the radiation energy density E[f, i, j] of each frequency group f by
#   dE/dt = div(D grad E) + c sigma_p (4 pi eps(T) - E)
# with the flux-limited diffusion coefficient D = c lambda(R) / sigma_t, where
# R = |grad E| / (sigma_t E) (Levermore & Pomraning 1981). The diffusion,
# including the flux limiter, and the exchange of energy with the gas are
# solved together, implicitly, with backward Euler.
# The gas receives whatever energy the radiation loses to the exchange, so the
# total of both is conserved to round-off.
#
# In a moving gas, the radiation is also carried along with it, and pushes on
# it with the force -lambda grad E, to first order in v/c (explicitly, before
# the implicit step): in the diffusion limit, lambda = 1/3, this is a radiation
# fluid of pressure E/3, and in the free-streaming limit the force vanishes.
#
# Not yet included: the Doppler shift of photons between frequency groups.

LIMITERS = ("levermore_pomraning", "larsen", "max", "none")
FACE_AVERAGES = ("harmonic", "arithmetic")


def _diffusion_coefficient(sigma, g, c, limiter):
    """
    The flux-limited diffusion coefficient D = c lambda(R) / sigma, with
    R = g / sigma and g = |grad E| / E, written so as to stay finite when
    sigma = 0 (free streaming, D = c / g)
    """

    tiny = jnp.finfo(jnp.result_type(sigma, g)).tiny
    if limiter == "levermore_pomraning":
        # lambda = (2 + R) / (6 + 3R + R^2)
        D = (2.0 * sigma + g) / jnp.maximum(
            6.0 * sigma**2 + 3.0 * sigma * g + g**2, tiny
        )
    elif limiter == "larsen":
        # lambda = (9 + R^2)^(-1/2)
        D = 1.0 / jnp.sqrt(jnp.maximum(9.0 * sigma**2 + g**2, tiny))
    elif limiter == "max":
        # lambda = 1 / max(3, R)
        D = 1.0 / jnp.maximum(jnp.maximum(3.0 * sigma, g), tiny)
    elif limiter == "none":
        # plain diffusion, lambda = 1/3
        D = 1.0 / jnp.maximum(3.0 * sigma, tiny)
    else:
        raise ValueError(f"unknown flux limiter: '{limiter}'")

    return c * D


def _face_coefficient(s_lo, s_hi, g, c, limiter, face_average):
    """
    The diffusion coefficient D and the flux limiter lambda = sigma D / c on a
    face between cells of opacities s_lo and s_hi. The face opacity is either
    their mean ('harmonic': the harmonic mean of the cells' diffusion
    coefficients, right for steady flow across a material interface) or their
    harmonic mean ('arithmetic': the arithmetic mean of the cells' diffusion
    coefficients, right for a heat front in one material, where the opacity of
    the cold side would otherwise choke the flux). The flux limiter then acts
    on the face opacity.
    """

    if face_average == "harmonic":
        sigma = 0.5 * (s_lo + s_hi)
    elif face_average == "arithmetic":
        tiny = jnp.finfo(jnp.result_type(s_lo, s_hi)).tiny
        sigma = 2.0 * s_lo * s_hi / jnp.maximum(s_lo + s_hi, tiny)
    else:
        raise ValueError(f"unknown face average: '{face_average}'")
    D = _diffusion_coefficient(sigma, g, c, limiter)
    return D, sigma * D / c


def _pad(f, axis, bc, ghost=None, odd=False):
    """
    Pad a field (..., x, y) with one ghost cell on each side of an axis: the
    periodic image, a mirror image (odd for the normal velocity), or the given
    ghost values (zero by default) at 'outflow' and 'driven' boundaries
    """

    ax = f.ndim - 2 + axis
    if bc == "periodic":
        lo = lax.index_in_dim(f, -1, ax)
        hi = lax.index_in_dim(f, 0, ax)
    elif bc in _MIRRORED:
        lo = lax.index_in_dim(f, 0, ax)
        hi = lax.index_in_dim(f, -1, ax)
        if odd:
            lo, hi = -lo, -hi
    elif bc in ("outflow", "driven"):
        if ghost is None:
            lo = jnp.zeros_like(lax.index_in_dim(f, 0, ax))
            hi = jnp.zeros_like(lo)
        else:
            lo, hi = ghost
            lo = jnp.broadcast_to(lo, lax.index_in_dim(f, 0, ax).shape)
            hi = jnp.broadcast_to(hi, lo.shape)
    else:
        raise ValueError(f"unknown boundary condition: '{bc}'")

    return jnp.concatenate((lo, f, hi), axis=ax)


def _sides(f, axis):
    """The cells below and above each face of a padded field (..., x, y)"""

    ax = f.ndim - 2 + axis
    n = f.shape[ax]
    return lax.slice_in_dim(f, 0, n - 1, axis=ax), lax.slice_in_dim(f, 1, n, axis=ax)


def _set_ends(f, axis, lo, hi):
    """Replace the first and last face values of a face field along an axis"""

    if axis == 0:
        return f.at[..., 0, :].set(lo).at[..., -1, :].set(hi)
    return f.at[..., 0].set(lo).at[..., -1].set(hi)


def _conductance(E, sigma_t, dx, c, limiter, axis, bc, ghost, face_average):
    """
    The conductance k of each face along an axis, such that the flux across it
    is k (E_below - E_above). At an 'outflow' or 'driven' boundary the half
    cell is in series with the Marshak condition, which lets the radiation
    leave at c E / 2 (and come in from the ghost energy density); a mirrored
    boundary has no flux.
    """

    E_lo, E_hi = _sides(_pad(E, axis, bc, ghost), axis)
    s_lo, s_hi = _sides(_pad_edges(sigma_t, axis, bc), axis)

    tiny = jnp.finfo(E.dtype).tiny
    E_face = 0.5 * (jnp.abs(E_lo) + jnp.abs(E_hi))
    g = jnp.abs(E_hi - E_lo) / jnp.maximum(dx * E_face, tiny)
    k = _face_coefficient(s_lo, s_hi, g, c, limiter, face_average)[0] / dx

    if bc in _MIRRORED:
        zero = jnp.zeros_like(lax.index_in_dim(k, 0, axis + 1, keepdims=False))
        k = _set_ends(k, axis, zero, zero)
    elif bc in ("outflow", "driven"):
        # the gradient across the half cell to the boundary is twice as steep
        D_b = _face_coefficient(s_lo, s_hi, 2.0 * g, c, limiter, face_average)[0]
        k_b = 1.0 / (0.5 * dx / jnp.maximum(D_b, tiny) + 2.0 / c)
        ax = axis + 1
        k = _set_ends(
            k,
            axis,
            lax.index_in_dim(k_b, 0, ax, keepdims=False),
            lax.index_in_dim(k_b, -1, ax, keepdims=False),
        )

    return k


def _pad_edges(f, axis, bc):
    """Pad a material property with copies of its edge cells (or periodically)"""

    return _pad(f, axis, "periodic" if bc == "periodic" else "reflective")


def _pad_velocity(v, axis, bc):
    """Pad the velocity normal to an axis: mirrored oddly at a mirrored boundary"""

    if bc in _MIRRORED:
        return _pad(v, axis, bc, odd=True)
    return _pad_edges(v, axis, bc)


def _flux_limiter(E, sigma_t, dx, limiter, axis, bc, ghost, face_average):
    """
    lambda grad E on the faces along an axis: the radiation force per unit
    volume is minus this (the c's cancel)
    """

    E_lo, E_hi = _sides(_pad(E, axis, bc, ghost), axis)
    s_lo, s_hi = _sides(_pad_edges(sigma_t, axis, bc), axis)

    tiny = jnp.finfo(E.dtype).tiny
    grad = (E_hi - E_lo) / dx
    E_face = 0.5 * (jnp.abs(E_lo) + jnp.abs(E_hi))
    g = jnp.abs(E_hi - E_lo) / jnp.maximum(dx * E_face, tiny)
    lam = _face_coefficient(s_lo, s_hi, g, 1.0, limiter, face_average)[1]
    if bc in _MIRRORED:
        zero = jnp.zeros_like(lax.index_in_dim(lam, 0, axis + 1, keepdims=False))
        lam = _set_ends(lam, axis, zero, zero)
    return lam, lam * grad


def _divergence(F_x, F_y, A_x, A_y, vol):
    """Volume-averaged divergence of face fluxes (..., x+1, y) and (..., x, y+1)"""

    AF_x = A_x * F_x
    AF_y = A_y * F_y
    return (
        AF_x[..., 1:, :] - AF_x[..., :-1, :] + AF_y[..., :, 1:] - AF_y[..., :, :-1]
    ) / vol


def _moving_gas(
    E, sigma_t, W, geom, A_x, A_y, dt, limiter, bc_x, bc_y, ghosts, face_average
):
    """
    The explicit, first order in v/c coupling to a moving gas: the radiation is
    carried with the gas, with its enthalpy (1 + lambda) E, and pushes on it with
    the force -lambda grad E, whose work it loses. The energy the gas gains is
    exactly what the radiation loses, apart from what is carried across faces.

    Returns the radiation energy density, and the change of the gas momentum
    and total energy densities.
    """

    vx, vy = W["vx"], W["vy"]
    terms = []
    for axis, (v, dx, bc) in enumerate(
        ((vx, geom["dx"], bc_x), (vy, geom["dy"], bc_y))
    ):
        ghost = ghosts[axis]
        lam, lam_grad = _flux_limiter(
            E, sigma_t, dx, limiter, axis, bc, ghost, face_average
        )
        v_face = 0.5 * sum(_sides(_pad_velocity(v, axis, bc), axis))
        E_lo, E_hi = _sides(_pad(E, axis, bc, ghost), axis)
        upwind = jnp.where(v_face > 0.0, E_lo, E_hi)
        flux = (1.0 + lam) * upwind * v_face
        # the force per unit volume at the cell center, from its two faces
        ax = axis + 1
        n = lam_grad.shape[ax]
        force = -0.5 * (
            lax.slice_in_dim(lam_grad, 0, n - 1, axis=ax)
            + lax.slice_in_dim(lam_grad, 1, n, axis=ax)
        )
        terms.append((flux, force))

    (flux_x, force_x), (flux_y, force_y) = terms
    work = W["vx"] * force_x + W["vy"] * force_y

    E_new = E - dt * (_divergence(flux_x, flux_y, A_x, A_y, geom["vol"]) + work)
    d_mom_x = dt * jnp.sum(force_x, axis=0)
    d_mom_y = dt * jnp.sum(force_y, axis=0)
    d_energy = dt * jnp.sum(work, axis=0)

    return E_new, d_mom_x, d_mom_y, d_energy


def radiation_fld2d_step(
    W,
    E,
    geom,
    dt,
    *,
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
    flux_limiter="levermore_pomraning",
    face_average="harmonic",
    tol=1.0e-4,
    max_iter=1000,
    max_newton=50,
    restart=30,
    static=False,
):
    """
    Advance the radiation energy density implicitly by dt, by flux-limited
    diffusion, and exchange energy and momentum with the gas.

    Parameters
    ----------
    W: dict
      Primitive gas fields; uses 'rho', 'vx', 'vy', 'P' and updates 'vx', 'vy'
      and 'P'. The gas temperature is T = P / (rho * gas_constant).
    E: jax.Array
      Radiation energy density of each frequency group, shape
      (num_groups, nx, ny).
    geom: dict
      Mesh metric factors on the bare grid (no ghost cells), from get_geometry.
    dt: float
      Timestep.
    opacity: dict
      Absorption coefficients rho*kappa, held fixed over the step:
      'sigma_s' (scattering), 'sigma_a' (Rosseland mean absorption) and
      'sigma_p' (Planck mean absorption), each broadcastable to
      (num_groups, nx, ny). The diffusion uses sigma_s + sigma_a, and the
      exchange of energy with the gas sigma_p.
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
      Boundary conditions. 'outflow' is a vacuum (Marshak) boundary, through
      which the radiation leaves at c E / 2; 'driven' is a Marshak boundary
      that also lets in radiation of the energy density given by
      ghost_x/ghost_y = (lo, hi), each of shape (num_groups, 1, ny) or
      (num_groups, nx, 1). Mirrored boundaries have no flux.
    flux_limiter: str
      'levermore_pomraning', 'larsen' (n = 2), 'max', or 'none' (plain
      diffusion, which does not limit the flux in optically thin gas).
    face_average: str
      How the diffusion coefficient on a face is formed from those of the cells
      on either side: their harmonic mean ('harmonic', from the mean opacity,
      right for steady flow across a material interface) or their arithmetic
      mean ('arithmetic', right for a heat front moving into cold material,
      whose high opacity would otherwise choke it; with a flux limiter, it can
      make the solution oscillate at a sharp interface between optically thin
      and thick material).
    tol: float
      Tolerance of the Newton iteration: it stops once its last update of the
      energy densities, summed over the grid, is this fraction of their change
      over the step.
    max_iter: int
      Maximum number of conjugate gradient (grey) or GMRES (multigroup)
      iterations per linear solve.
    max_newton: int
      Maximum number of Newton iterations.
    restart: int
      GMRES restart length (multigroup).
    static: bool
      Hold the material fixed: the radiation exchanges energy with it but does
      not push on it.

    Returns
    -------
    W: dict
      The gas fields after the energy and momentum exchange.
    E: jax.Array
      The radiation energy densities at the end of the step.
    """

    num_groups = len(group_edges) + 1
    if E.shape[0] != num_groups:
        raise ValueError(
            f"energy densities have {E.shape[0]} groups but group_edges gives "
            f"{num_groups}"
        )
    shape = E.shape

    rho, P = W["rho"], W["P"]
    sigma_s = jnp.broadcast_to(opacity["sigma_s"], shape)
    sigma_a = jnp.broadcast_to(opacity["sigma_a"], shape)
    sigma_p = jnp.broadcast_to(opacity["sigma_p"], shape)
    sigma_t = sigma_s + sigma_a

    nx = shape[1]
    A_x = jnp.broadcast_to(_face_areas_x(geom), (nx + 1, 1))
    A_y, vol = geom["area_y"], geom["vol"]
    ghosts = (ghost_x, ghost_y)

    # the coupling to a moving gas, explicitly
    if static:
        E_star = E
    else:
        E_star, d_mom_x, d_mom_y, d_energy = _moving_gas(
            E,
            sigma_t,
            W,
            geom,
            A_x,
            A_y,
            dt,
            flux_limiter,
            bc_x,
            bc_y,
            ghosts,
            face_average,
        )

    # The face conductances, with the flux limiter evaluated from the energy
    # densities at the end of the step: lagging it (from the start of the
    # step) would let the radiation diffuse without limit through gas it has
    # not yet reached, where the energy density is still flat.
    def conductances(E):
        k_x = _conductance(
            E, sigma_t, geom["dx"], c, flux_limiter, 0, bc_x, ghost_x, face_average
        )
        k_y = _conductance(
            E, sigma_t, geom["dy"], c, flux_limiter, 1, bc_y, ghost_y, face_average
        )
        return k_x, k_y

    def diffusion(E, k):
        k_x, k_y = k
        E_lo, E_hi = _sides(_pad(E, 0, bc_x, ghost_x), 0)
        F_x = k_x * (E_lo - E_hi)
        E_lo, E_hi = _sides(_pad(E, 1, bc_y, ghost_y), 1)
        F_y = k_y * (E_lo - E_hi)
        return dt * _divergence(F_x, F_y, A_x, A_y, vol)

    def diffusion_diag(k):
        k_x, k_y = k
        return (
            dt
            * (
                A_x[1:] * k_x[..., 1:, :]
                + A_x[:-1] * k_x[..., :-1, :]
                + A_y * (k_y[..., 1:] + k_y[..., :-1])
            )
            / vol
        )

    # the gas temperature at the end of the step, for given energy densities:
    # the root of C_v (T - T_m) + c dt sum_f sigma_p (4 pi eps_f(T) - E_f) = 0
    k = c * dt
    T_m = P / (rho * gas_constant)
    C_v = rho * gas_constant / (gamma - 1.0)

    def emission(T):
        return 4.0 * jnp.pi * group_emission(T, group_edges, a_r)

    def phi_of(T, E, p):
        exchange = jnp.sum(p["sigma_p"] * (emission(T) - E), axis=0)
        return p["C_v"] * (T - p["T_m"]) + k * exchange

    p = {"sigma_p": sigma_p, "C_v": C_v, "T_m": T_m}

    def temperature(E):
        sg = lax.stop_gradient
        ps = jax.tree.map(sg, p)
        absorbed = k * jnp.sum(ps["sigma_p"] * jnp.maximum(sg(E), 0.0), axis=0)
        T_hi = ps["T_m"] + absorbed / ps["C_v"]
        T_lo = 1.0e-6 * ps["T_m"]
        T = _solve_temperature(lambda T: phi_of(T, sg(E), ps), T_lo, T_hi, ps["T_m"])
        # one more Newton step from the root gives its derivatives
        f, df = jax.jvp(lambda T: phi_of(T, E, p), (T,), (jnp.ones_like(T),))
        return T - f / lax.stop_gradient(df)

    def residual_with(E, cond):
        T = temperature(E)
        return E - E_star + diffusion(E, cond) - k * sigma_p * (emission(T) - E)

    def residual(E):
        return residual_with(E, conductances(E))

    # a Jacobi preconditioner: the diagonal of the Jacobian, which includes the
    # cell's own heating of the gas, and the re-emission that follows
    T_0 = temperature(E_star)
    d_emission = jax.jvp(emission, (T_0,), (jnp.ones_like(T_0),))[1]
    Q = C_v + k * jnp.sum(sigma_p * d_emission, axis=0)
    diag_exchange = 1.0 + k * sigma_p * (1.0 - k * sigma_p * d_emission / Q)
    diag_exchange = lax.stop_gradient(diag_exchange)

    # Weighted by the cell volumes, the grey system is symmetric positive
    # definite (diffusion, plus absorption and the Marshak boundaries on the
    # diagonal), and is solved by conjugate gradients. The groups of a
    # multigroup system are coupled through the gas temperature, which breaks
    # the symmetry, and it is solved by GMRES.
    atol = 100.0 * jnp.finfo(E.dtype).eps * jnp.linalg.norm(vol * E)

    def linear_solve(matvec, b, tol_rel, diag, symmetric):
        def weighted(x):
            return vol * matvec(x)

        def precondition(x):
            return x / (vol * diag)

        if symmetric:
            x, _ = cg(
                weighted,
                vol * b,
                tol=tol_rel,
                atol=atol,
                maxiter=max_iter,
                M=precondition,
            )
        else:
            x, _ = gmres(
                weighted,
                vol * b,
                tol=tol_rel,
                atol=atol,
                restart=restart,
                maxiter=max_iter // restart + 1,
                M=precondition,
            )
        return x

    # Each iteration holds the conductances at the current iterate (a Picard
    # linearization of the flux limiter), which keeps the grey system
    # symmetric; the iterations converge to the fully implicit solution.
    def solve(f, x0):
        def cond(carry):
            x, dx, it = carry
            return ~_converged(dx, x, E, tol) & (it < max_newton)

        def body(carry):
            x, _, it = carry
            cond_x = conductances(x)
            r, f_lin = jax.linearize(lambda e: residual_with(e, cond_x), x)
            diag = diag_exchange + diffusion_diag(cond_x)
            dx = linear_solve(f_lin, -r, 0.1 * tol, diag, num_groups == 1)
            return x + dx, dx, it + 1

        x, _, _ = lax.while_loop(cond, body, body((x0, None, 0)))
        return x

    # the exact linearization, which includes the derivatives of the flux
    # limiter, is not symmetric
    diag_tangent = lax.stop_gradient(
        diag_exchange + diffusion_diag(conductances(E_star))
    )

    def tangent_solve(g, y):
        def solve_linear(matvec, b):
            return linear_solve(matvec, b, tol, diag_tangent, False)

        return lax.custom_linear_solve(g, y, solve_linear, transpose_solve=solve_linear)

    E_new = lax.custom_root(residual, E_star, solve, tangent_solve)

    # what the radiation lost to the gas is its change less what was diffused
    diffused = diffusion(E_new, conductances(E_new))
    d_internal = -jnp.sum(E_new - E_star + diffused, axis=0)

    if static:
        return {**W, "P": P + (gamma - 1.0) * d_internal}, E_new

    vx, vy = W["vx"], W["vy"]
    vx_new = vx + d_mom_x / rho
    vy_new = vy + d_mom_y / rho
    d_kinetic = 0.5 * rho * (vx_new**2 + vy_new**2 - vx**2 - vy**2)
    P_new = P + (gamma - 1.0) * (d_internal + d_energy - d_kinetic)

    return {**W, "vx": vx_new, "vy": vy_new, "P": P_new}, E_new


def radiation_fld_flux(
    E, opacity, geom, c, flux_limiter="levermore_pomraning", face_average="harmonic"
):
    """
    The radiation flux F = -D grad E of each group at the cell centers, for
    periodic boundaries: a diagnostic, returned as (F_x, F_y)
    """

    sigma_t = jnp.broadcast_to(opacity["sigma_s"] + opacity["sigma_a"], E.shape)
    fluxes = []
    for axis, dx in enumerate((geom["dx"], geom["dy"])):
        k = _conductance(
            E, sigma_t, dx, c, flux_limiter, axis, "periodic", None, face_average
        )
        E_lo, E_hi = _sides(_pad(E, axis, "periodic"), axis)
        F = k * (E_lo - E_hi)
        ax = axis + 1
        n = F.shape[ax]
        fluxes.append(
            0.5
            * (
                lax.slice_in_dim(F, 0, n - 1, axis=ax)
                + lax.slice_in_dim(F, 1, n, axis=ax)
            )
        )
    return tuple(fluxes)
