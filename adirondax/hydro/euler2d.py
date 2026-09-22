import jax.numpy as jnp

from .common2d import (
    apply_fluxes,
    extrapolate_to_face,
    get_gradient,
    pad_edge,
    slope_limit,
    zero_ghost_gradients,
)

# Pure functions for 2D Euler hydrodynamics


def get_conserved(rho, vx, vy, P, vphi, gamma, geom):
    """Calculate the conserved variables from the primitive variables"""

    vol = geom["vol"]

    v_sq = vx**2 + vy**2
    if vphi is not None:
        v_sq = v_sq + vphi**2

    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P / (gamma - 1.0) + 0.5 * rho * v_sq) * vol
    Angmom = None if vphi is None else rho * geom["r"] * vphi * vol

    return Mass, Momx, Momy, Energy, Angmom


def get_primitive(Mass, Momx, Momy, Energy, Angmom, gamma, geom):
    """Calculate the primitive variable from the conserved variables"""

    vol = geom["vol"]

    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol

    v_sq = vx**2 + vy**2
    if Angmom is None:
        vphi = None
    else:
        # the centroid radius is strictly positive, even in the cell on the axis
        vphi = Angmom / (Mass * geom["r"])
        v_sq = v_sq + vphi**2

    P = (Energy / vol - 0.5 * rho * v_sq) * (gamma - 1.0)

    return rho, vx, vy, P, vphi


def get_flux_llf(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, vphi_L, vphi_R, gamma):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    has_rotation = vphi_L is not None

    # left and right energies
    v_sq_L = vx_L**2 + vy_L**2
    v_sq_R = vx_R**2 + vy_R**2
    if has_rotation:
        v_sq_L = v_sq_L + vphi_L**2
        v_sq_R = v_sq_R + vphi_R**2
    en_L = P_L / (gamma - 1.0) + 0.5 * rho_L * v_sq_L
    en_R = P_R / (gamma - 1.0) + 0.5 * rho_R * v_sq_R

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)

    mom_sq_star = momx_star**2 + momy_star**2
    if has_rotation:
        momphi_star = 0.5 * (rho_L * vphi_L + rho_R * vphi_R)
        mom_sq_star = mom_sq_star + momphi_star**2

    P_star = (gamma - 1.0) * (en_star - 0.5 * mom_sq_star / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star
    flux_Momphi = None if not has_rotation else momx_star * momphi_star / rho_star

    # find wavespeeds (the azimuthal velocity advects nothing across the face
    # in axisymmetry, so it does not enter the signal speed)
    C_L = jnp.sqrt(gamma * P_L / rho_L) + jnp.abs(vx_L)
    C_R = jnp.sqrt(gamma * P_R / rho_R) + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_R - rho_L)
    flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
    flux_Energy -= C * 0.5 * (en_R - en_L)
    if has_rotation:
        flux_Momphi -= C * 0.5 * (rho_R * vphi_R - rho_L * vphi_L)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_Momphi


def get_flux_hllc(
    rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, vphi_L, vphi_R, gamma
):
    """
    Calculate fluxes between 2 states with the HLLC approximate Riemann solver
    """

    has_rotation = vphi_L is not None

    rho_l, u_l, v_l, p_l = rho_L, vx_L, vy_L, P_L
    rho_r, u_r, v_r, p_r = rho_R, vx_R, vy_R, P_R
    w_l = vphi_L if has_rotation else 0.0
    w_r = vphi_R if has_rotation else 0.0

    # total energy density
    en_l = p_l / (gamma - 1.0) + 0.5 * rho_l * (u_l**2 + v_l**2 + w_l**2)
    en_r = p_r / (gamma - 1.0) + 0.5 * rho_r * (u_r**2 + v_r**2 + w_r**2)

    # outer signal speeds (Davis estimate)
    a_l = jnp.sqrt(gamma * p_l / rho_l)
    a_r = jnp.sqrt(gamma * p_r / rho_r)
    s_l = jnp.minimum(u_l - a_l, u_r - a_r)
    s_r = jnp.maximum(u_l + a_l, u_r + a_r)

    # contact wave speed. The denominator cannot vanish for physical states:
    # (s_l - u_l) <= -a_l < 0 while (s_r - u_r) >= a_r > 0.
    d_l = rho_l * (s_l - u_l)
    d_r = rho_r * (s_r - u_r)
    s_star = (p_r - p_l + d_l * u_l - d_r * u_r) / (d_l - d_r)

    def state_and_flux(rho_k, u_k, v_k, w_k, p_k, en_k):
        U = (rho_k, rho_k * u_k, rho_k * v_k, en_k, rho_k * w_k)
        F = (
            rho_k * u_k,
            rho_k * u_k**2 + p_k,
            rho_k * u_k * v_k,
            (en_k + p_k) * u_k,
            rho_k * u_k * w_k,
        )
        return U, F

    def star_flux(rho_k, u_k, v_k, w_k, p_k, en_k, s_k, d_k):
        """Flux of the intermediate state: F*_k = F_k + s_k (U*_k - U_k)"""
        U, F = state_and_flux(rho_k, u_k, v_k, w_k, p_k, en_k)
        rho_star = rho_k * (s_k - u_k) / (s_k - s_star)
        en_star = rho_star * (en_k / rho_k + (s_star - u_k) * (s_star + p_k / d_k))
        U_star = (
            rho_star,
            rho_star * s_star,
            rho_star * v_k,
            en_star,
            rho_star * w_k,
        )
        return tuple(f + s_k * (us - u) for f, us, u in zip(F, U_star, U))

    _, flux_l = state_and_flux(rho_l, u_l, v_l, w_l, p_l, en_l)
    _, flux_r = state_and_flux(rho_r, u_r, v_r, w_r, p_r, en_r)
    flux_star_l = star_flux(rho_l, u_l, v_l, w_l, p_l, en_l, s_l, d_l)
    flux_star_r = star_flux(rho_r, u_r, v_r, w_r, p_r, en_r, s_r, d_r)

    # pick the state the interface sits in
    flux = tuple(
        jnp.where(
            s_l >= 0.0,
            f_l,
            jnp.where(s_star >= 0.0, fs_l, jnp.where(s_r >= 0.0, fs_r, f_r)),
        )
        for f_l, fs_l, fs_r, f_r in zip(flux_l, flux_star_l, flux_star_r, flux_r)
    )

    flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_Momphi = flux
    if not has_rotation:
        flux_Momphi = None

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_Momphi


def get_flux(
    rho_L,
    rho_R,
    vx_L,
    vx_R,
    vy_L,
    vy_R,
    P_L,
    P_R,
    vphi_L,
    vphi_R,
    gamma,
    riemann_solver_type,
):
    if riemann_solver_type == "hllc":
        return get_flux_hllc(
            rho_L,
            rho_R,
            vx_L,
            vx_R,
            vy_L,
            vy_R,
            P_L,
            P_R,
            vphi_L,
            vphi_R,
            gamma,
        )
    else:
        # default
        return get_flux_llf(
            rho_L,
            rho_R,
            vx_L,
            vx_R,
            vy_L,
            vy_R,
            P_L,
            P_R,
            vphi_L,
            vphi_R,
            gamma,
        )


def hydro_euler2d_timestep(rho, vx, vy, P, gamma, dx, dy):
    """Calculate the simulation timestep based on CFL condition"""

    # get time step (CFL) = dx / max signal speed
    dl = jnp.minimum(dx, dy)
    dt = jnp.min(dl / (jnp.sqrt(gamma * P / rho) + jnp.sqrt(vx**2 + vy**2)))

    return dt


def _mirror(f, axis, sign_lo, sign_hi):
    """Pad a field with one mirrored ghost cell on each side of the given axis"""

    if axis == 0:
        return jnp.concatenate((sign_lo * f[0:1, :], f, sign_hi * f[-1:, :]), axis=0)
    else:
        return jnp.concatenate((sign_lo * f[:, 0:1], f, sign_hi * f[:, -1:]), axis=1)


def add_ghost_cells(rho, vx, vy, P, vphi, axis, bc):
    """
    Add ghost cells for a non-periodic boundary along the given axis

    'outflow' copies the edge value outwards, so waves leave the domain.
    'reflective' and 'axis' mirror the fields evenly, except the velocity
    component normal to the boundary, which is mirrored oddly. On the
    cylindrical axis the azimuthal velocity is odd as well (it reverses sense
    through R=0), whereas at an ordinary free-slip wall it is tangential and
    therefore even.
    """

    if bc == "outflow":
        return tuple(
            None if f is None else pad_edge(f, axis) for f in (rho, vx, vy, P, vphi)
        )

    is_axis = bc == "axis"

    if axis == 0:
        rho_new = _mirror(rho, 0, 1.0, 1.0)
        vx_new = _mirror(vx, 0, -1.0, -1.0)
        vy_new = _mirror(vy, 0, 1.0, 1.0)
        P_new = _mirror(P, 0, 1.0, 1.0)
        vphi_new = (
            None if vphi is None else _mirror(vphi, 0, -1.0 if is_axis else 1.0, 1.0)
        )
    else:
        rho_new = _mirror(rho, 1, 1.0, 1.0)
        vx_new = _mirror(vx, 1, 1.0, 1.0)
        vy_new = _mirror(vy, 1, -1.0, -1.0)
        P_new = _mirror(P, 1, 1.0, 1.0)
        vphi_new = None if vphi is None else _mirror(vphi, 1, 1.0, 1.0)

    return rho_new, vx_new, vy_new, P_new, vphi_new


def remove_ghost_cells(Mass, Momx, Momy, Energy, Angmom, axis):
    """Remove ghost cells for reflective boundary conditions along given axis"""

    sl = (slice(1, -1), slice(None)) if axis == 0 else (slice(None), slice(1, -1))

    return (
        Mass[sl],
        Momx[sl],
        Momy[sl],
        Energy[sl],
        None if Angmom is None else Angmom[sl],
    )


def set_ghost_gradients(f_dx, axis, is_odd_lo=False, is_odd_hi=False):
    """
    Set the normal gradient in the ghost cells by mirroring the first interior
    cell (f_dx already has ghost cells).
    """

    s_lo = 1.0 if is_odd_lo else -1.0
    s_hi = 1.0 if is_odd_hi else -1.0

    if axis == 0:
        f_dx = f_dx.at[0, :].set(s_lo * f_dx[1, :])
        f_dx = f_dx.at[-1, :].set(s_hi * f_dx[-2, :])
    elif axis == 1:
        f_dx = f_dx.at[:, 0].set(s_lo * f_dx[:, 1])
        f_dx = f_dx.at[:, -1].set(s_hi * f_dx[:, -2])

    return f_dx


def hydro_euler2d_fluxes(
    rho,
    vx,
    vy,
    P,
    vphi,
    gamma,
    geom,
    dt,
    riemann_solver_type,
    use_slope_limiting,
    bc_x,
    bc_y,
):
    """Take a simulation timestep"""

    dx = geom["dx"]
    dy = geom["dy"]
    is_cylindrical = geom["is_cylindrical"]
    x_has_ghosts = bc_x != "periodic"
    y_has_ghosts = bc_y != "periodic"

    # Add Ghost Cells (if needed)
    if x_has_ghosts:
        rho, vx, vy, P, vphi = add_ghost_cells(rho, vx, vy, P, vphi, 0, bc_x)
    if y_has_ghosts:
        rho, vx, vy, P, vphi = add_ghost_cells(rho, vx, vy, P, vphi, 1, bc_y)

    # get Conserved variables
    Mass, Momx, Momy, Energy, Angmom = get_conserved(rho, vx, vy, P, vphi, gamma, geom)

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx, dy)
    vx_dx, vx_dy = get_gradient(vx, dx, dy)
    vy_dx, vy_dy = get_gradient(vy, dx, dy)
    P_dx, P_dy = get_gradient(P, dx, dy)
    if vphi is not None:
        vphi_dx, vphi_dy = get_gradient(vphi, dx, dy)

    # slope limit gradients
    if use_slope_limiting:
        rho_dx, rho_dy = slope_limit(rho, rho_dx, rho_dy, dx, dy)
        vx_dx, vx_dy = slope_limit(vx, vx_dx, vx_dy, dx, dy)
        vy_dx, vy_dy = slope_limit(vy, vy_dx, vy_dy, dx, dy)
        P_dx, P_dy = slope_limit(P, P_dx, P_dy, dx, dy)
        if vphi is not None:
            vphi_dx, vphi_dy = slope_limit(vphi, vphi_dx, vphi_dy, dx, dy)

    # set ghost cell gradients
    if bc_x == "outflow":
        rho_dx = zero_ghost_gradients(rho_dx, axis=0)
        vx_dx = zero_ghost_gradients(vx_dx, axis=0)
        vy_dx = zero_ghost_gradients(vy_dx, axis=0)
        P_dx = zero_ghost_gradients(P_dx, axis=0)
        if vphi is not None:
            vphi_dx = zero_ghost_gradients(vphi_dx, axis=0)
    elif x_has_ghosts:
        rho_dx = set_ghost_gradients(rho_dx, axis=0)
        vx_dx = set_ghost_gradients(vx_dx, axis=0, is_odd_lo=True, is_odd_hi=True)
        vy_dx = set_ghost_gradients(vy_dx, axis=0)
        P_dx = set_ghost_gradients(P_dx, axis=0)
        if vphi is not None:
            vphi_dx = set_ghost_gradients(vphi_dx, axis=0, is_odd_lo=bc_x == "axis")
    if bc_y == "outflow":
        rho_dy = zero_ghost_gradients(rho_dy, axis=1)
        vx_dy = zero_ghost_gradients(vx_dy, axis=1)
        vy_dy = zero_ghost_gradients(vy_dy, axis=1)
        P_dy = zero_ghost_gradients(P_dy, axis=1)
        if vphi is not None:
            vphi_dy = zero_ghost_gradients(vphi_dy, axis=1)
    elif y_has_ghosts:
        rho_dy = set_ghost_gradients(rho_dy, axis=1)
        vx_dy = set_ghost_gradients(vx_dy, axis=1)
        vy_dy = set_ghost_gradients(vy_dy, axis=1, is_odd_lo=True, is_odd_hi=True)
        P_dy = set_ghost_gradients(P_dy, axis=1)
        if vphi is not None:
            vphi_dy = set_ghost_gradients(vphi_dy, axis=1)

    # velocity divergence picks up a geometric term in cylindrical geometry
    div_v = vx_dx + vy_dy
    if is_cylindrical:
        div_v = div_v + vx / geom["r"]

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + vy * rho_dy + rho * div_v)
    vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1.0 / rho) * P_dx)
    vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1.0 / rho) * P_dy)
    P_prime = P - 0.5 * dt * (gamma * P * div_v + vx * P_dx + vy * P_dy)
    if vphi is None:
        vphi_prime = None
    else:
        # in terms of the specific angular momentum R*vphi this is pure
        # advection; written for vphi it carries the -vx*vphi/R term
        vphi_prime = vphi - 0.5 * dt * (
            vx * vphi_dx + vy * vphi_dy + vx * vphi / geom["r"]
        )
        # centrifugal acceleration
        vx_prime = vx_prime + 0.5 * dt * vphi**2 / geom["r"]

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(
        rho_prime, rho_dx, rho_dy, dx, dy
    )
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx_prime, vx_dx, vx_dy, dx, dy)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy_prime, vy_dx, vy_dy, dx, dy)
    P_XL, P_XR, P_YL, P_YR = extrapolate_to_face(P_prime, P_dx, P_dy, dx, dy)
    if vphi is None:
        vphi_XL = vphi_XR = vphi_YL = vphi_YR = None
    else:
        vphi_XL, vphi_XR, vphi_YL, vphi_YR = extrapolate_to_face(
            vphi_prime, vphi_dx, vphi_dy, dx, dy
        )

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_Momphi_X = get_flux(
        rho_XL,
        rho_XR,
        vx_XL,
        vx_XR,
        vy_XL,
        vy_XR,
        P_XL,
        P_XR,
        vphi_XL,
        vphi_XR,
        gamma,
        riemann_solver_type,
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_Momphi_Y = get_flux(
        rho_YL,
        rho_YR,
        vy_YL,
        vy_YR,
        vx_YL,
        vx_YR,
        P_YL,
        P_YR,
        vphi_YL,
        vphi_YR,
        gamma,
        riemann_solver_type,
    )

    # update solution
    area_x = geom["area_x"]
    area_y = geom["area_y"]
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, area_x, area_y, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, area_x, area_y, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, area_x, area_y, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, area_x, area_y, dt)
    if Angmom is not None:
        Angmom = apply_fluxes(
            Angmom,
            geom["r_face_x"] * flux_Momphi_X,
            geom["r"] * flux_Momphi_Y,
            area_x,
            area_y,
            dt,
        )

    # geometric source terms (time-centered, from the half-step primitives)
    if is_cylindrical:
        Momx = Momx + dt * P_prime * geom["d_area_x"]
        if vphi is not None:
            Momx = Momx + dt * (rho_prime * geom["vol"]) * vphi_prime**2 / geom["r"]

    # remove ghost cells
    if x_has_ghosts:
        Mass, Momx, Momy, Energy, Angmom = remove_ghost_cells(
            Mass, Momx, Momy, Energy, Angmom, axis=0
        )
    if y_has_ghosts:
        Mass, Momx, Momy, Energy, Angmom = remove_ghost_cells(
            Mass, Momx, Momy, Energy, Angmom, axis=1
        )

    rho, vx, vy, P, vphi = get_primitive(
        Mass, Momx, Momy, Energy, Angmom, gamma, geom_strip(geom, x_has_ghosts)
    )

    return rho, vx, vy, P, vphi


def geom_strip(geom, has_ghosts):
    """Return the geometry factors restricted to the interior cells"""

    if not has_ghosts or not geom["is_cylindrical"]:
        return geom

    stripped = dict(geom)
    for key in ("vol", "area_x", "area_y", "d_area_x", "r", "r_face_x"):
        stripped[key] = geom[key][1:-1]
    return stripped


def hydro_euler2d_accelerate(rho, vx, vy, P, vphi, ax, ay, gamma, geom, dt):
    Mass, Momx, Momy, Energy, Angmom = get_conserved(rho, vx, vy, P, vphi, gamma, geom)

    Energy += dt * (Momx * ax + Momy * ay)
    Momx += dt * Mass * ax
    Momy += dt * Mass * ay

    _, vx_new, vy_new, P_new, _ = get_primitive(
        Mass, Momx, Momy, Energy, Angmom, gamma, geom
    )
    return vx_new, vy_new, P_new
