import jax.numpy as jnp
from .common2d import get_gradient, slope_limit, extrapolate_to_face, apply_fluxes

# Pure functions for 2D Euler hydrodynamics


def get_conserved(rho, vx, vy, P, gamma, vol):
    """Calculate the conserved variables from the primitive variables"""

    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2)) * vol

    return Mass, Momx, Momy, Energy


def get_primitive(Mass, Momx, Momy, Energy, gamma, vol):
    """Calculate the primitive variable from the conserved variables"""

    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy / vol - 0.5 * rho * (vx**2 + vy**2)) * (gamma - 1.0)

    return rho, vx, vy, P


def get_flux_llf(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    # left and right energies
    en_L = P_L / (gamma - 1.0) + 0.5 * rho_L * (vx_L**2 + vy_L**2)
    en_R = P_R / (gamma - 1.0) + 0.5 * rho_R * (vx_R**2 + vy_R**2)

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)

    P_star = (gamma - 1.0) * (en_star - 0.5 * (momx_star**2 + momy_star**2) / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star

    # find wavespeeds
    C_L = jnp.sqrt(gamma * P_L / rho_L) + jnp.abs(vx_L)
    C_R = jnp.sqrt(gamma * P_R / rho_R) + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy


def get_flux(
    rho_L,
    rho_R,
    vx_L,
    vx_R,
    vy_L,
    vy_R,
    P_L,
    P_R,
    gamma,
    riemann_solver_type,
):
    if riemann_solver_type == "XXX":
        return None
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
            gamma,
        )


def hydro_euler2d_timestep(rho, vx, vy, P, gamma, dx, dy):
    """Calculate the simulation timestep based on CFL condition"""

    # get time step (CFL) = dx / max signal speed
    dl = jnp.minimum(dx, dy)
    dt = jnp.min(dl / (jnp.sqrt(gamma * P / rho) + jnp.sqrt(vx**2 + vy**2)))

    return dt


def add_ghost_cells(rho, vx, vy, P, axis):
    """Add ghost cells for reflective boundary conditions along given axis"""

    if axis == 0:
        # x-axis
        rho_new = jnp.concatenate((rho[0:1, :], rho, rho[-1:, :]), axis=0)
        vx_new = jnp.concatenate((-vx[0:1, :], vx, -vx[-1:, :]), axis=0)
        vy_new = jnp.concatenate((vy[0:1, :], vy, vy[-1:, :]), axis=0)
        P_new = jnp.concatenate((P[0:1, :], P, P[-1:, :]), axis=0)
    elif axis == 1:
        # y-axis
        rho_new = jnp.concatenate((rho[:, 0:1], rho, rho[:, -1:]), axis=1)
        vx_new = jnp.concatenate((vx[:, 0:1], vx, vx[:, -1:]), axis=1)
        vy_new = jnp.concatenate((-vy[:, 0:1], vy, -vy[:, -1:]), axis=1)
        P_new = jnp.concatenate((P[:, 0:1], P, P[:, -1:]), axis=1)

    return rho_new, vx_new, vy_new, P_new


def remove_ghost_cells(Mass, Momx, Momy, Energy, axis):
    """Remove ghost cells for reflective boundary conditions along given axis"""

    if axis == 0:
        # x-axis
        Mass_new = Mass[1:-1, :]
        Momx_new = Momx[1:-1, :]
        Momy_new = Momy[1:-1, :]
        Energy_new = Energy[1:-1, :]
    elif axis == 1:
        # y-axis
        Mass_new = Mass[:, 1:-1]
        Momx_new = Momx[:, 1:-1]
        Momy_new = Momy[:, 1:-1]
        Energy_new = Energy[:, 1:-1]

    return Mass_new, Momx_new, Momy_new, Energy_new


def set_ghost_gradients(f_dx, axis):
    """Set gradients in ghost cells to (-1) x  value of the first interior cell (f_dx already has ghost cells)"""

    if axis == 0:
        f_dx = f_dx.at[0, :].set(-f_dx[1, :])
        f_dx = f_dx.at[-1, :].set(-f_dx[-2, :])
    elif axis == 1:
        f_dx = f_dx.at[:, 0].set(-f_dx[:, 1])
        f_dx = f_dx.at[:, -1].set(-f_dx[:, -2])

    return f_dx


def hydro_euler2d_fluxes(
    rho,
    vx,
    vy,
    P,
    gamma,
    dx,
    dy,
    dt,
    riemann_solver_type,
    use_slope_limiting,
    bc_x_is_reflective,
    bc_y_is_reflective,
):
    """Take a simulation timestep"""

    # Add Ghost Cells (if needed)
    if bc_x_is_reflective:
        rho, vx, vy, P = add_ghost_cells(rho, vx, vy, P, axis=0)
    if bc_y_is_reflective:
        rho, vx, vy, P = add_ghost_cells(rho, vx, vy, P, axis=1)

    # get Conserved variables
    Mass, Momx, Momy, Energy = get_conserved(rho, vx, vy, P, gamma, dx * dy)

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx, dy)
    vx_dx, vx_dy = get_gradient(vx, dx, dy)
    vy_dx, vy_dy = get_gradient(vy, dx, dy)
    P_dx, P_dy = get_gradient(P, dx, dy)

    # slope limit gradients
    if use_slope_limiting:
        rho_dx, rho_dy = slope_limit(rho, rho_dx, rho_dy, dx, dy)
        vx_dx, vx_dy = slope_limit(vx, vx_dx, vx_dy, dx, dy)
        vy_dx, vy_dy = slope_limit(vy, vy_dx, vy_dy, dx, dy)
        P_dx, P_dy = slope_limit(P, P_dx, P_dy, dx, dy)

    # set ghost cell gradients
    if bc_x_is_reflective:
        rho_dx = set_ghost_gradients(rho_dx, axis=0)
        vx_dx = set_ghost_gradients(vx_dx, axis=0)
        vy_dx = set_ghost_gradients(vy_dx, axis=0)
        P_dx = set_ghost_gradients(P_dx, axis=0)
    if bc_y_is_reflective:
        rho_dy = set_ghost_gradients(rho_dy, axis=1)
        vx_dy = set_ghost_gradients(vx_dy, axis=1)
        vy_dy = set_ghost_gradients(vy_dy, axis=1)
        P_dy = set_ghost_gradients(P_dy, axis=1)

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
    vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1.0 / rho) * P_dx)
    vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1.0 / rho) * P_dy)
    P_prime = P - 0.5 * dt * (gamma * P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(
        rho_prime, rho_dx, rho_dy, dx, dy
    )
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx_prime, vx_dx, vx_dy, dx, dy)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy_prime, vy_dx, vy_dy, dx, dy)
    P_XL, P_XR, P_YL, P_YR = extrapolate_to_face(P_prime, P_dx, P_dy, dx, dy)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = get_flux(
        rho_XL,
        rho_XR,
        vx_XL,
        vx_XR,
        vy_XL,
        vy_XR,
        P_XL,
        P_XR,
        gamma,
        riemann_solver_type,
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = get_flux(
        rho_YL,
        rho_YR,
        vy_YL,
        vy_YR,
        vx_YL,
        vx_YR,
        P_YL,
        P_YR,
        gamma,
        riemann_solver_type,
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dy, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dy, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dy, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dy, dt)

    # remove ghost cells
    if bc_x_is_reflective:
        Mass, Momx, Momy, Energy = remove_ghost_cells(Mass, Momx, Momy, Energy, axis=0)
    if bc_y_is_reflective:
        Mass, Momx, Momy, Energy = remove_ghost_cells(Mass, Momx, Momy, Energy, axis=1)

    rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, gamma, dx * dy)

    return rho, vx, vy, P


def hydro_euler2d_accelerate(rho, vx, vy, P, ax, ay, gamma, dx, dy, dt):
    Mass, Momx, Momy, Energy = get_conserved(rho, vx, vy, P, gamma, dx * dy)

    Energy += dt * (Momx * ax + Momy * ay)
    Momx += dt * Mass * ax
    Momy += dt * Mass * ay

    _, vx_new, vy_new, P_new = get_primitive(Mass, Momx, Momy, Energy, gamma, dx * dy)
    return vx_new, vy_new, P_new
