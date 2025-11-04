import jax.numpy as jnp
from .common2d import (
    get_curl,
    get_div,
    get_avg,
    get_gradient,
    slope_limit,
    extrapolate_to_face,
    apply_fluxes,
)

# Pure functions for 2D magnetohydrodynamics


def get_conserved(rho, vx, vy, P, Bx, By, gamma, vol):
    """
    Calculate the conserved variable from the primitive
    """
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (
        (P - 0.5 * (Bx**2 + By**2)) / (gamma - 1.0)
        + 0.5 * rho * (vx**2 + vy**2)
        + 0.5 * (Bx**2 + By**2)
    ) * vol

    return Mass, Momx, Momy, Energy


def get_primitive(Mass, Momx, Momy, Energy, Bx, By, gamma, vol):
    """
    Calculate the primitive variable from the conservative
    """
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P_tot = (Energy / vol - 0.5 * rho * (vx**2 + vy**2) - 0.5 * (Bx**2 + By**2)) * (
        gamma - 1.0
    ) + 0.5 * (Bx**2 + By**2)

    return rho, vx, vy, P_tot


def constrained_transport(bx, by, flux_By_X, flux_Bx_Y, dx, dt):
    """
    Apply fluxes to face-centered magnetic fields in a constrained transport manner
    """
    # update solution
    # Ez at top right node of cell = avg of 4 fluxes
    Ez = 0.25 * (
        -flux_By_X
        - jnp.roll(flux_By_X, -1, axis=1)
        + flux_Bx_Y
        + jnp.roll(flux_Bx_Y, -1, axis=0)
    )
    dbx, dby = get_curl(-Ez, dx)

    bx += dt * dbx
    by += dt * dby

    return bx, by


# local Lax-Friedrichs/Rusanov
def get_flux(
    rho_L,
    rho_R,
    vx_L,
    vx_R,
    vy_L,
    vy_R,
    P_L,
    P_R,
    Bx_L,
    Bx_R,
    By_L,
    By_R,
    gamma,
):
    """
    Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule
    """

    # left and right energies
    en_L = (
        (P_L - 0.5 * (Bx_L**2 + By_L**2)) / (gamma - 1)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2)
        + 0.5 * (Bx_L**2 + By_L**2)
    )
    en_R = (
        (P_R - 0.5 * (Bx_R**2 + By_R**2)) / (gamma - 1)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2)
        + 0.5 * (Bx_R**2 + By_R**2)
    )

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)
    Bx_star = 0.5 * (Bx_L + Bx_R)
    By_star = 0.5 * (By_L + By_R)

    P_star = (gamma - 1) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star - Bx_star * Bx_star
    flux_Momy = momx_star * momy_star / rho_star - Bx_star * By_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star - Bx_star * (
        Bx_star * momx_star + By_star * momy_star
    ) / rho_star
    flux_By = (By_star * momx_star - Bx_star * momy_star) / rho_star

    # find wavespeeds
    c0_L = jnp.sqrt(gamma * (P_L - 0.5 * (Bx_L**2 + By_L**2)) / rho_L)
    c0_R = jnp.sqrt(gamma * (P_R - 0.5 * (Bx_R**2 + By_R**2)) / rho_R)
    ca_L = jnp.sqrt((Bx_L**2 + By_L**2) / rho_L)
    ca_R = jnp.sqrt((Bx_R**2 + By_R**2) / rho_R)
    cf_L = jnp.sqrt(
        0.5 * (c0_L**2 + ca_L**2) + 0.5 * jnp.sqrt((c0_L**2 + ca_L**2) ** 2)
    )
    cf_R = jnp.sqrt(
        0.5 * (c0_R**2 + ca_R**2) + 0.5 * jnp.sqrt((c0_R**2 + ca_R**2) ** 2)
    )
    C_L = cf_L + jnp.abs(vx_L)
    C_R = cf_R + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)
    flux_By -= C * 0.5 * (By_L - By_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_By


def hydro_mhd2d_fluxes(rho, vx, vy, P, bx, by, gamma, dx, dt):
    """Take a simulation timestep"""

    # get Conserved variables
    Bx, By = get_avg(bx, by)
    Mass, Momx, Momy, Energy = get_conserved(rho, vx, vy, P, Bx, By, gamma, dx**2)

    # get time step (CFL) = dx / max signal speed
    # dt = courant_fac * jnp.min(dx / (jnp.sqrt(gamma * P / rho) + jnp.sqrt(vx**2 + vy**2)))

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx)
    vx_dx, vx_dy = get_gradient(vx, dx)
    vy_dx, vy_dy = get_gradient(vy, dx)
    P_dx, P_dy = get_gradient(P, dx)
    Bx_dx, Bx_dy = get_gradient(Bx, dx)
    By_dx, By_dy = get_gradient(By, dx)

    # slope limit gradients
    rho_dx, rho_dy = slope_limit(rho, dx, rho_dx, rho_dy)
    vx_dx, vx_dy = slope_limit(vx, dx, vx_dx, vx_dy)
    vy_dx, vy_dy = slope_limit(vy, dx, vy_dx, vy_dy)
    P_dx, P_dy = slope_limit(P, dx, P_dx, P_dy)
    Bx_dx, Bx_dy = slope_limit(Bx, dx, Bx_dx, Bx_dy)
    By_dx, By_dy = slope_limit(By, dx, By_dx, By_dy)

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
    vx_prime = vx - 0.5 * dt * (
        vx * vx_dx
        + vy * vx_dy
        + (1 / rho) * P_dx
        - (2 * Bx / rho) * Bx_dx
        - (By / rho) * Bx_dy
        - (Bx / rho) * By_dy
    )
    vy_prime = vy - 0.5 * dt * (
        vx * vy_dx
        + vy * vy_dy
        + (1 / rho) * P_dy
        - (2 * By / rho) * By_dy
        - (Bx / rho) * By_dx
        - (By / rho) * Bx_dx
    )
    P_prime = P - 0.5 * dt * (
        (gamma * (P - 0.5 * (Bx**2 + By**2)) + By**2) * vx_dx
        - Bx * By * vy_dx
        + vx * P_dx
        + (gamma - 2) * (Bx * vx + By * vy) * Bx_dx
        - By * Bx * vx_dy
        + (gamma * (P - 0.5 * (Bx**2 + By**2)) + Bx**2) * vy_dy
        + vy * P_dy
        + (gamma - 2) * (Bx * vx + By * vy) * By_dy
    )
    Bx_prime = Bx - 0.5 * dt * (-By * vx_dy + Bx * vy_dy + vy * Bx_dy - vx * By_dy)
    By_prime = By - 0.5 * dt * (By * vx_dx - Bx * vy_dx - vy * Bx_dx + vx * By_dx)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(rho_prime, rho_dx, rho_dy, dx)
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx_prime, vx_dx, vx_dy, dx)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy_prime, vy_dx, vy_dy, dx)
    P_XL, P_XR, P_YL, P_YR = extrapolate_to_face(P_prime, P_dx, P_dy, dx)
    Bx_XL, Bx_XR, Bx_YL, Bx_YR = extrapolate_to_face(Bx_prime, Bx_dx, Bx_dy, dx)
    By_XL, By_XR, By_YL, By_YR = extrapolate_to_face(By_prime, By_dx, By_dy, dx)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_By_X = get_flux(
        rho_XL,
        rho_XR,
        vx_XL,
        vx_XR,
        vy_XL,
        vy_XR,
        P_XL,
        P_XR,
        Bx_XL,
        Bx_XR,
        By_XL,
        By_XR,
        gamma,
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_Bx_Y = get_flux(
        rho_YL,
        rho_YR,
        vy_YL,
        vy_YR,
        vx_YL,
        vx_YR,
        P_YL,
        P_YR,
        By_YL,
        By_YR,
        Bx_YL,
        Bx_YR,
        gamma,
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
    bx, by = constrained_transport(bx, by, flux_By_X, flux_Bx_Y, dx, dt)

    # get Primitive variables
    Bx, By = get_avg(bx, by)
    rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, Bx, By, gamma, dx**2)

    return rho, vx, vy, P, bx, by
