import jax.numpy as jnp
from .common2d import (
    get_curl,
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

    bx_new = bx + dt * dbx
    by_new = by + dt * dby

    return bx_new, by_new


# local Lax-Friedrichs/Rusanov
def get_flux_llf(
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
        (P_L - 0.5 * (Bx_L**2 + By_L**2)) / (gamma - 1.0)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2)
        + 0.5 * (Bx_L**2 + By_L**2)
    )
    en_R = (
        (P_R - 0.5 * (Bx_R**2 + By_R**2)) / (gamma - 1.0)
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

    P_star = (gamma - 1.0) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2)

    # compute fluxes
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
    flux_Mass -= C * 0.5 * (rho_R - rho_L)
    flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
    flux_Energy -= C * 0.5 * (en_R - en_L)
    flux_By -= C * 0.5 * (By_R - By_L)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_By


# HLLD Riemann solver
def get_flux_hlld(
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
    Calculate fluxes between 2 states with HLLD Riemann solver
    """

    epsilon = 1.0e-8

    P_L -= 0.5 * (Bx_L**2 + By_L**2)
    P_R -= 0.5 * (Bx_R**2 + By_R**2)

    Bxi = 0.5 * (Bx_L + Bx_R)

    Mx_L = rho_L * vx_L
    My_L = rho_L * vy_L
    E_L = (
        P_L / (gamma - 1.0)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2)
        + 0.5 * (Bx_L**2 + By_L**2)
    )

    Mx_R = rho_R * vx_R
    My_R = rho_R * vy_R
    E_R = (
        P_R / (gamma - 1.0)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2)
        + 0.5 * (Bx_R**2 + By_R**2)
    )

    # Step 2
    # Compute left & right wave speeds according to Miyoshi & Kusano, eqn. (67)

    pbl = 0.5 * (Bxi**2 + By_L**2)
    pbr = 0.5 * (Bxi**2 + By_R**2)
    gpl = gamma * P_L
    gpr = gamma * P_R
    gpbl = gpl + 2.0 * pbl
    gpbr = gpr + 2.0 * pbr

    Bxsq = Bxi**2
    cfl = jnp.sqrt((gpbl + jnp.sqrt(gpbl**2 - 4.0 * gpl * Bxsq)) / (2.0 * rho_L))
    cfr = jnp.sqrt((gpbr + jnp.sqrt(gpbr**2 - 4.0 * gpr * Bxsq)) / (2.0 * rho_R))
    cfmax = jnp.maximum(cfl, cfr)

    spd1 = (vx_L - cfmax) * (vx_L <= vx_R) + (vx_R - cfmax) * (vx_L > vx_R)
    spd5 = (vx_R + cfmax) * (vx_L <= vx_R) + (vx_L + cfmax) * (vx_L > vx_R)

    # Step 3
    # Compute L/R fluxes

    # total pressure
    ptl = P_L + pbl
    ptr = P_R + pbr

    FL_d = Mx_L
    FL_Mx = Mx_L * vx_L + ptl - Bxsq
    FL_My = rho_L * vx_L * vy_L - Bxi * By_L
    FL_E = vx_L * (E_L + ptl - Bxsq) - Bxi * (vy_L * By_L)
    FL_By = By_L * vx_L - Bxi * vy_L
    FR_d = Mx_R
    FR_Mx = Mx_R * vx_R + ptr - Bxsq
    FR_My = rho_R * vx_R * vy_R - Bxi * By_R
    FR_E = vx_R * (E_R + ptr - Bxsq) - Bxi * (vy_R * By_R)
    FR_By = By_R * vx_R - Bxi * vy_R

    # Step 4
    # Return upwind flux if flow is supersonic

    # deferred to the end

    # Step 5
    # Compute middle and Alfven wave speeds

    sdl = spd1 - vx_L
    sdr = spd5 - vx_R

    # S_M: eqn (38) of Miyoshi & Kusano
    spd3 = (sdr * rho_R * vx_R - sdl * rho_L * vx_L - ptr + ptl) / (
        sdr * rho_R - sdl * rho_L
    )

    sdml = spd1 - spd3
    sdmr = spd5 - spd3
    # eqn (43) of Miyoshi & Kusano
    ULst_d = rho_L * sdl / sdml
    URst_d = rho_R * sdr / sdmr
    sqrtdl = jnp.sqrt(ULst_d)
    sqrtdr = jnp.sqrt(URst_d)

    # eqn (51) of Miyoshi & Kusano
    spd2 = spd3 - jnp.abs(Bxi) / sqrtdl
    spd4 = spd3 + jnp.abs(Bxi) / sqrtdr

    # Step 6
    # Compute intermediate states

    ptst = ptl + rho_L * sdl * (sdl - sdml)

    # Ul*
    # eqn (39) of M&K
    ULst_Mx = ULst_d * spd3
    # ULst_Bx = Bxi
    isDegen = jnp.abs(rho_L * sdl * sdml / Bxsq - 1.0) < epsilon

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdl - sdml) / (rho_L * sdl * sdml - Bxsq)
    ULst_My = (ULst_d * vy_L) * isDegen + (ULst_d * (vy_L - By_L * tmp)) * (~isDegen)

    # eqns (45) and (47) of M&K
    tmp = (rho_L * (sdl) ** 2 - Bxsq) / (rho_L * sdl * sdml - Bxsq)
    ULst_By = (By_L) * isDegen + (By_L * tmp) * (~isDegen)

    vbstl = (ULst_Mx * Bxi + ULst_My * ULst_By) / ULst_d
    # eqn (48) of M&K
    ULst_E = (
        sdl * E_L - ptl * vx_L + ptst * spd3 + Bxi * (vx_L * Bxi + vy_L * By_L - vbstl)
    ) / sdml

    WLst_vy = ULst_My / ULst_d

    # Ur*
    # eqn (39) of M&K
    URst_Mx = URst_d * spd3
    # URst_Bx = Bxi
    isDegen = jnp.abs(rho_R * sdr * sdmr / Bxsq - 1.0) < epsilon

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdr - sdmr) / (rho_R * sdr * sdmr - Bxsq)
    URst_My = (URst_d * vy_R) * isDegen + (URst_d * (vy_R - By_R * tmp)) * (~isDegen)

    # eqns (45) and (47) of M&K
    tmp = (rho_R * (sdr) ** 2 - Bxsq) / (rho_R * sdr * sdmr - Bxsq)
    URst_By = (By_R) * isDegen + (By_R * tmp) * (~isDegen)

    vbstr = (URst_Mx * Bxi + URst_My * URst_By) / URst_d
    # eqn (48) of M&K
    URst_E = (
        sdr * E_R - ptr * vx_R + ptst * spd3 + Bxi * (vx_R * Bxi + vy_R * By_R - vbstr)
    ) / sdmr

    WRst_vy = URst_My / URst_d

    # Ul** and Ur**  - if Bx is zero, same as *-states
    # if(Bxi == 0.0)
    isDegen = 0.5 * Bxsq / jnp.minimum(pbl, pbr) < (epsilon) ** 2
    ULdst_d = ULst_d * isDegen
    ULdst_Mx = ULst_Mx * isDegen
    ULdst_My = ULst_My * isDegen
    ULdst_By = ULst_By * isDegen
    ULdst_E = ULst_E * isDegen

    URdst_d = URst_d * isDegen
    URdst_Mx = URst_Mx * isDegen
    URdst_My = URst_My * isDegen
    URdst_By = URst_By * isDegen
    URdst_E = URst_E * isDegen

    # else
    invsumd = 1.0 / (sqrtdl + sqrtdr)
    # Bxsig = 0 * Bxi - 1
    # Bxsig[Bxi > 0] = 1
    Bxsig = jnp.sign(Bxi)

    ULdst_d = ULdst_d + ULst_d * (~isDegen)
    URdst_d = URdst_d + URst_d * (~isDegen)

    ULdst_Mx = ULdst_Mx + ULst_Mx * (~isDegen)
    URdst_Mx = URdst_Mx + URst_Mx * (~isDegen)

    # eqn (59) of M&K
    tmp = invsumd * (sqrtdl * WLst_vy + sqrtdr * WRst_vy + Bxsig * (URst_By - ULst_By))
    ULdst_My = ULdst_My + ULdst_d * tmp * (~isDegen)
    URdst_My = URdst_My + URdst_d * tmp * (~isDegen)

    # eqn (61) of M&K
    tmp = invsumd * (
        sqrtdl * URst_By
        + sqrtdr * ULst_By
        + Bxsig * sqrtdl * sqrtdr * (WRst_vy - WLst_vy)
    )
    ULdst_By = ULdst_By + tmp * (~isDegen)
    URdst_By = URdst_By + tmp * (~isDegen)

    # eqn (63) of M&K
    tmp = spd3 * Bxi + (ULdst_My * ULdst_By) / ULdst_d
    ULdst_E = ULdst_E + (ULst_E - sqrtdl * Bxsig * (vbstl - tmp)) * (~isDegen)
    URdst_E = URdst_E + (URst_E + sqrtdr * Bxsig * (vbstr - tmp)) * (~isDegen)

    # Step 7
    # Compute flux

    flux_Mass = FL_d * (spd1 >= 0)
    flux_Momx = FL_Mx * (spd1 >= 0)
    flux_Momy = FL_My * (spd1 >= 0)
    flux_Energy = FL_E * (spd1 >= 0)
    flux_By = FL_By * (spd1 >= 0)

    flux_Mass += FR_d * (spd5 <= 0)
    flux_Momx += FR_Mx * (spd5 <= 0)
    flux_Momy += FR_My * (spd5 <= 0)
    flux_Energy += FR_E * (spd5 <= 0)
    flux_By += FR_By * (spd5 <= 0)

    # if(spd2 >= 0)
    # return Fl*
    flux_Mass += (FL_d + spd1 * (ULst_d - rho_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Momx += (FL_Mx + spd1 * (ULst_Mx - Mx_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Momy += (FL_My + spd1 * (ULst_My - My_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Energy += (FL_E + spd1 * (ULst_E - E_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_By += (FL_By + spd1 * (ULst_By - By_L)) * ((spd1 < 0) & (spd2 >= 0))

    # elseif(spd3 >= 0)
    # return Fl**
    tmp = spd2 - spd1
    flux_Mass += (FL_d - spd1 * rho_L - tmp * ULst_d + spd2 * ULdst_d) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Momx += (FL_Mx - spd1 * Mx_L - tmp * ULst_Mx + spd2 * ULdst_Mx) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Momy += (FL_My - spd1 * My_L - tmp * ULst_My + spd2 * ULdst_My) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Energy += (FL_E - spd1 * E_L - tmp * ULst_E + spd2 * ULdst_E) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_By += (FL_By - spd1 * By_L - tmp * ULst_By + spd2 * ULdst_By) * (
        (spd2 < 0) & (spd3 >= 0)
    )

    # elseif(spd4 > 0)
    # return Fr**
    tmp = spd4 - spd5
    flux_Mass += (FR_d - spd5 * rho_R - tmp * URst_d + spd4 * URdst_d) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Momx += (FR_Mx - spd5 * Mx_R - tmp * URst_Mx + spd4 * URdst_Mx) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Momy += (FR_My - spd5 * My_R - tmp * URst_My + spd4 * URdst_My) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Energy += (FR_E - spd5 * E_R - tmp * URst_E + spd4 * URdst_E) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_By += (FR_By - spd5 * By_R - tmp * URst_By + spd4 * URdst_By) * (
        (spd3 < 0) & (spd4 > 0)
    )

    # else
    # return Fr*
    flux_Mass += (FR_d + spd5 * (URst_d - rho_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Momx += (FR_Mx + spd5 * (URst_Mx - Mx_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Momy += (FR_My + spd5 * (URst_My - My_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Energy += (FR_E + spd5 * (URst_E - E_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_By += (FR_By + spd5 * (URst_By - By_R)) * ((spd4 <= 0) & (spd5 > 0))

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_By


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
    riemann_solver_type,
):
    if riemann_solver_type == "hlld":
        return get_flux_hlld(
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
            Bx_L,
            Bx_R,
            By_L,
            By_R,
            gamma,
        )


def hydro_mhd2d_timestep(rho, vx, vy, P, bx, by, gamma, dx):
    """Calculate the simulation timestep based on CFL condition"""

    # get time step (CFL) = dx / max signal speed
    Bx, By = get_avg(bx, by)
    dt = jnp.min(
        dx
        / (jnp.sqrt(gamma * P / rho) + jnp.sqrt(vx**2 + vy**2 + (Bx**2 + By**2) / rho))
    )

    return dt


def hydro_mhd2d_fluxes(
    rho, vx, vy, P, bx, by, gamma, dx, dt, riemann_solver_type, use_slope_limiting
):
    """Take a simulation timestep"""

    # get Conserved variables
    Bx, By = get_avg(bx, by)
    Mass, Momx, Momy, Energy = get_conserved(rho, vx, vy, P, Bx, By, gamma, dx**2)

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx)
    vx_dx, vx_dy = get_gradient(vx, dx)
    vy_dx, vy_dy = get_gradient(vy, dx)
    P_dx, P_dy = get_gradient(P, dx)
    Bx_dx, Bx_dy = get_gradient(Bx, dx)
    By_dx, By_dy = get_gradient(By, dx)

    # slope limit gradients
    if use_slope_limiting:
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
        + (1.0 / rho) * P_dx
        - (2.0 * Bx / rho) * Bx_dx
        - (By / rho) * Bx_dy
        - (Bx / rho) * By_dy
    )
    vy_prime = vy - 0.5 * dt * (
        vx * vy_dx
        + vy * vy_dy
        + (1.0 / rho) * P_dy
        - (2.0 * By / rho) * By_dy
        - (Bx / rho) * By_dx
        - (By / rho) * Bx_dx
    )
    P_prime = P - 0.5 * dt * (
        (gamma * (P - 0.5 * (Bx**2 + By**2)) + By**2) * vx_dx
        - Bx * By * vy_dx
        + vx * P_dx
        + (gamma - 2.0) * (Bx * vx + By * vy) * Bx_dx
        - By * Bx * vx_dy
        + (gamma * (P - 0.5 * (Bx**2 + By**2)) + Bx**2) * vy_dy
        + vy * P_dy
        + (gamma - 2.0) * (Bx * vx + By * vy) * By_dy
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

    # compute fluxes
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_By_X = get_flux(
        rho_XR,
        rho_XL,
        vx_XR,
        vx_XL,
        vy_XR,
        vy_XL,
        P_XR,
        P_XL,
        Bx_XR,
        Bx_XL,
        By_XR,
        By_XL,
        gamma,
        riemann_solver_type,
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_Bx_Y = get_flux(
        rho_YR,
        rho_YL,
        vy_YR,
        vy_YL,
        vx_YR,
        vx_YL,
        P_YR,
        P_YL,
        By_YR,
        By_YL,
        Bx_YR,
        Bx_YL,
        gamma,
        riemann_solver_type,
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
