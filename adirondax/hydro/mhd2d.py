import jax.numpy as jnp

from .common2d import (
    apply_fluxes,
    extrapolate_to_face,
    get_avg,
    get_curl,
    get_gradient,
    pad_edge,
    slope_limit,
    strip_ghosts,
    zero_ghost_gradients,
)

# Pure functions for 2D magnetohydrodynamics


def get_conserved(rho, vx, vy, P, Bx, By, gamma, vol, vz=None, Bz=None, r=None):
    """
    Calculate the conserved variable from the primitive
    """
    v_sq = vx**2 + vy**2
    B_sq = Bx**2 + By**2
    if vz is not None:
        v_sq = v_sq + vz**2
        B_sq = B_sq + Bz**2

    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = ((P - 0.5 * B_sq) / (gamma - 1.0) + 0.5 * rho * v_sq + 0.5 * B_sq) * vol
    # in cylindrical geometry the out-of-plane momentum is carried as angular
    # momentum rho*R*vphi, whose flux is free of geometric source terms
    if vz is None:
        Momz = None
    elif r is None:
        Momz = rho * vz * vol
    else:
        Momz = rho * r * vz * vol

    return Mass, Momx, Momy, Energy, Momz


def get_primitive(
    Mass, Momx, Momy, Energy, Bx, By, gamma, vol, Momz=None, Bz=None, r=None
):
    """
    Calculate the primitive variable from the conservative
    """
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol

    v_sq = vx**2 + vy**2
    B_sq = Bx**2 + By**2
    if Momz is None:
        vz = None
    else:
        vz = Momz / (Mass * r) if r is not None else Momz / rho / vol
        v_sq = v_sq + vz**2
        B_sq = B_sq + Bz**2

    P_tot = (Energy / vol - 0.5 * rho * v_sq - 0.5 * B_sq) * (gamma - 1.0) + 0.5 * B_sq

    return rho, vx, vy, P_tot, vz


def _pad_driven(f, axis, lo, hi):
    """Pad a field with caller-supplied ghost values on each side of an axis"""

    return jnp.concatenate((lo, f, hi), axis=axis)


def _ghost_parity(f_d, axis, is_odd):
    """
    Mirror the normal gradient into the ghost cells.
    """

    sign = 1.0 if is_odd else -1.0
    if axis == 0:
        f_d = f_d.at[0, :].set(sign * f_d[1, :])
        f_d = f_d.at[-1, :].set(sign * f_d[-2, :])
    else:
        f_d = f_d.at[:, 0].set(sign * f_d[:, 1])
        f_d = f_d.at[:, -1].set(sign * f_d[:, -2])
    return f_d


def _mirror(f, axis, sign_lo, sign_hi):
    """Pad a field with one mirrored ghost cell on each side of the given axis"""

    if axis == 0:
        return jnp.concatenate((sign_lo * f[0:1, :], f, sign_hi * f[-1:, :]), axis=0)
    else:
        return jnp.concatenate((sign_lo * f[:, 0:1], f, sign_hi * f[:, -1:]), axis=1)


def constrained_transport(bx, by, flux_By_X, flux_Bx_Y, dx, dy, dt, geom=None):
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
    if geom is None or not geom["is_cylindrical"]:
        dbx, dby = get_curl(-Ez, dx, dy)
    else:
        # d(b_R)/dt = -dE/dz
        dbx = -(Ez - jnp.roll(Ez, 1, axis=1)) / dy
        # d(b_z)/dt = +(1/R) d(R E)/dR
        rE = geom["r_face_x"] * Ez
        dby = (rE - jnp.roll(rE, 1, axis=0)) / (geom["r"] * dx)

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
    vz_L,
    vz_R,
    Bz_L,
    Bz_R,
    gamma,
):
    """
    Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule
    """

    has_out = vz_L is not None
    vt_L = (vy_L, vz_L) if has_out else (vy_L,)
    vt_R = (vy_R, vz_R) if has_out else (vy_R,)
    Bt_L = (By_L, Bz_L) if has_out else (By_L,)
    Bt_R = (By_R, Bz_R) if has_out else (By_R,)

    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    B_sq_L = Bx_L**2 + dot(Bt_L, Bt_L)
    B_sq_R = Bx_R**2 + dot(Bt_R, Bt_R)

    # left and right energies
    en_L = (
        (P_L - 0.5 * B_sq_L) / (gamma - 1.0)
        + 0.5 * rho_L * (vx_L**2 + dot(vt_L, vt_L))
        + 0.5 * B_sq_L
    )
    en_R = (
        (P_R - 0.5 * B_sq_R) / (gamma - 1.0)
        + 0.5 * rho_R * (vx_R**2 + dot(vt_R, vt_R))
        + 0.5 * B_sq_R
    )

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momt_star = tuple(0.5 * (rho_L * a + rho_R * b) for a, b in zip(vt_L, vt_R))
    en_star = 0.5 * (en_L + en_R)
    Bx_star = 0.5 * (Bx_L + Bx_R)
    Bt_star = tuple(0.5 * (a + b) for a, b in zip(Bt_L, Bt_R))

    B_sq_star = Bx_star**2 + dot(Bt_star, Bt_star)
    P_star = (gamma - 1.0) * (
        en_star
        - 0.5 * (momx_star**2 + dot(momt_star, momt_star)) / rho_star
        - 0.5 * B_sq_star
    ) + 0.5 * B_sq_star

    # compute fluxes
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star - Bx_star * Bx_star
    flux_Momt = tuple(
        momx_star * m / rho_star - Bx_star * B for m, B in zip(momt_star, Bt_star)
    )
    flux_Energy = (en_star + P_star) * momx_star / rho_star - Bx_star * (
        Bx_star * momx_star + dot(Bt_star, momt_star)
    ) / rho_star
    flux_Bt = tuple(
        (B * momx_star - Bx_star * m) / rho_star for m, B in zip(momt_star, Bt_star)
    )

    # find wavespeeds
    c0_L = jnp.sqrt(gamma * (P_L - 0.5 * B_sq_L) / rho_L)
    c0_R = jnp.sqrt(gamma * (P_R - 0.5 * B_sq_R) / rho_R)
    ca_L = jnp.sqrt(B_sq_L / rho_L)
    ca_R = jnp.sqrt(B_sq_R / rho_R)
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
    flux_Momt = tuple(
        f - C * 0.5 * (rho_R * b - rho_L * a) for f, a, b in zip(flux_Momt, vt_L, vt_R)
    )
    flux_Energy -= C * 0.5 * (en_R - en_L)
    flux_Bt = tuple(f - C * 0.5 * (b - a) for f, a, b in zip(flux_Bt, Bt_L, Bt_R))

    flux_Momz = flux_Momt[1] if has_out else None
    flux_Bz = flux_Bt[1] if has_out else None

    return (
        flux_Mass,
        flux_Momx,
        flux_Momt[0],
        flux_Energy,
        flux_Bt[0],
        flux_Momz,
        flux_Bz,
    )


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
    vz_L,
    vz_R,
    Bz_L,
    Bz_R,
    gamma,
):
    """
    Calculate fluxes between 2 states with HLLD Riemann solver
    """

    epsilon = 1.0e-8

    has_out = vz_L is not None
    vt_L = (vy_L, vz_L) if has_out else (vy_L,)
    vt_R = (vy_R, vz_R) if has_out else (vy_R,)
    Bt_L = (By_L, Bz_L) if has_out else (By_L,)
    Bt_R = (By_R, Bz_R) if has_out else (By_R,)

    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    Bt_sq_L = dot(Bt_L, Bt_L)
    Bt_sq_R = dot(Bt_R, Bt_R)

    P_L -= 0.5 * (Bx_L**2 + Bt_sq_L)
    P_R -= 0.5 * (Bx_R**2 + Bt_sq_R)

    Bxi = 0.5 * (Bx_L + Bx_R)

    Mx_L = rho_L * vx_L
    Mt_L = tuple(rho_L * v for v in vt_L)
    E_L = (
        P_L / (gamma - 1.0)
        + 0.5 * rho_L * (vx_L**2 + dot(vt_L, vt_L))
        + 0.5 * (Bx_L**2 + Bt_sq_L)
    )

    Mx_R = rho_R * vx_R
    Mt_R = tuple(rho_R * v for v in vt_R)
    E_R = (
        P_R / (gamma - 1.0)
        + 0.5 * rho_R * (vx_R**2 + dot(vt_R, vt_R))
        + 0.5 * (Bx_R**2 + Bt_sq_R)
    )

    # Step 2
    # Compute left & right wave speeds according to Miyoshi & Kusano, eqn. (67)

    pbl = 0.5 * (Bxi**2 + Bt_sq_L)
    pbr = 0.5 * (Bxi**2 + Bt_sq_R)
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
    FL_Mt = tuple(rho_L * vx_L * v - Bxi * B for v, B in zip(vt_L, Bt_L))
    FL_E = vx_L * (E_L + ptl - Bxsq) - Bxi * dot(vt_L, Bt_L)
    FL_Bt = tuple(B * vx_L - Bxi * v for v, B in zip(vt_L, Bt_L))

    FR_d = Mx_R
    FR_Mx = Mx_R * vx_R + ptr - Bxsq
    FR_Mt = tuple(rho_R * vx_R * v - Bxi * B for v, B in zip(vt_R, Bt_R))
    FR_E = vx_R * (E_R + ptr - Bxsq) - Bxi * dot(vt_R, Bt_R)
    FR_Bt = tuple(B * vx_R - Bxi * v for v, B in zip(vt_R, Bt_R))

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
    isDegenL = jnp.abs(rho_L * sdl * sdml / Bxsq - 1.0) < epsilon

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdl - sdml) / (rho_L * sdl * sdml - Bxsq)
    ULst_Mt = tuple(
        (ULst_d * v) * isDegenL + (ULst_d * (v - B * tmp)) * (~isDegenL)
        for v, B in zip(vt_L, Bt_L)
    )

    # eqns (45) and (47) of M&K
    tmp = (rho_L * (sdl) ** 2 - Bxsq) / (rho_L * sdl * sdml - Bxsq)
    ULst_Bt = tuple(B * isDegenL + (B * tmp) * (~isDegenL) for B in Bt_L)

    vbstl = (ULst_Mx * Bxi + dot(ULst_Mt, ULst_Bt)) / ULst_d
    # eqn (48) of M&K
    ULst_E = (
        sdl * E_L
        - ptl * vx_L
        + ptst * spd3
        + Bxi * (vx_L * Bxi + dot(vt_L, Bt_L) - vbstl)
    ) / sdml

    WLst_vt = tuple(M / ULst_d for M in ULst_Mt)

    # Ur*
    # eqn (39) of M&K
    URst_Mx = URst_d * spd3
    # URst_Bx = Bxi
    isDegenR = jnp.abs(rho_R * sdr * sdmr / Bxsq - 1.0) < epsilon

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdr - sdmr) / (rho_R * sdr * sdmr - Bxsq)
    URst_Mt = tuple(
        (URst_d * v) * isDegenR + (URst_d * (v - B * tmp)) * (~isDegenR)
        for v, B in zip(vt_R, Bt_R)
    )

    # eqns (45) and (47) of M&K
    tmp = (rho_R * (sdr) ** 2 - Bxsq) / (rho_R * sdr * sdmr - Bxsq)
    URst_Bt = tuple(B * isDegenR + (B * tmp) * (~isDegenR) for B in Bt_R)

    vbstr = (URst_Mx * Bxi + dot(URst_Mt, URst_Bt)) / URst_d
    # eqn (48) of M&K
    URst_E = (
        sdr * E_R
        - ptr * vx_R
        + ptst * spd3
        + Bxi * (vx_R * Bxi + dot(vt_R, Bt_R) - vbstr)
    ) / sdmr

    WRst_vt = tuple(M / URst_d for M in URst_Mt)

    # Ul** and Ur**  - if Bx is zero, same as *-states
    # if(Bxi == 0.0)
    isDegen = 0.5 * Bxsq / jnp.minimum(pbl, pbr) < (epsilon) ** 2
    ULdst_d = ULst_d * isDegen
    ULdst_Mx = ULst_Mx * isDegen
    ULdst_Mt = tuple(M * isDegen for M in ULst_Mt)
    ULdst_Bt = tuple(B * isDegen for B in ULst_Bt)
    ULdst_E = ULst_E * isDegen

    URdst_d = URst_d * isDegen
    URdst_Mx = URst_Mx * isDegen
    URdst_Mt = tuple(M * isDegen for M in URst_Mt)
    URdst_Bt = tuple(B * isDegen for B in URst_Bt)
    URdst_E = URst_E * isDegen

    # else
    invsumd = 1.0 / (sqrtdl + sqrtdr)
    Bxsig = jnp.sign(Bxi)

    ULdst_d = ULdst_d + ULst_d * (~isDegen)
    URdst_d = URdst_d + URst_d * (~isDegen)

    ULdst_Mx = ULdst_Mx + ULst_Mx * (~isDegen)
    URdst_Mx = URdst_Mx + URst_Mx * (~isDegen)

    # eqn (59) of M&K
    tmp_v = tuple(
        invsumd * (sqrtdl * wl + sqrtdr * wr + Bxsig * (bR - bL))
        for wl, wr, bR, bL in zip(WLst_vt, WRst_vt, URst_Bt, ULst_Bt)
    )
    ULdst_Mt = tuple(M + ULdst_d * t * (~isDegen) for M, t in zip(ULdst_Mt, tmp_v))
    URdst_Mt = tuple(M + URdst_d * t * (~isDegen) for M, t in zip(URdst_Mt, tmp_v))

    # eqn (61) of M&K
    tmp_b = tuple(
        invsumd * (sqrtdl * bR + sqrtdr * bL + Bxsig * sqrtdl * sqrtdr * (wr - wl))
        for bR, bL, wl, wr in zip(URst_Bt, ULst_Bt, WLst_vt, WRst_vt)
    )
    ULdst_Bt = tuple(B + t * (~isDegen) for B, t in zip(ULdst_Bt, tmp_b))
    URdst_Bt = tuple(B + t * (~isDegen) for B, t in zip(URdst_Bt, tmp_b))

    # eqn (63) of M&K
    tmp = spd3 * Bxi + dot(ULdst_Mt, ULdst_Bt) / ULdst_d
    ULdst_E = ULdst_E + (ULst_E - sqrtdl * Bxsig * (vbstl - tmp)) * (~isDegen)
    URdst_E = URdst_E + (URst_E + sqrtdr * Bxsig * (vbstr - tmp)) * (~isDegen)

    # Step 7
    # Compute flux

    # Which of the six regions the interface sits in. These have to be built
    # as a mutually exclusive chain, not as independent conditions: when the
    # normal field vanishes the Alfven speeds collapse onto the contact
    # (spd2 = spd3 = spd4), and independent conditions then select two regions
    # at once and add the flux twice.
    in_L = spd1 >= 0
    in_R = (~in_L) & (spd5 <= 0)
    rest = (~in_L) & (~in_R)
    in_Lst = rest & (spd2 >= 0)
    rest = rest & (~in_Lst)
    in_Ldst = rest & (spd3 >= 0)
    rest = rest & (~in_Ldst)
    in_Rdst = rest & (spd4 > 0)
    in_Rst = rest & (~in_Rdst)
    tmpl = spd2 - spd1
    tmpr = spd4 - spd5

    def assemble(FL, FR, UL, UR, ULst, URst, ULdst, URdst):
        flux = FL * in_L + FR * in_R
        flux += (FL + spd1 * (ULst - UL)) * in_Lst
        flux += (FL - spd1 * UL - tmpl * ULst + spd2 * ULdst) * in_Ldst
        flux += (FR - spd5 * UR - tmpr * URst + spd4 * URdst) * in_Rdst
        flux += (FR + spd5 * (URst - UR)) * in_Rst
        return flux

    flux_Mass = assemble(FL_d, FR_d, rho_L, rho_R, ULst_d, URst_d, ULdst_d, URdst_d)
    flux_Momx = assemble(FL_Mx, FR_Mx, Mx_L, Mx_R, ULst_Mx, URst_Mx, ULdst_Mx, URdst_Mx)
    flux_Energy = assemble(FL_E, FR_E, E_L, E_R, ULst_E, URst_E, ULdst_E, URdst_E)
    flux_Momt = tuple(
        assemble(fl, fr, ul, ur, uls, urs, uld, urd)
        for fl, fr, ul, ur, uls, urs, uld, urd in zip(
            FL_Mt, FR_Mt, Mt_L, Mt_R, ULst_Mt, URst_Mt, ULdst_Mt, URdst_Mt
        )
    )
    flux_Bt = tuple(
        assemble(fl, fr, bl, br, uls, urs, uld, urd)
        for fl, fr, bl, br, uls, urs, uld, urd in zip(
            FL_Bt, FR_Bt, Bt_L, Bt_R, ULst_Bt, URst_Bt, ULdst_Bt, URdst_Bt
        )
    )

    flux_Momz = flux_Momt[1] if has_out else None
    flux_Bz = flux_Bt[1] if has_out else None

    return (
        flux_Mass,
        flux_Momx,
        flux_Momt[0],
        flux_Energy,
        flux_Bt[0],
        flux_Momz,
        flux_Bz,
    )


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
    vz_L,
    vz_R,
    Bz_L,
    Bz_R,
    gamma,
    riemann_solver_type,
):
    args = (
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
        vz_L,
        vz_R,
        Bz_L,
        Bz_R,
        gamma,
    )
    if riemann_solver_type == "hlld":
        return get_flux_hlld(*args)
    else:
        # default
        return get_flux_llf(*args)


def hydro_mhd2d_timestep(rho, vx, vy, P, bx, by, gamma, dx, dy, vz=None, Bz=None):
    """
    Calculate the simulation timestep based on CFL condition
    """

    Bx, By = get_avg(bx, by)
    v_sq = vx**2 + vy**2
    B_sq = Bx**2 + By**2
    if vz is not None:
        v_sq = v_sq + vz**2
        B_sq = B_sq + Bz**2

    c_s_sq = gamma * (P - 0.5 * B_sq) / rho
    v_a_sq = B_sq / rho

    # get time step (CFL) = dx / max signal speed
    dl = jnp.minimum(dx, dy)
    dt = jnp.min(dl / (jnp.sqrt(v_sq) + jnp.sqrt(c_s_sq + v_a_sq)))

    return dt


def hydro_mhd2d_fluxes(
    rho,
    vx,
    vy,
    P,
    bx,
    by,
    gamma,
    geom,
    dt,
    riemann_solver_type,
    use_slope_limiting,
    bc_x="periodic",
    bc_y="periodic",
    vz=None,
    bz=None,
    geom_bare=None,
    ghost_x=None,
    ghost_y=None,
):
    """
    Take a simulation timestep
    """

    use_out_of_plane = vz is not None
    dx = geom["dx"]
    dy = geom["dy"]
    is_cylindrical = geom["is_cylindrical"]
    r = geom["r"]
    if geom_bare is None:
        geom_bare = geom

    x_has_ghosts = bc_x != "periodic"
    y_has_ghosts = bc_y != "periodic"

    for axis, bc in ((0, bc_x), (1, bc_y)):
        if bc == "periodic":
            continue
        if bc == "outflow":
            rho, vx, vy, P, bx, by = (
                pad_edge(f, axis) for f in (rho, vx, vy, P, bx, by)
            )
            if use_out_of_plane:
                vz, bz = (pad_edge(f, axis) for f in (vz, bz))
        elif bc == "driven":
            # Ghost values supplied by the caller, e.g. to implement an external driver.
            ghost = ghost_x if axis == 0 else ghost_y
            rho, vx, vy, P, bx, by = (
                _pad_driven(f, axis, *ghost[name])
                for f, name in zip(
                    (rho, vx, vy, P, bx, by),
                    ("rho", "vx", "vy", "P", "bx", "by"),
                )
            )
            if use_out_of_plane:
                vz = _pad_driven(vz, axis, *ghost["vz"])
                bz = _pad_driven(bz, axis, *ghost["bz"])
        elif bc == "wall":
            # A perfectly conducting wall.
            rho, vy, P = (_mirror(f, axis, 1.0, 1.0) for f in (rho, vy, P))
            vx = _mirror(vx, axis, -1.0, -1.0)
            by = _mirror(by, axis, 1.0, 1.0)
            bx = jnp.concatenate(
                (jnp.zeros_like(bx[0:1, :]), bx, jnp.zeros_like(bx[-1:, :])), axis=0
            )
            if use_out_of_plane:
                vz = _mirror(vz, axis, 1.0, 1.0)
                bz = _mirror(bz, axis, 1.0, 1.0)
        elif bc == "axis":
            rho, vy, P = (_mirror(f, axis, 1.0, 1.0) for f in (rho, vy, P))
            vx = _mirror(vx, axis, -1.0, -1.0)
            by = _mirror(by, axis, 1.0, 1.0)
            bx = jnp.concatenate(
                (jnp.zeros_like(bx[0:1, :]), bx, jnp.zeros_like(bx[-1:, :])), axis=0
            )
            if use_out_of_plane:
                vz = _mirror(vz, axis, -1.0, 1.0)
                bz = _mirror(bz, axis, -1.0, 1.0)
        else:
            raise NotImplementedError(
                f"'{bc}' boundaries are not implemented for magnetic fields"
            )

    # get Conserved variables
    Bx, By = get_avg(bx, by)
    Mass, Momx, Momy, Energy, Momz = get_conserved(
        rho, vx, vy, P, Bx, By, gamma, geom["vol"], vz, bz, r
    )
    Bz_cons = None if not use_out_of_plane else bz * (dx * dy)

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx, dy)
    vx_dx, vx_dy = get_gradient(vx, dx, dy)
    vy_dx, vy_dy = get_gradient(vy, dx, dy)
    P_dx, P_dy = get_gradient(P, dx, dy)
    Bx_dx, Bx_dy = get_gradient(Bx, dx, dy)
    By_dx, By_dy = get_gradient(By, dx, dy)
    if use_out_of_plane:
        vz_dx, vz_dy = get_gradient(vz, dx, dy)
        Bz_dx, Bz_dy = get_gradient(bz, dx, dy)

    # slope limit gradients
    if use_slope_limiting:
        rho_dx, rho_dy = slope_limit(rho, rho_dx, rho_dy, dx, dy)
        vx_dx, vx_dy = slope_limit(vx, vx_dx, vx_dy, dx, dy)
        vy_dx, vy_dy = slope_limit(vy, vy_dx, vy_dy, dx, dy)
        P_dx, P_dy = slope_limit(P, P_dx, P_dy, dx, dy)
        Bx_dx, Bx_dy = slope_limit(Bx, Bx_dx, Bx_dy, dx, dy)
        By_dx, By_dy = slope_limit(By, By_dx, By_dy, dx, dy)
        if use_out_of_plane:
            vz_dx, vz_dy = slope_limit(vz, vz_dx, vz_dy, dx, dy)
            Bz_dx, Bz_dy = slope_limit(bz, Bz_dx, Bz_dy, dx, dy)

    # set ghost cell gradients
    for axis, has_ghosts in ((0, x_has_ghosts), (1, y_has_ghosts)):
        if not has_ghosts:
            continue
        if axis == 0 and bc_x in ("driven", "wall"):
            if bc_x == "driven":
                # the ghost values are data, so flatten the slopes into them
                rho_dx = zero_ghost_gradients(rho_dx, axis)
                vx_dx = zero_ghost_gradients(vx_dx, axis)
                vy_dx = zero_ghost_gradients(vy_dx, axis)
                P_dx = zero_ghost_gradients(P_dx, axis)
                Bx_dx = zero_ghost_gradients(Bx_dx, axis)
                By_dx = zero_ghost_gradients(By_dx, axis)
                if use_out_of_plane:
                    vz_dx = zero_ghost_gradients(vz_dx, axis)
                    Bz_dx = zero_ghost_gradients(Bz_dx, axis)
            else:
                rho_dx = _ghost_parity(rho_dx, axis, False)
                vy_dx = _ghost_parity(vy_dx, axis, False)
                P_dx = _ghost_parity(P_dx, axis, False)
                Bx_dx = _ghost_parity(Bx_dx, axis, True)
                By_dx = _ghost_parity(By_dx, axis, False)
                vx_dx = _ghost_parity(vx_dx, axis, True)
                if use_out_of_plane:
                    vz_dx = _ghost_parity(vz_dx, axis, False)
                    Bz_dx = _ghost_parity(Bz_dx, axis, False)
        elif axis == 0 and bc_x == "axis":
            rho_dx = _ghost_parity(rho_dx, axis, False)
            vy_dx = _ghost_parity(vy_dx, axis, False)
            P_dx = _ghost_parity(P_dx, axis, False)
            Bx_dx = _ghost_parity(Bx_dx, axis, True)
            By_dx = _ghost_parity(By_dx, axis, False)
            vx_dx = _ghost_parity(vx_dx, axis, True)
            if use_out_of_plane:
                vz_dx = _ghost_parity(vz_dx, axis, True)
                Bz_dx = _ghost_parity(Bz_dx, axis, True)
        elif axis == 0:
            rho_dx = zero_ghost_gradients(rho_dx, axis)
            vx_dx = zero_ghost_gradients(vx_dx, axis)
            vy_dx = zero_ghost_gradients(vy_dx, axis)
            P_dx = zero_ghost_gradients(P_dx, axis)
            Bx_dx = zero_ghost_gradients(Bx_dx, axis)
            By_dx = zero_ghost_gradients(By_dx, axis)
            if use_out_of_plane:
                vz_dx = zero_ghost_gradients(vz_dx, axis)
                Bz_dx = zero_ghost_gradients(Bz_dx, axis)
        else:
            rho_dy = zero_ghost_gradients(rho_dy, axis)
            vx_dy = zero_ghost_gradients(vx_dy, axis)
            vy_dy = zero_ghost_gradients(vy_dy, axis)
            P_dy = zero_ghost_gradients(P_dy, axis)
            Bx_dy = zero_ghost_gradients(Bx_dy, axis)
            By_dy = zero_ghost_gradients(By_dy, axis)
            if use_out_of_plane:
                vz_dy = zero_ghost_gradients(vz_dy, axis)
                Bz_dy = zero_ghost_gradients(Bz_dy, axis)

    # extrapolate half-step in time
    div_v = vx_dx + vy_dy
    if is_cylindrical:
        div_v = div_v + vx / r

    rho_prime = rho - 0.5 * dt * (vx * rho_dx + vy * rho_dy + rho * div_v)
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
    B_sq = Bx**2 + By**2
    if use_out_of_plane:
        B_sq = B_sq + bz**2
    P_prime = P - 0.5 * dt * (
        gamma * (P - 0.5 * B_sq) * div_v
        + By**2 * vx_dx
        - Bx * By * vy_dx
        + vx * P_dx
        + (gamma - 2.0) * (Bx * vx + By * vy) * Bx_dx
        - By * Bx * vx_dy
        + Bx**2 * vy_dy
        + vy * P_dy
        + (gamma - 2.0) * (Bx * vx + By * vy) * By_dy
    )
    if use_out_of_plane:
        # the out-of-plane field adds magnetic pressure that resists in-plane
        # compression, and couples to the shear in vz
        P_prime = P_prime - 0.5 * dt * (
            bz**2 * (vx_dx + vy_dy) - bz * Bx * vz_dx - bz * By * vz_dy
        )

    Bx_prime = Bx - 0.5 * dt * (-By * vx_dy + Bx * vy_dy + vy * Bx_dy - vx * By_dy)
    By_prime = By - 0.5 * dt * (By * vx_dx - Bx * vy_dx - vy * Bx_dx + vx * By_dx)
    if use_out_of_plane:
        # d(vz)/dt = (B.grad) Bz / rho; in cylindrical the azimuthal equation
        # also carries -v_R v_phi / R and +B_R B_phi / (rho R)
        vz_prime = vz - 0.5 * dt * (
            vx * vz_dx + vy * vz_dy - (Bx / rho) * Bz_dx - (By / rho) * Bz_dy
        )
        if is_cylindrical:
            vz_prime = vz_prime - 0.5 * dt * (vx * vz / r - Bx * bz / (rho * r))
        # d(Bz)/dt = -div(vx Bz - vz Bx, vy Bz - vz By), using div(B) = 0
        bz_prime = bz - 0.5 * dt * (
            vx * Bz_dx + vy * Bz_dy + bz * (vx_dx + vy_dy) - Bx * vz_dx - By * vz_dy
        )

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(
        rho_prime, rho_dx, rho_dy, dx, dy
    )
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx_prime, vx_dx, vx_dy, dx, dy)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy_prime, vy_dx, vy_dy, dx, dy)
    P_XL, P_XR, P_YL, P_YR = extrapolate_to_face(P_prime, P_dx, P_dy, dx, dy)
    Bx_XL, Bx_XR, Bx_YL, Bx_YR = extrapolate_to_face(Bx_prime, Bx_dx, Bx_dy, dx, dy)
    By_XL, By_XR, By_YL, By_YR = extrapolate_to_face(By_prime, By_dx, By_dy, dx, dy)
    if use_out_of_plane:
        vz_XL, vz_XR, vz_YL, vz_YR = extrapolate_to_face(vz_prime, vz_dx, vz_dy, dx, dy)
        Bz_XL, Bz_XR, Bz_YL, Bz_YR = extrapolate_to_face(bz_prime, Bz_dx, Bz_dy, dx, dy)
    else:
        vz_XL = vz_XR = vz_YL = vz_YR = None
        Bz_XL = Bz_XR = Bz_YL = Bz_YR = None

    # compute fluxes
    (
        flux_Mass_X,
        flux_Momx_X,
        flux_Momy_X,
        flux_Energy_X,
        flux_By_X,
        flux_Momz_X,
        flux_Bz_X,
    ) = get_flux(
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
        vz_XL,
        vz_XR,
        Bz_XL,
        Bz_XR,
        gamma,
        riemann_solver_type,
    )
    (
        flux_Mass_Y,
        flux_Momy_Y,
        flux_Momx_Y,
        flux_Energy_Y,
        flux_Bx_Y,
        flux_Momz_Y,
        flux_Bz_Y,
    ) = get_flux(
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
        vz_YL,
        vz_YR,
        Bz_YL,
        Bz_YR,
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
    if use_out_of_plane:
        if is_cylindrical:
            # angular momentum: the flux is weighted by the shared face radius,
            # and already contains the Maxwell stress -R B_R B_phi
            Momz = apply_fluxes(
                Momz,
                geom["r_face_x"] * flux_Momz_X,
                r * flux_Momz_Y,
                area_x,
                area_y,
                dt,
            )
        else:
            Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, area_x, area_y, dt)
        Bz_cons = apply_fluxes(Bz_cons, flux_Bz_X, flux_Bz_Y, dy, dx, dt)
    bx, by = constrained_transport(bx, by, flux_By_X, flux_Bx_Y, dx, dy, dt, geom)

    # geometric source terms in the radial momentum
    if is_cylindrical:
        Momx = Momx + dt * P_prime * geom["d_area_x"]
        if use_out_of_plane:
            # centrifugal force and the magnetic hoop stress
            Momx = Momx + dt * geom["vol"] * (rho_prime * vz_prime**2 - bz_prime**2) / r

    # remove ghost cells
    for axis, has_ghosts in ((0, x_has_ghosts), (1, y_has_ghosts)):
        if has_ghosts:
            Mass, Momx, Momy, Energy, bx, by = (
                strip_ghosts(f, axis) for f in (Mass, Momx, Momy, Energy, bx, by)
            )
            if use_out_of_plane:
                Momz, Bz_cons = (strip_ghosts(f, axis) for f in (Momz, Bz_cons))

    # get Primitive variables
    Bx, By = get_avg(bx, by)
    bz = None if not use_out_of_plane else Bz_cons / (dx * dy)
    rho, vx, vy, P, vz = get_primitive(
        Mass,
        Momx,
        Momy,
        Energy,
        Bx,
        By,
        gamma,
        geom_bare["vol"],
        Momz,
        bz,
        geom_bare["r"],
    )

    return rho, vx, vy, P, bx, by, vz, bz
