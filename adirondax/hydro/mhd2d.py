import jax
import jax.numpy as jnp

from .boundary import (
    add_ghost_cells,
    reset_ghost_cells,
    set_ghost_gradients,
    strip_ghosts,
)
from .common2d import (
    apply_fluxes,
    face_states,
    get_avg,
    get_curl,
    get_gradients,
    swap_xy,
)
from .geometry import geom_strip

# Pure functions for 2D magnetohydrodynamics


def get_conserved(W, gamma, geom):
    """
    Calculate the conserved variable from the primitive
    """
    rho, vx, vy, P, Bx, By = (W[k] for k in ("rho", "vx", "vy", "P", "Bx", "By"))
    vol = geom["vol"]
    r = geom["r"]

    v_sq = vx**2 + vy**2
    B_sq = Bx**2 + By**2
    if "vz" in W:
        v_sq = v_sq + W["vz"] ** 2
        B_sq = B_sq + W["Bz"] ** 2

    U = {
        "mass": rho * vol,
        "momx": rho * vx * vol,
        "momy": rho * vy * vol,
        "energy": ((P - 0.5 * B_sq) / (gamma - 1.0) + 0.5 * rho * v_sq + 0.5 * B_sq)
        * vol,
    }
    # in cylindrical geometry the out-of-plane momentum is carried as angular
    # momentum rho*R*vphi, whose flux is free of geometric source terms
    if "vz" in W:
        U["momz"] = rho * W["vz"] * vol if r is None else rho * r * W["vz"] * vol
        U["Bz"] = W["Bz"] * (geom["dx"] * geom["dy"])

    return U


def get_primitive(U, Bx, By, gamma, geom):
    """
    Calculate the primitive variable from the conservative
    """
    vol = geom["vol"]
    r = geom["r"]

    rho = U["mass"] / vol
    vx = U["momx"] / rho / vol
    vy = U["momy"] / rho / vol
    W = {"rho": rho, "vx": vx, "vy": vy}

    v_sq = vx**2 + vy**2
    B_sq = Bx**2 + By**2
    if "momz" in U:
        W["vz"] = (
            U["momz"] / (U["mass"] * r) if r is not None else U["momz"] / rho / vol
        )
        W["bz"] = U["Bz"] / (geom["dx"] * geom["dy"])
        v_sq = v_sq + W["vz"] ** 2
        B_sq = B_sq + W["bz"] ** 2

    W["P"] = (U["energy"] / vol - 0.5 * rho * v_sq - 0.5 * B_sq) * (
        gamma - 1.0
    ) + 0.5 * B_sq

    return W


def get_cell_centered_field(
    bx, by, bc_x="periodic", bc_y="periodic", ghost_x=None, ghost_y=None
):
    """
    Calculate the volume-averaged magnetic field from the face-centered one

    The grid stores the field on the upper face of each cell, so the lower face
    of the first cell along a non-periodic axis lies outside it, and takes its
    value from the boundary condition (as the solver's ghost cells do), rather
    than wrapping around to the far side of the domain.
    """

    b = {"bx": bx, "by": by}
    for axis, bc, ghost in ((0, bc_x, ghost_x), (1, bc_y, ghost_y)):
        if bc != "periodic":
            b = add_ghost_cells(b, axis, bc, ghost)
    Bx, By = get_avg(b["bx"], b["by"])
    for axis, bc in ((0, bc_x), (1, bc_y)):
        if bc != "periodic":
            Bx, By = strip_ghosts(Bx, axis), strip_ghosts(By, axis)

    return Bx, By


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


def _flux_dict(flux_Mass, flux_Momx, flux_Momt, flux_Energy, flux_Bt):
    flux = {
        "mass": flux_Mass,
        "momx": flux_Momx,
        "momy": flux_Momt[0],
        "energy": flux_Energy,
        "By": flux_Bt[0],
    }
    if len(flux_Momt) > 1:
        flux["momz"] = flux_Momt[1]
        flux["Bz"] = flux_Bt[1]
    return flux


# local Lax-Friedrichs/Rusanov
def get_flux_llf(WL, WR, gamma):
    """
    Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule
    """

    rho_L, vx_L, P_L, Bx_L = WL["rho"], WL["vx"], WL["P"], WL["Bx"]
    rho_R, vx_R, P_R, Bx_R = WR["rho"], WR["vx"], WR["P"], WR["Bx"]

    has_out = "vz" in WL
    vt_L = (WL["vy"], WL["vz"]) if has_out else (WL["vy"],)
    vt_R = (WR["vy"], WR["vz"]) if has_out else (WR["vy"],)
    Bt_L = (WL["By"], WL["Bz"]) if has_out else (WL["By"],)
    Bt_R = (WR["By"], WR["Bz"]) if has_out else (WR["By"],)

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

    # the normal field is continuous across the face
    Bxi = 0.5 * (Bx_L + Bx_R)

    def side_flux(rho, vx, vt, P, Bt, en):
        """Physical flux of one side; P is the total pressure"""
        return (
            rho * vx,
            rho * vx**2 + P - Bxi**2,
            tuple(rho * vx * v - Bxi * B for v, B in zip(vt, Bt)),
            (en + P) * vx - Bxi * (Bxi * vx + dot(vt, Bt)),
            tuple(B * vx - Bxi * v for v, B in zip(vt, Bt)),
        )

    # compute fluxes: the mean of the left and right fluxes
    F_L = side_flux(rho_L, vx_L, vt_L, P_L, Bt_L, en_L)
    F_R = side_flux(rho_R, vx_R, vt_R, P_R, Bt_R, en_R)
    flux_Mass, flux_Momx, flux_Momt, flux_Energy, flux_Bt = jax.tree.map(
        lambda a, b: 0.5 * (a + b), F_L, F_R
    )

    # find wavespeeds: the fast magnetosonic speed normal to the face
    def fast_speed(rho, P, B_sq, Bx):
        a_sq = gamma * (P - 0.5 * B_sq) / rho
        ca_sq = B_sq / rho
        cax_sq = Bx**2 / rho
        # (a^2 + ca^2)^2 - 4 a^2 cax^2 >= (a^2 - ca^2)^2 >= 0; its square root
        # is guarded at zero (a = cax), where its derivative is infinite
        disc = (a_sq + ca_sq) ** 2 - 4.0 * a_sq * cax_sq
        is_pos = disc > 0.0
        root = jnp.where(is_pos, jnp.sqrt(jnp.where(is_pos, disc, 1.0)), 0.0)
        return jnp.sqrt(0.5 * (a_sq + ca_sq + root))

    cf_L = fast_speed(rho_L, P_L, B_sq_L, Bx_L)
    cf_R = fast_speed(rho_R, P_R, B_sq_R, Bx_R)
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

    return _flux_dict(flux_Mass, flux_Momx, flux_Momt, flux_Energy, flux_Bt)


# HLLD Riemann solver
def get_flux_hlld(WL, WR, gamma):
    """
    Calculate fluxes between 2 states with HLLD Riemann solver
    """

    epsilon = 1.0e-8

    rho_L, vx_L, P_L, Bx_L = WL["rho"], WL["vx"], WL["P"], WL["Bx"]
    rho_R, vx_R, P_R, Bx_R = WR["rho"], WR["vx"], WR["P"], WR["Bx"]

    has_out = "vz" in WL
    vt_L = (WL["vy"], WL["vz"]) if has_out else (WL["vy"],)
    vt_R = (WR["vy"], WR["vz"]) if has_out else (WR["vy"],)
    Bt_L = (WL["By"], WL["Bz"]) if has_out else (WL["By"],)
    Bt_R = (WR["By"], WR["Bz"]) if has_out else (WR["By"],)

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

    spd1 = jnp.minimum(vx_L, vx_R) - cfmax
    spd5 = jnp.maximum(vx_L, vx_R) + cfmax

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
    # The branches below are chosen with jnp.where, never by multiplying with a
    # mask: the branch not taken may be infinite (0 * inf = nan), and its
    # derivative is propagated too. Denominators that vanish where a branch is
    # not taken are replaced by 1 there.
    denom = rho_L * sdl * sdml - Bxsq
    isDegenL = jnp.abs(denom) < epsilon * Bxsq
    denom = jnp.where(isDegenL, 1.0, denom)

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdl - sdml) / denom
    ULst_Mt = tuple(
        jnp.where(isDegenL, ULst_d * v, ULst_d * (v - B * tmp))
        for v, B in zip(vt_L, Bt_L)
    )

    # eqns (45) and (47) of M&K
    tmp = (rho_L * (sdl) ** 2 - Bxsq) / denom
    ULst_Bt = tuple(jnp.where(isDegenL, B, B * tmp) for B in Bt_L)

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
    denom = rho_R * sdr * sdmr - Bxsq
    isDegenR = jnp.abs(denom) < epsilon * Bxsq
    denom = jnp.where(isDegenR, 1.0, denom)

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdr - sdmr) / denom
    URst_Mt = tuple(
        jnp.where(isDegenR, URst_d * v, URst_d * (v - B * tmp))
        for v, B in zip(vt_R, Bt_R)
    )

    # eqns (45) and (47) of M&K
    tmp = (rho_R * (sdr) ** 2 - Bxsq) / denom
    URst_Bt = tuple(jnp.where(isDegenR, B, B * tmp) for B in Bt_R)

    vbstr = (URst_Mx * Bxi + dot(URst_Mt, URst_Bt)) / URst_d
    # eqn (48) of M&K
    URst_E = (
        sdr * E_R
        - ptr * vx_R
        + ptst * spd3
        + Bxi * (vx_R * Bxi + dot(vt_R, Bt_R) - vbstr)
    ) / sdmr

    WRst_vt = tuple(M / URst_d for M in URst_Mt)

    # Ul** and Ur**: if Bx is zero, the same as the *-states (the comparison
    # holds, rather than giving 0/0, when the field vanishes altogether)
    isDegen = 0.5 * Bxsq <= epsilon**2 * jnp.minimum(pbl, pbr)

    # the density and normal momentum do not change across the Alfven waves
    ULdst_d, URdst_d = ULst_d, URst_d
    ULdst_Mx, URdst_Mx = ULst_Mx, URst_Mx

    invsumd = 1.0 / (sqrtdl + sqrtdr)
    Bxsig = jnp.sign(Bxi)

    # eqn (59) of M&K
    tmp_v = tuple(
        invsumd * (sqrtdl * wl + sqrtdr * wr + Bxsig * (bR - bL))
        for wl, wr, bR, bL in zip(WLst_vt, WRst_vt, URst_Bt, ULst_Bt)
    )
    ULdst_Mt = tuple(jnp.where(isDegen, M, ULdst_d * t) for M, t in zip(ULst_Mt, tmp_v))
    URdst_Mt = tuple(jnp.where(isDegen, M, URdst_d * t) for M, t in zip(URst_Mt, tmp_v))

    # eqn (61) of M&K
    tmp_b = tuple(
        invsumd * (sqrtdl * bR + sqrtdr * bL + Bxsig * sqrtdl * sqrtdr * (wr - wl))
        for bR, bL, wl, wr in zip(URst_Bt, ULst_Bt, WLst_vt, WRst_vt)
    )
    ULdst_Bt = tuple(jnp.where(isDegen, B, t) for B, t in zip(ULst_Bt, tmp_b))
    URdst_Bt = tuple(jnp.where(isDegen, B, t) for B, t in zip(URst_Bt, tmp_b))

    # eqn (63) of M&K
    tmp = spd3 * Bxi + dot(ULdst_Mt, ULdst_Bt) / ULdst_d
    ULdst_E = jnp.where(isDegen, ULst_E, ULst_E - sqrtdl * Bxsig * (vbstl - tmp))
    URdst_E = jnp.where(isDegen, URst_E, URst_E + sqrtdr * Bxsig * (vbstr - tmp))

    # Step 7
    # Compute flux

    # Which of the six regions the interface sits in. These have to be tested
    # as a chain, each only where the ones before it fail: when the normal
    # field vanishes the Alfven speeds collapse onto the contact
    # (spd2 = spd3 = spd4), and independent conditions would then select two
    # regions at once.
    tmpl = spd2 - spd1
    tmpr = spd4 - spd5

    def assemble(FL, FR, UL, UR, ULst, URst, ULdst, URdst):
        return jnp.where(
            spd1 >= 0,
            FL,
            jnp.where(
                spd5 <= 0,
                FR,
                jnp.where(
                    spd2 >= 0,
                    FL + spd1 * (ULst - UL),
                    jnp.where(
                        spd3 >= 0,
                        FL - spd1 * UL - tmpl * ULst + spd2 * ULdst,
                        jnp.where(
                            spd4 > 0,
                            FR - spd5 * UR - tmpr * URst + spd4 * URdst,
                            FR + spd5 * (URst - UR),
                        ),
                    ),
                ),
            ),
        )

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

    return _flux_dict(flux_Mass, flux_Momx, flux_Momt, flux_Energy, flux_Bt)


def get_flux(WL, WR, gamma, riemann_solver):
    if riemann_solver == "hlld":
        return get_flux_hlld(WL, WR, gamma)
    else:
        # default
        return get_flux_llf(WL, WR, gamma)


def hydro_mhd2d_timestep(
    W, gamma, dx, dy, bc_x="periodic", bc_y="periodic", ghost_x=None, ghost_y=None
):
    """
    Calculate the simulation timestep based on CFL condition
    """

    rho, vx, vy, P = W["rho"], W["vx"], W["vy"], W["P"]
    Bx, By = get_cell_centered_field(W["bx"], W["by"], bc_x, bc_y, ghost_x, ghost_y)
    v_sq = vx**2 + vy**2
    B_sq = Bx**2 + By**2
    if "vz" in W:
        v_sq = v_sq + W["vz"] ** 2
        B_sq = B_sq + W["bz"] ** 2

    c_s_sq = gamma * (P - 0.5 * B_sq) / rho
    v_a_sq = B_sq / rho

    # get time step (CFL) = dx / max signal speed
    dl = jnp.minimum(dx, dy)
    dt = jnp.min(dl / (jnp.sqrt(v_sq) + jnp.sqrt(c_s_sq + v_a_sq)))

    return dt


def half_step(W, W_dx, W_dy, gamma, geom, dt):
    """
    Extrapolate the cell-centered primitive fields half a timestep in time
    """

    is_cylindrical = geom["is_cylindrical"]
    r = geom["r"]
    use_out_of_plane = "vz" in W
    Bx, By = W["Bx"], W["By"]

    rho, vx, vy, P = W["rho"], W["vx"], W["vy"], W["P"]
    rho_dx, vx_dx, vy_dx, P_dx = (W_dx[k] for k in ("rho", "vx", "vy", "P"))
    rho_dy, vx_dy, vy_dy, P_dy = (W_dy[k] for k in ("rho", "vx", "vy", "P"))
    Bx_dx, By_dx, Bx_dy, By_dy = W_dx["Bx"], W_dx["By"], W_dy["Bx"], W_dy["By"]

    div_v = vx_dx + vy_dy
    if is_cylindrical:
        div_v = div_v + vx / r

    Wp = {}
    Wp["rho"] = rho - 0.5 * dt * (vx * rho_dx + vy * rho_dy + rho * div_v)
    Wp["vx"] = vx - 0.5 * dt * (
        vx * vx_dx
        + vy * vx_dy
        + (1.0 / rho) * P_dx
        - (2.0 * Bx / rho) * Bx_dx
        - (By / rho) * Bx_dy
        - (Bx / rho) * By_dy
    )
    Wp["vy"] = vy - 0.5 * dt * (
        vx * vy_dx
        + vy * vy_dy
        + (1.0 / rho) * P_dy
        - (2.0 * By / rho) * By_dy
        - (Bx / rho) * By_dx
        - (By / rho) * Bx_dx
    )
    B_sq = Bx**2 + By**2
    if use_out_of_plane:
        vz, bz = W["vz"], W["Bz"]
        vz_dx, vz_dy, Bz_dx, Bz_dy = W_dx["vz"], W_dy["vz"], W_dx["Bz"], W_dy["Bz"]
        B_sq = B_sq + bz**2
    Wp["P"] = P - 0.5 * dt * (
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
        Wp["P"] = Wp["P"] - 0.5 * dt * (
            bz**2 * (vx_dx + vy_dy) - bz * Bx * vz_dx - bz * By * vz_dy
        )

    Wp["Bx"] = Bx - 0.5 * dt * (-By * vx_dy + Bx * vy_dy + vy * Bx_dy - vx * By_dy)
    Wp["By"] = By - 0.5 * dt * (By * vx_dx - Bx * vy_dx - vy * Bx_dx + vx * By_dx)
    if use_out_of_plane:
        # d(vz)/dt = (B.grad) Bz / rho; in cylindrical the azimuthal equation
        # also carries -v_R v_phi / R and +B_R B_phi / (rho R)
        Wp["vz"] = vz - 0.5 * dt * (
            vx * vz_dx + vy * vz_dy - (Bx / rho) * Bz_dx - (By / rho) * Bz_dy
        )
        if is_cylindrical:
            Wp["vz"] = Wp["vz"] - 0.5 * dt * (vx * vz / r - Bx * bz / (rho * r))
        # d(Bz)/dt = -div(vx Bz - vz Bx, vy Bz - vz By), using div(B) = 0
        Wp["Bz"] = bz - 0.5 * dt * (
            vx * Bz_dx + vy * Bz_dy + bz * (vx_dx + vy_dy) - Bx * vz_dx - By * vz_dy
        )

    # the terms above are written with cartesian divergences; in cylindrical
    # geometry each picks up a 1/R term, and v_R feels the centrifugal force
    # and the hoop stress. The force of B_phi on v_R is written as
    # -(B_phi / rho R) d(R B_phi)/dR, from the slope of R B_phi ('RBz'), which
    # vanishes in a current-free field even where the slopes are limited
    if is_cylindrical:
        Wp["vx"] = Wp["vx"] + 0.5 * dt * Bx**2 / (rho * r)
        Wp["vy"] = Wp["vy"] + 0.5 * dt * Bx * By / (rho * r)
        Wp["P"] = (
            Wp["P"]
            - 0.5
            * dt
            * ((Bx**2 + By**2) * vx + (gamma - 2.0) * (Bx * vx + By * vy) * Bx)
            / r
        )
        Wp["By"] = Wp["By"] - 0.5 * dt * (vx * By - vy * Bx) / r
        if use_out_of_plane:
            Wp["vx"] = Wp["vx"] + 0.5 * dt * (
                vz**2 / r + (bz / rho) * (Bz_dx - W_dx["RBz"] / r)
            )
            Wp["P"] = Wp["P"] - 0.5 * dt * Bx * bz * vz / r
            Wp["Bz"] = Wp["Bz"] - 0.5 * dt * vz * Bx / r

    return Wp


def hydro_mhd2d_fluxes(
    W,
    geom,
    dt,
    *,
    gamma,
    riemann_solver="llf",
    slope_limiting=False,
    bc_x="periodic",
    bc_y="periodic",
    ghost_x=None,
    ghost_y=None,
):
    """
    Take a simulation timestep

    W holds the primitive fields 'rho', 'vx', 'vy', 'P' (total pressure), the
    face-centered 'bx', 'by', and, for 2.5D, the out-of-plane 'vz', 'bz'. A
    'driven' boundary takes its ghost cells from ghost_x/ghost_y, which map
    each field to its (lo, hi) ghost values.
    """

    dx = geom["dx"]
    dy = geom["dy"]
    is_cylindrical = geom["is_cylindrical"]
    r = geom["r"]
    use_out_of_plane = "vz" in W
    x_has_ghosts = bc_x != "periodic"
    y_has_ghosts = bc_y != "periodic"

    if x_has_ghosts:
        W = add_ghost_cells(W, 0, bc_x, ghost_x)
    if y_has_ghosts:
        W = add_ghost_cells(W, 1, bc_y, ghost_y)

    # get Conserved variables
    bx, by = W["bx"], W["by"]
    Bx, By = get_avg(bx, by)
    Wc = {
        "rho": W["rho"],
        "vx": W["vx"],
        "vy": W["vy"],
        "P": W["P"],
        "Bx": Bx,
        "By": By,
    }
    if use_out_of_plane:
        Wc["vz"], Wc["Bz"] = W["vz"], W["bz"]
    U = get_conserved(Wc, gamma, geom)

    # calculate gradients
    W_d = dict(Wc)
    if is_cylindrical and use_out_of_plane:
        W_d["RBz"] = r * Wc["Bz"]
    W_dx, W_dy = get_gradients(W_d, dx, dy, slope_limiting)
    if x_has_ghosts:
        W_dx = set_ghost_gradients(W_dx, 0, bc_x)
    if y_has_ghosts:
        W_dy = set_ghost_gradients(W_dy, 1, bc_y)

    # extrapolate half-step in time
    Wp = half_step(Wc, W_dx, W_dy, gamma, geom, dt)
    Wp = reset_ghost_cells(Wp, Wc, 0, bc_x)
    Wp = reset_ghost_cells(Wp, Wc, 1, bc_y)

    # extrapolate in space to face centers, and compute fluxes
    W_XL, W_XR, W_YL, W_YR = face_states(Wp, W_dx, W_dy, dx, dy)
    flux_X = get_flux(W_XL, W_XR, gamma, riemann_solver)
    flux_Y = swap_xy(get_flux(swap_xy(W_YL), swap_xy(W_YR), gamma, riemann_solver))

    # update solution
    area_x = geom["area_x"]
    area_y = geom["area_y"]
    for name in ("mass", "momx", "momy", "energy"):
        U[name] = apply_fluxes(U[name], flux_X[name], flux_Y[name], area_x, area_y, dt)
    if use_out_of_plane:
        if is_cylindrical:
            # angular momentum: the flux is weighted by the shared face radius,
            # and already contains the Maxwell stress -R B_R B_phi
            U["momz"] = apply_fluxes(
                U["momz"],
                geom["r_face_x"] * flux_X["momz"],
                r * flux_Y["momz"],
                area_x,
                area_y,
                dt,
            )
        else:
            U["momz"] = apply_fluxes(
                U["momz"], flux_X["momz"], flux_Y["momz"], area_x, area_y, dt
            )
        U["Bz"] = apply_fluxes(U["Bz"], flux_X["Bz"], flux_Y["Bz"], dy, dx, dt)
    bx, by = constrained_transport(bx, by, flux_X["By"], flux_Y["Bx"], dx, dy, dt, geom)

    # geometric source terms in the radial momentum
    if is_cylindrical:
        U["momx"] = U["momx"] + dt * Wp["P"] * geom["d_area_x"]
        if use_out_of_plane:
            # centrifugal force and the magnetic hoop stress
            U["momx"] = (
                U["momx"]
                + dt * geom["vol"] * (Wp["rho"] * Wp["vz"] ** 2 - Wp["Bz"] ** 2) / r
            )

    # remove ghost cells
    for axis, has_ghosts in ((0, x_has_ghosts), (1, y_has_ghosts)):
        if has_ghosts:
            U = {name: strip_ghosts(f, axis) for name, f in U.items()}
            bx, by = strip_ghosts(bx, axis), strip_ghosts(by, axis)

    # get Primitive variables, with the boundary faces set as at the start of
    # the step, so that the total pressure is consistent with the next step
    Bx, By = get_cell_centered_field(bx, by, bc_x, bc_y, ghost_x, ghost_y)
    W = get_primitive(U, Bx, By, gamma, geom_strip(geom, x_has_ghosts))
    W["bx"], W["by"] = bx, by

    return W
