import jax.numpy as jnp

from .boundary import add_ghost_cells, set_ghost_gradients, strip_ghosts
from .common2d import apply_fluxes, face_states, get_gradients, swap_xy
from .geometry import geom_strip

# Pure functions for 2D Euler hydrodynamics


def get_conserved(W, gamma, geom):
    """Calculate the conserved variables from the primitive variables"""

    vol = geom["vol"]
    rho, vx, vy, P = W["rho"], W["vx"], W["vy"], W["P"]

    v_sq = vx**2 + vy**2
    if "vz" in W:
        v_sq = v_sq + W["vz"] ** 2

    U = {
        "mass": rho * vol,
        "momx": rho * vx * vol,
        "momy": rho * vy * vol,
        "energy": (P / (gamma - 1.0) + 0.5 * rho * v_sq) * vol,
    }
    if "vz" in W:
        U["momz"] = rho * geom["r"] * W["vz"] * vol

    return U


def get_primitive(U, gamma, geom):
    """Calculate the primitive variable from the conserved variables"""

    vol = geom["vol"]

    rho = U["mass"] / vol
    vx = U["momx"] / rho / vol
    vy = U["momy"] / rho / vol
    W = {"rho": rho, "vx": vx, "vy": vy}

    v_sq = vx**2 + vy**2
    if "momz" in U:
        # the centroid radius is strictly positive, even in the cell on the axis
        W["vz"] = U["momz"] / (U["mass"] * geom["r"])
        v_sq = v_sq + W["vz"] ** 2

    W["P"] = (U["energy"] / vol - 0.5 * rho * v_sq) * (gamma - 1.0)

    return W


def get_flux_llf(WL, WR, gamma):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    rho_L, vx_L, vy_L, P_L = WL["rho"], WL["vx"], WL["vy"], WL["P"]
    rho_R, vx_R, vy_R, P_R = WR["rho"], WR["vx"], WR["vy"], WR["P"]
    has_rotation = "vz" in WL

    # left and right energies
    v_sq_L = vx_L**2 + vy_L**2
    v_sq_R = vx_R**2 + vy_R**2
    if has_rotation:
        vz_L, vz_R = WL["vz"], WR["vz"]
        v_sq_L = v_sq_L + vz_L**2
        v_sq_R = v_sq_R + vz_R**2
    en_L = P_L / (gamma - 1.0) + 0.5 * rho_L * v_sq_L
    en_R = P_R / (gamma - 1.0) + 0.5 * rho_R * v_sq_R

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)

    mom_sq_star = momx_star**2 + momy_star**2
    if has_rotation:
        momz_star = 0.5 * (rho_L * vz_L + rho_R * vz_R)
        mom_sq_star = mom_sq_star + momz_star**2

    P_star = (gamma - 1.0) * (en_star - 0.5 * mom_sq_star / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star

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

    flux = {
        "mass": flux_Mass,
        "momx": flux_Momx,
        "momy": flux_Momy,
        "energy": flux_Energy,
    }
    if has_rotation:
        flux_Momz = momx_star * momz_star / rho_star
        flux_Momz -= C * 0.5 * (rho_R * vz_R - rho_L * vz_L)
        flux["momz"] = flux_Momz

    return flux


def get_flux_hllc(WL, WR, gamma):
    """
    Calculate fluxes between 2 states with the HLLC approximate Riemann solver
    """

    has_rotation = "vz" in WL

    rho_l, u_l, v_l, p_l = WL["rho"], WL["vx"], WL["vy"], WL["P"]
    rho_r, u_r, v_r, p_r = WR["rho"], WR["vx"], WR["vy"], WR["P"]
    w_l = WL["vz"] if has_rotation else 0.0
    w_r = WR["vz"] if has_rotation else 0.0

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

    flux = dict(zip(("mass", "momx", "momy", "energy", "momz"), flux))
    if not has_rotation:
        del flux["momz"]

    return flux


def get_flux(WL, WR, gamma, riemann_solver):
    if riemann_solver == "hllc":
        return get_flux_hllc(WL, WR, gamma)
    else:
        # default
        return get_flux_llf(WL, WR, gamma)


def hydro_euler2d_timestep(W, gamma, dx, dy):
    """Calculate the simulation timestep based on CFL condition"""

    rho, vx, vy, P = W["rho"], W["vx"], W["vy"], W["P"]

    # get time step (CFL) = dx / max signal speed
    dl = jnp.minimum(dx, dy)
    dt = jnp.min(dl / (jnp.sqrt(gamma * P / rho) + jnp.sqrt(vx**2 + vy**2)))

    return dt


def hydro_euler2d_fluxes(
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

    W holds the primitive fields 'rho', 'vx', 'vy', 'P', and, with rotation,
    the azimuthal velocity 'vz'. A 'driven' boundary takes its ghost cells from
    ghost_x/ghost_y, which map each field to its (lo, hi) ghost values.
    """

    dx = geom["dx"]
    dy = geom["dy"]
    is_cylindrical = geom["is_cylindrical"]
    x_has_ghosts = bc_x != "periodic"
    y_has_ghosts = bc_y != "periodic"

    # Add Ghost Cells (if needed)
    if x_has_ghosts:
        W = add_ghost_cells(W, 0, bc_x, ghost_x)
    if y_has_ghosts:
        W = add_ghost_cells(W, 1, bc_y, ghost_y)

    # get Conserved variables
    U = get_conserved(W, gamma, geom)

    # calculate gradients
    W_dx, W_dy = get_gradients(W, dx, dy, slope_limiting)
    if x_has_ghosts:
        W_dx = set_ghost_gradients(W_dx, 0, bc_x)
    if y_has_ghosts:
        W_dy = set_ghost_gradients(W_dy, 1, bc_y)

    rho, vx, vy, P = W["rho"], W["vx"], W["vy"], W["P"]

    # velocity divergence picks up a geometric term in cylindrical geometry
    div_v = W_dx["vx"] + W_dy["vy"]
    if is_cylindrical:
        div_v = div_v + vx / geom["r"]

    # extrapolate half-step in time
    Wp = {
        "rho": rho - 0.5 * dt * (vx * W_dx["rho"] + vy * W_dy["rho"] + rho * div_v),
        "vx": vx
        - 0.5 * dt * (vx * W_dx["vx"] + vy * W_dy["vx"] + (1.0 / rho) * W_dx["P"]),
        "vy": vy
        - 0.5 * dt * (vx * W_dx["vy"] + vy * W_dy["vy"] + (1.0 / rho) * W_dy["P"]),
        "P": P - 0.5 * dt * (gamma * P * div_v + vx * W_dx["P"] + vy * W_dy["P"]),
    }
    if "vz" in W:
        vz = W["vz"]
        # in terms of the specific angular momentum R*vphi this is pure
        # advection; written for vphi it carries the -vx*vphi/R term
        Wp["vz"] = vz - 0.5 * dt * (
            vx * W_dx["vz"] + vy * W_dy["vz"] + vx * vz / geom["r"]
        )
        # centrifugal acceleration
        Wp["vx"] = Wp["vx"] + 0.5 * dt * vz**2 / geom["r"]

    # extrapolate in space to face centers, and compute fluxes
    W_XL, W_XR, W_YL, W_YR = face_states(Wp, W_dx, W_dy, dx, dy)
    flux_X = get_flux(W_XL, W_XR, gamma, riemann_solver)
    flux_Y = swap_xy(get_flux(swap_xy(W_YL), swap_xy(W_YR), gamma, riemann_solver))

    # update solution
    area_x = geom["area_x"]
    area_y = geom["area_y"]
    for name in ("mass", "momx", "momy", "energy"):
        U[name] = apply_fluxes(U[name], flux_X[name], flux_Y[name], area_x, area_y, dt)
    if "momz" in U:
        U["momz"] = apply_fluxes(
            U["momz"],
            geom["r_face_x"] * flux_X["momz"],
            geom["r"] * flux_Y["momz"],
            area_x,
            area_y,
            dt,
        )

    # geometric source terms (time-centered, from the half-step primitives)
    if is_cylindrical:
        U["momx"] = U["momx"] + dt * Wp["P"] * geom["d_area_x"]
        if "vz" in W:
            U["momx"] = (
                U["momx"] + dt * (Wp["rho"] * geom["vol"]) * Wp["vz"] ** 2 / geom["r"]
            )

    # remove ghost cells
    if x_has_ghosts:
        U = {name: strip_ghosts(f, 0) for name, f in U.items()}
    if y_has_ghosts:
        U = {name: strip_ghosts(f, 1) for name, f in U.items()}

    return get_primitive(U, gamma, geom_strip(geom, x_has_ghosts))


def hydro_euler2d_accelerate(W, ax, ay, gamma, geom, dt):
    U = get_conserved(W, gamma, geom)

    U["energy"] = U["energy"] + dt * (U["momx"] * ax + U["momy"] * ay)
    U["momx"] = U["momx"] + dt * U["mass"] * ax
    U["momy"] = U["momy"] + dt * U["mass"] * ay

    W_new = get_primitive(U, gamma, geom)
    return {**W, "vx": W_new["vx"], "vy": W_new["vy"], "P": W_new["P"]}
