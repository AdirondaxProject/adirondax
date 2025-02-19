import jax
import jax.numpy as jnp

# Pure functions for hydro simulation


@jax.jit
def get_conserved(rho, vx, vy, P, gamma, vol):
    """Calculate the conserved variables from the primitive variables"""

    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)) * vol

    return Mass, Momx, Momy, Energy


@jax.jit
def get_primitive(Mass, Momx, Momy, Energy, gamma, vol):
    """Calculate the primitive variable from the conserved variables"""

    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy / vol - 0.5 * rho * (vx**2 + vy**2)) * (gamma - 1.0)

    return rho, vx, vy, P


@jax.jit
def get_gradient(f, dx):
    """Calculate the gradients of a field"""

    # (right - left) / 2dx
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)

    return f_dx, f_dy


@jax.jit
def extrapolate_to_face(f, f_dx, f_dy, dx):
    """Extrapolate the field from face centers to faces using gradients"""

    f_XL = f - f_dx * dx / 2.0
    f_XL = jnp.roll(f_XL, -1, axis=0)  # right/up roll
    f_XR = f + f_dx * dx / 2.0

    f_YL = f - f_dy * dx / 2.0
    f_YL = jnp.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy * dx / 2.0

    return f_XL, f_XR, f_YL, f_YR


@jax.jit
def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """Apply fluxes to conserved variables to update solution state"""

    F += -dt * dx * flux_F_X
    F += dt * dx * jnp.roll(flux_F_X, 1, axis=0)  # left/down roll
    F += -dt * dx * flux_F_Y
    F += dt * dx * jnp.roll(flux_F_Y, 1, axis=1)

    return F


@jax.jit
def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    # left and right energies
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2)

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)

    P_star = (gamma - 1) * (en_star - 0.5 * (momx_star**2 + momy_star**2) / rho_star)

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


@jax.jit
def update_hydro(rho, vx, vy, P, vol, dx, gamma, dt):
    """Take a simulation timestep"""

    # get Conserved variables
    Mass, Momx, Momy, Energy = get_conserved(rho, vx, vy, P, gamma, vol)

    # get time step (CFL) = dx / max signal speed
    # dt = courant_fac * jnp.min(dx / (jnp.sqrt(gamma * P / rho) + jnp.sqrt(vx**2 + vy**2)))

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx)
    vx_dx, vx_dy = get_gradient(vx, dx)
    vy_dx, vy_dy = get_gradient(vy, dx)
    P_dx, P_dy = get_gradient(P, dx)

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
    vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1.0 / rho) * P_dx)
    vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1.0 / rho) * P_dy)
    P_prime = P - 0.5 * dt * (gamma * P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(rho_prime, rho_dx, rho_dy, dx)
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx_prime, vx_dx, vx_dy, dx)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy_prime, vy_dx, vy_dy, dx)
    P_XL, P_XR, P_YL, P_YR = extrapolate_to_face(P_prime, P_dx, P_dy, dx)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)

    rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, gamma, vol)

    return rho, vx, vy, P
