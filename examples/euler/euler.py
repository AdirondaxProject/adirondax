import jax
import jax.numpy as jnp
import time


@jax.jit
def get_gradient(f):
    """Calculate the gradients of a field"""

    f_dx = jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)
    f_dy = jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)

    return f_dx, f_dy


@jax.jit
def extrapolate_to_face(f, f_dx, f_dy):
    """Extrapolate the field from face centers to faces using gradients"""

    f_XL = f - f_dx
    f_XL = jnp.roll(f_XL, -1, axis=0)
    f_XR = f + f_dx

    f_YL = f - f_dy
    f_YL = jnp.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy

    return f_XL, f_XR, f_YL, f_YR


@jax.jit
def apply_fluxes(F, flux_F_X, flux_F_Y):
    """Apply fluxes to conserved variables to update solution state"""

    F += -flux_F_X
    F += jnp.roll(flux_F_X, 1, axis=0)
    F += -flux_F_Y
    F += jnp.roll(flux_F_Y, 1, axis=1)

    return F


@jax.jit
def get_flux(A_L, A_R, B_L, B_R):
    """Calculate fluxes between 2 states"""

    A_star = 0.5 * (A_L + A_R)
    B_star = 0.5 * (B_L + B_R)

    flux_A = B_star
    flux_B = B_star**2 / A_star

    flux_A -= 0.1 * (A_L - A_R)
    flux_B -= 0.1 * (B_L - B_R)

    return flux_A, flux_B


# @jax.jit  # <---  XXX Adding this line slows down the code a lot!!
def update(A, B):
    """Take a simulation timestep"""

    A_dx, A_dy = get_gradient(A)
    B_dx, B_dy = get_gradient(B)

    A_XL, A_XR, A_YL, A_YR = extrapolate_to_face(A, A_dx, A_dy)
    B_XL, B_XR, B_YL, B_YR = extrapolate_to_face(B, B_dx, B_dy)

    flux_A_X, flux_B_X = get_flux(A_XL, A_XR, B_XL, B_XR)
    flux_A_Y, flux_B_Y = get_flux(A_YL, A_YR, B_YL, B_YR)

    A = apply_fluxes(A, flux_A_X, flux_A_Y)
    B = apply_fluxes(B, flux_B_X, flux_B_Y)

    return A, B


@jax.jit
def update_compiled_SLOW(A, B):
    return update(A, B)


def main():

    N = 1024

    A = jnp.ones((N, N))
    B = jnp.ones((N, N))
    tic = time.time()
    for _ in range(200):
        (
            A,
            B,
        ) = update(A, B)
    print("Total time not compiled: ", time.time() - tic)

    A = jnp.ones((N, N))
    B = jnp.ones((N, N))
    tic = time.time()
    for _ in range(200):
        A, B = update_compiled_SLOW(A, B)
    print("Total time compiled: ", time.time() - tic)


if __name__ == "__main__":
    main()
