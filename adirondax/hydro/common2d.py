import jax.numpy as jnp

# Pure functions for 2D hydrodynamics


def get_curl(Az, dx):
    """
    Calculate the discrete curl
    """

    bx = (Az - jnp.roll(Az, 1, axis=1)) / dx  # = d Az / d y
    by = -(Az - jnp.roll(Az, 1, axis=0)) / dx  # =-d Az / d x

    return bx, by


def get_div(bx, by, dx):
    """
    Calculate the discrete divergence
    """

    div_B = (bx - jnp.roll(bx, 1, axis=0) + by - jnp.roll(by, 1, axis=1)) / dx

    return div_B


def get_avg(bx, by):
    """
    Calculate the volume-averaged magnetic field
    """

    Bx = 0.5 * (bx + jnp.roll(bx, 1, axis=0))
    By = 0.5 * (by + jnp.roll(by, 1, axis=1))

    return Bx, By


def get_gradient(f, dx):
    """Calculate the gradients of a field"""

    # (right - left) / (2*dx)
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)

    return f_dx, f_dy


def slope_limit(f, dx, f_dx, f_dy):
    """
    Apply slope limiter to slopes
    """

    f_dx = (
        jnp.maximum(
            0.0,
            jnp.minimum(
                1.0, ((f - jnp.roll(f, 1, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dx = (
        jnp.maximum(
            0.0,
            jnp.minimum(
                1.0,
                (-(f - jnp.roll(f, -1, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)),
            ),
        )
        * f_dx
    )
    f_dy = (
        jnp.maximum(
            0.0,
            jnp.minimum(
                1.0, ((f - jnp.roll(f, 1, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
            ),
        )
        * f_dy
    )
    f_dy = (
        jnp.maximum(
            0.0,
            jnp.minimum(
                1.0,
                (-(f - jnp.roll(f, -1, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)),
            ),
        )
        * f_dy
    )

    return f_dx, f_dy


def extrapolate_to_face(f, f_dx, f_dy, dx):
    """Extrapolate the field from face centers to faces using gradients"""

    f_XL = f - f_dx * dx / 2.0
    f_XL = jnp.roll(f_XL, -1, axis=0)  # right/up roll
    f_XR = f + f_dx * dx / 2.0

    f_YL = f - f_dy * dx / 2.0
    f_YL = jnp.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy * dx / 2.0

    return f_XL, f_XR, f_YL, f_YR


def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """
    Apply fluxes to conserved variables
    """
    F += -dt * dx * flux_F_X
    F += dt * dx * jnp.roll(flux_F_X, 1, axis=0)
    F += -dt * dx * flux_F_Y
    F += dt * dx * jnp.roll(flux_F_Y, 1, axis=1)

    return F
